"""
TODO
- Consider speeding up training function by replacing choice with epochs and permutation
- Make sure mask_length is consistent with trained model (it currently isn't)
"""
import tensorflow as tf
import yaml
import path
from loaders.loader import LoadObjDNV, LoadObjSC
from loaders.loader_parquet import LoaderDS
import numpy as np
from models import SequenceHead, Encoder, ClassifierHead, DenovoDecoder
import os
from tqdm import tqdm
from collections import deque
from time import time
from utils import AccRecPrec
cross_entropy = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, reduction='none'
)
choice = np.random.choice

class DownstreamObj:
    def __init__(self, config, task='denovo', base_model=None):
        
        self.config = config
        self.task = task
        assert task in config.keys()
        # Optimizer(s)
        self.opt_head = tf.keras.optimizers.Adam(learning_rate=config['lr'])
        self.opt_encoder = tf.keras.optimizers.Adam(learning_rate=config['lr'])

        # Base model
        # - must do base model beforehand dataloader in order to transfer over 
        #   its configuration settings to self.config
        self.configure_encoder(base_model) # self.config updated
        #if not config['train_encoder']: self.encoder.trainable = False

        self.global_step = 0
    
    def configure_encoder(self, imported_encoder=None):
        
        self.imported = True if imported_encoder is not None else False
        
        # If encoder model passed in as argument OR no saved weights
        if self.imported or (self.config['pretrain_path'] is None):
            # Get configuration settings from current pretrain yaml files
            yaml_config_path = './yaml/config.yaml'
            yaml_model_path = './yaml/models.yaml'
        
        # If encoder is loaded from saved weights
        else:
            # Get configuration settings from saved experiment
            assert os.path.exists(self.config['pretrain_path'])
            yaml_config_path = self.config['pretrain_path']+'/yaml/config.yaml'
            yaml_model_path = self.config['pretrain_path']+'/yaml/models.yaml'
        
        # Open yaml files
        with open(yaml_config_path) as stream:
            ptconf = yaml.safe_load(stream)
        with open(yaml_model_path) as stream:
            ptmodconf = yaml.safe_load(stream)
        # Transfer over settings to self.config
        self.config[self.task]['loader']['top_pks'] = ptconf['max_peaks']
        self.config['encoder_dict'] = ptmodconf['encoder_dict']
        # Set self.encoder
        self.encoder = (
            imported_encoder 
            if imported_encoder is not None else
            Encoder(**self.config['encoder_dict'])
        )
    
    def initialize_weights(self, special=False):
        
        # Next 13 lines establish initial weights in head (and maybe encoder)
        indices = self.dl.inds['train'][:self.config['batch_size']]
        batch = self.dl.load_batch(indices)
        mzab = tf.concat([batch['mz'][...,None], batch['ab'][...,None]], -1)
        model_inp = {
            'x': mzab,
            'charge': batch['charge'],
            'mass': batch['mass'],
            'length': batch['length'],
            'training': False
        }
        out = self.encoder(**model_inp) # forward pass in eval mode
        
        # Special function for inference if special, else just standard
        # encoder_embedding as input to call function
        if special:
            self.head.initialize_variables(batch, out['emb'])
        else:
            head_out = self.head(out['emb'], training=False)
        self.opt_head.build(self.head.trainable_variables)
        # If the encoder is imported, assume this is pretraining and encoder is
        # already trained, i.e. do not change its weights in this program.
        # ELSE: change its weights all you desire
        if self.imported == False:
            if self.config['pretrain_path'] is not None:
                self.encoder.load_weights(
                    self.config['pretrain_path'] + '/weights/model_encoder.wts'
                )
            # This option will be true if you plan to allow the encoder weights
            # to float during downstream training; else just train the head.
            if self.config['train_encoder']:
                self.initialize_encoder_optimizer_weights()

    def initialize_encoder_optimizer_weights(self):
        self.opt_encoder.build(self.encoder.trainable_variables)

    def save_head(self, fp='./head.wts'):
        self.head.save_weights(fp)
    
    def save_encoder(self, fp='./encoder.wts'):
        self.encoder.save_weights(fp)

    def save_all_weights(self, fp='./'):
        self.save_head(fp=fp+'head.wts')
        self.save_encoder(fp=fp+'encoder.wts')

    def split_labels_str(self, incl_str):
        return [label for label in self.dl.labels if incl_str in label]

    def encinp(self, batch, mask_length=True, training=False):
        mzab = tf.concat([batch['mz'][...,None], batch['ab'][...,None]], -1)
        model_inp = {
            'x': mzab,
            'charge': (
                batch['charge'] 
                if self.config['encoder_dict']['use_charge'] else 
                None
            ),
            'mass': (
                batch['mass']
                if self.config['encoder_dict']['use_mass'] else
                None
            ),
            'length': batch['length'] if mask_length else None,
            'training': True if (self.config['train_encoder'] and training) else False
        }

        return model_inp
    
    def call(self, enc_inp_dict, training=False):
        enc_inp_dict['training'] = training
        embedding = self.encoder(**enc_inp_dict)['emb']
        out = self.head(embedding, training=training)

        return out
    
    def LossFunction(self, target, prediction):
        targ_one_hot = tf.one_hot(target, self.predcats)
        all_loss = cross_entropy(targ_one_hot, prediction)

        return all_loss
    
    def train_step(self, batch, trenc=True):
        enc_input, target = self.inptarg(batch, training=True)
        
        if trenc==False:
            embedding = self.encoder(**enc_input)['emb']
        with tf.GradientTape(persistent=True) as tape:
            if trenc==True:
                embedding = self.encoder(**enc_input)['emb']
            head_out = self.head(embedding, training=True)
            all_loss = self.LossFunction(target, head_out)
            loss = tf.reduce_mean(all_loss)

        grads = tape.gradient(loss, self.head.trainable_variables)
        self.opt_head.apply_gradients(zip(grads, self.head.trainable_variables))
        if trenc:
            grads = tape.gradient(loss, self.encoder.trainable_variables)
            self.opt_encoder.apply_gradients(zip(grads, self.encoder.trainable_variables))

        return loss

    def train_epoch(self, SeqInts=False):
        graph = (
            self.train_step
            if self.config['debug'] else 
            tf.function(self.train_step)
        )
        bs = self.config['batch_size']
        spe = self.config['steps_per_epoch']
        running_loss = deque(maxlen=50)
        
        T = tqdm(range(spe))
        perm = np.random.choice(self.dl.inds['train'], spe*bs, replace=True)
        for step in T:
            inds = perm[step*bs : (step+1)*bs]
            batch = self.dl.load_batch(inds, 'train', SeqInts=SeqInts)
        
            # Are we training the encoder? Two conditions must be met.
            train_encoder = (
                True 
                if (
                    self.config['train_encoder'] and 
                    (self.global_step>self.config['encoder_start'])
                ) else 
                False
            ) # boolean argument into train_step (graph)
            loss = graph(batch, train_encoder)
            self.global_step += 1
            running_loss.append(loss.numpy())
            
            if step%50==0:
                T.set_description("Loss: %.6f"%np.mean(running_loss), refresh=True)

class BaseDenovo(DownstreamObj):
    def __init__(self, config, base_model=None, ar=False):
        super().__init__(config=config, base_model=base_model)
        self.ar = ar
        """
        func = self.head.predict_sequence if self.ar else self.call
        self.egraph = (
            func
            if self.config['debug'] else
            tf.function(func)
        )
        """

    def evaluation(self, dset='val'):
        func = self.head.predict_sequence if self.ar else self.call
        graph = (
            func
            if self.config['debug'] else
            tf.function(func)
        )
        totsz = self.dl.dfs[dset].shape[0]
        steps = totsz // self.config['batch_size']
        steps += 0 if (totsz % self.config['batch_size'])==0 else 1
        
        out = {'ce': 0, 'accuracy': 0, 'recall': 0, 'precision': 0}
        tots = {
            'accuracy': {'sum': 0,'total': 0},
            'recall': {'sum': 0,'total': 0},
            'precision': {'sum': 0,'total': 0},
        }
        for step in tqdm(range(steps)):
            first = step*self.config['batch_size']
            last = np.minimum((step+1)*self.config['batch_size'], totsz)
            batch_inds = np.arange(first, last, 1)
            batch = self.dl.load_batch(batch_inds, dset=dset, SeqInts=True)
            
            # Fork in the code for the 2 types of denovo models I created
            if self.ar:
                enc_input, seqint, target = self.inptarg(
                    batch, training=False, full_seqint=True
                )
                embedding = self.encoder(**enc_input)
                prediction = graph(embedding, batch)
            else:
                enc_input, target = self.inptarg(batch, training=False)
                pred = graph(enc_input, training=False) # logits
                prediction = tf.cast(tf.argmax(pred, axis=-1), tf.int32) # bs, sl
            
            out['ce'] += (
                0 if self.ar else 
                tf.reduce_sum(self.LossFunction(target, pred))
            )
            
            stats = AccRecPrec(target, prediction, self.dl.amod_dic['X'])
            for metric in stats.keys():
                for key, val in stats[metric].items():
                    tots[metric][key] += val
        
        out['ce'] = out['ce'] / (totsz * self.config['sl'])
        for metric in stats.keys():
            out[metric] = tots[metric]['sum'] / tots[metric]['total']

        return out

    def TrainEval(self, eval_dset='val'):
        start_time = time()
        lines = []
        for i in range(self.config['epochs']):
            self.train_epoch(SeqInts=True) # Notice: SeqInts is true
            out = self.evaluation(dset=eval_dset)
            line = "ValEpoch %d: Cross-entropy=%.6f, Accuracy=%.6f, Recall=%.6f, Precision=%.6f"%(
                (i,) + tuple(out.values())
            )
            line += " (%.1f s)"%(time()-start_time)
            lines.append(line)
            print(line)
        
        return lines

class DenovoArDSObj(BaseDenovo):
    def __init__(self, config, base_model=None):
        super().__init__(config=config, base_model=base_model, ar=True)

        task = 'denovo_ar'
        
        # Dataloader
        self.dl = LoaderDS(self.config[task]['loader'])
        self.predcats = len(self.dl.amod_dic)

        # Head model
        head_dict = self.config[task]['head_dict']
        self.config['sl'] = self.config[task]['loader']['pep_length'][1]
        self.head = DenovoDecoder(
            token_dict=self.dl.amod_dic, dec_config=head_dict, 
            encoder=base_model
        )
        
        self.initialize_weights(special=True)

    def inptarg(self, batch, training=True, full_seqint=False):

        enc_input = self.encinp(batch, training=training)
        
        # Take the variable batch['seqint'] and add a start token to the 
        # beginning and null on the end
        intseq = self.head.prepend_startok(batch['seqint'][...,:-1])
        
        # Find the indices first null tokens so that when you choose random
        # token you avoid trivial trailing null tokens (beyoond final null)
        nonnull = tf.reduce_sum(
            tf.cast(intseq != self.head.inpdict['X'], tf.int32), axis=1
        )
        
        # Choose random tokens to predict
        # - the values of inds will be final non-hidden value in decoder input
        # - batch['seqint'](inds) will be the target for decoder output
        # - must use combination of uniform() and round() because int32 is not
        #   yet implemented when feeding vectors into low/high arguments
        inds = tf.random.uniform(
            (tf.shape(intseq)[0], ), 
            tf.fill((tf.shape(intseq)[0], ), -0.5), 
            tf.cast(nonnull, tf.float32) - 0.5
        )
        inds = tf.cast(tf.math.round(inds), tf.int32)

        # Set the predict token and fill with hidden tokens to the end
        # - this will be the decoder's input
        #dec_inp = self.head.set_tokens(intseq, inds2, self.head.pred_token)
        dec_inp = self.head.fill_hidden(intseq, inds)
        
        # Indices of chosen predict tokens
        # - save for LossFunction
        all_inds = tf.tile(
            tf.range(tf.shape(intseq)[1], dtype=tf.int32)[None],
            [tf.shape(intseq)[0], 1]
        )
        inds_ = tf.where(all_inds <= inds[:,None])
        #inds_ = tf.concat(
        #    [tf.range(tf.shape(inds)[0], dtype=tf.int32)[:,None], inds[:,None]], 
        #    axis=1
        #)
        self.inds = inds_

        # Target is the actual (intseq) identity of the chosen predict indices
        #intseq_ = self.head.append_nulltok(batch['seqint'])
        targ = batch['seqint'] if full_seqint else tf.gather_nd(batch['seqint'], inds_)

        return enc_input, dec_inp, targ

    def LossFunction(self, target, decout):
        targ_one_hot = tf.one_hot(target, self.predcats)
        logits = tf.gather_nd(decout, self.inds)
        loss = cross_entropy(targ_one_hot, logits)

        return loss

    def train_step(self, batch, trenc=True):
        enc_input, seqint, target = self.inptarg(batch, training=True)
        
        if trenc==False:
            embedding = self.encoder(**enc_input)
        with tf.GradientTape(persistent=True) as tape:
            if trenc==True:
                embedding = self.encoder(**enc_input)
            head_out = self.head(seqint, embedding, batch, training=True)
            all_loss = self.LossFunction(target, head_out)
            loss = tf.reduce_mean(all_loss)

        grads = tape.gradient(loss, self.head.trainable_variables)
        self.opt_head.apply_gradients(zip(grads, self.head.trainable_variables))
        if trenc:
            grads = tape.gradient(loss, self.encoder.trainable_variables)
            self.opt_encoder.apply_gradients(zip(grads, self.encoder.trainable_variables))

        return loss

class DenovoDSObj(BaseDenovo):
    def __init__(self, config, base_model=None):
        super().__init__(config=config, base_model=base_model)

        # fork in the code for task
        task = 'denovo'
        head_dict = self.config['denovo']['head_dict']  
        # Dataloader
        self.dl = LoaderDS(self.config[task]['loader'])
        
        # Head model
        # Place values into head dictionary that can't be determined beforehand 
        self.config['sl'] = self.config[task]['loader']['pep_length'][1]# + 1
        head_dict['final_seq_len'] = self.config['sl']
        self.predcats = len(self.dl.amod_dic)
        head_dict['final_units'] = len(self.dl.amod_dic)
        self.head = SequenceHead(**head_dict)
        
        self.initialize_weights()
    
    def inptarg(self, batch, training=True):
        enc_input = self.encinp(batch, training=training)
        target = batch['seqint']

        return enc_input, target
    
class AttributeDSObj(DownstreamObj):
    def __init__(self, config, base_model=None):
        super().__init__(config=config, base_model=base_model)

    def evaluation(self, dset='val'):
        graph = (
            self.call
            if self.config['debug'] else
            tf.function(self.call)
        )
        totsz = self.dl.dfs[dset].shape[0]
        steps = totsz // self.config['batch_size']
        steps += 0 if (totsz % self.config['batch_size'])==0 else 1

        out = {'ce': 0, 'accuracy': 0,}
        tots = {
            'ce': {'sum': 0, 'total': 0},
            'accuracy': {'sum': 0,'total': 0},
        }
        for step in tqdm(range(steps)):
            first = step*self.config['batch_size']
            last = np.minimum((step+1)*self.config['batch_size'], totsz)
            batch_inds = np.arange(first, last, 1)
            batch = self.dl.load_batch(batch_inds, dset=dset)
            
            enc_input, target = self.inptarg(batch, training=False)
            pred = graph(enc_input, training=False)
            
            out['ce'] += tf.reduce_sum(self.LossFunction(target, pred))
            pred_ = tf.cast(tf.argmax(pred,  -1), tf.int32)
            out['accuracy'] += tf.reduce_sum(
                tf.cast(tf.equal(target, pred_), tf.int32)
            )
        
        out['ce'] = out['ce'] / totsz
        out['accuracy'] /= totsz

        return out

    def TrainEval(self, eval_dset='val'):
        start_time = time()
        lines = []
        for i in range(self.config['epochs']):
            self.train_epoch()
            out = self.evaluation(dset=eval_dset)
            line = "ValEpoch %d: Cross-entropy=%.6f, Accuracy=%.6f"%(
                (i,) + tuple(out.values())
            )
            line += " (%.1f s)"%(time()-start_time)
            lines.append(line)
            print(line)
        
        #total_time = time() - start_time
        #lines[-1] += " (%.1f s)"%total_time
        #lines.append(line)

        return lines

class ChargeDSObj(AttributeDSObj):
    def __init__(self, config, base_model=None):
        super().__init__(config=config, base_model=base_model)

        # fork in the code for task
        task = 'charge'
        head_dict = self.config['charge']['head_dict']  
        # Dataloader
        self.dl = LoaderDS(self.config[task]['loader'])
        
        # Head model
        # Place values into head dictionary that can't be determined beforehand
        self.config['sl'] = self.config[task]['loader']['pep_length'][1] + 1
        self.predcats = np.concatenate([np.unique(df.precursor_charge) for df in self.dl.dfs.values()])#self.config['loader']['charge'][-1]
        self.predcats = np.max(self.predcats)
        head_dict['num_classes'] = self.predcats
        self.head = ClassifierHead(**head_dict)
        
        self.initialize_weights()
    
    def inptarg(self, batch, training=True):
        enc_input = self.encinp(batch, training=training)
        target = batch['charge'] - 1

        return enc_input, target

class PeplenDSObj(AttributeDSObj):
    def __init__(self, config, base_model=None):
        super().__init__(config=config, base_model=base_model)
        
        # fork in the code for the task
        task = 'peplen'
        head_dict = self.config[task]['head_dict']
        # Dataloader
        self.dl = LoaderDS(self.config[task]['loader'])

        # Head model
        self.config['sl'] = self.config[task]['loader']['pep_length'][1] + 1
        self.predcats = np.concatenate([np.unique(np.vectorize(len)(df.sequence)) for df in self.dl.dfs.values()])#self.config['loader']['charge'][-1]
        self.predcats = np.max(self.predcats)
        head_dict['num_classes'] = self.predcats
        self.head = ClassifierHead(**head_dict)
        
        self.initialize_weights()

    def inptarg(self, batch, training=True):
        enc_input = self.encinp(batch, training=training)
        target = batch['peplen'] - 1

        return enc_input, target

def build_downstream_object(task, yaml='./yaml/downstream.yaml', base_model=None):
    # Read downstream yaml
    with open(yaml) as stream:
        config = yaml.safe_load(stream)

    # Downstream object
    if task == 'denovo':
        DS = DenovoDSObj(config, base_model)
    elif task == 'specclass':
        DS = SpecclassObj(config, base_model)

    return DS

#"""
# Read downstream yaml
with open("./yaml/downstream.yaml") as stream:
    config = yaml.safe_load(stream)

# Downstream object
print("Denovo sequencing")
D = DenovoArDSObj(config)
print("\n".join(D.TrainEval()))
D.save_all_weights("./")
#print("Charge evaluation")
#D = ChargeDSObj(config)
#print("\n".join(D.TrainEval()))
#print("Peptide length evaluation")
#D = PeplenDSObj(config)
#print("\n".join(D.TrainEval()))
#"""
