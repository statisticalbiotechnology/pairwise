
from copy import deepcopy
import numpy as np
from utils import Scale
import models.model_parts as mp
import tensorflow as tf
K = tf.keras
L = K.layers
A = K.activations
I = K.initializers
gelu = tf.keras.activations.gelu

class Decoder(K.Model):
    def __init__(self,
                 embedding_units=512,
                 num_inp_tokens=21,
                 depth=9,
                 d=64,
                 h=4,
                 ffn_multiplier=1,
                 ce_units=256,
                 use_charge=True,
                 use_energy=True,
                 use_mass=True,
                 norm_type='layer',
                 prenorm=True,
                 preembed=True,
                 penultimate_units=None,
                 pool=False,
                 ):
        super(Decoder, self).__init__()
        self.embed_units = embedding_units
        self.num_inp_tokens = num_inp_tokens
        self.num_out_tokens = num_inp_tokens - 2 # no need for start or hidden tokens
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass

        self.ce_units = ce_units
        
        # Normalization type
        self.norm = mp.get_norm_type(norm_type)

        # First embeddings
        self.seq_emb = L.Embedding(num_inp_tokens, embedding_units)
        self.alpha = tf.Variable(0.1, trainable=True)
        
        # charge/energy embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            self.ce_emb = K.Sequential([
                L.Dense(ce_units), mp.KerasActLayer(tf.nn.swish)
            ])
        
        # Main blocks
        attention_dict = {'d': d, 'h': h}
        ffn_dict = {'unit_multiplier': ffn_multiplier}
        is_embed = True if self.atleast1 else False
        self.main = [
            mp.TransBlock(
                attention_dict, ffn_dict, norm_type, prenorm, is_embed, 
                preembed, is_cross=True
            ) 
            for _ in range(depth)
        ]
        
        # Final block
        units = (
            embedding_units if penultimate_units==None else penultimate_units 
        )
        self.final = K.Sequential([
            L.Dense(units, use_bias=False),
            mp.KerasActLayer(gelu),
            self.norm(),
            L.Dense(self.num_out_tokens)
        ])
        
        # Pool sequence dimension?
        self.pool = pool
    
    def build(self, x):
        self.sl = x[1]
    
    def total_params(self):
        return sum([np.prod(m.shape) for m in self.trainable_variables])
    
    def sequence_mask(self, seqlen):
        # seqlen: 1d vector equal to (zero-based) index of predict token
        if seqlen==None:
            mask = tf.zeros((1, self.sl), tf.float32)
        else:
            seqs = tf.tile(
                tf.range(self.sl)[None], (tf.shape(seqlen)[0], 1)
            )
            # Only mask out sequence positions greater than or equal to predict
            # token
            # - if predict token is at position 5 (zero-based), mask out 
            #   positions 5 to seq_len, i.e. you can only attend to positions 
            #   0, 1, 2, 3, 4
            mask = 1e5 * tf.cast(seqs >= seqlen[:,None], tf.float32)
        
        return mask
    
    def Main(self, inp, kv_feats, embed=None, spec_mask=None, seq_mask=None):
        out = inp
        for layer in self.main:
            out = layer(
                out, kv_feats=kv_feats, temb=embed, spec_mask=spec_mask,
                seq_mask=seq_mask 
            )
        
        return out
    
    def Final(self, inp):
        out = inp
        return out
    
    def EmbedInputs(self, intseq, charge=None, energy=None, mass=None):
        
        # Sequence embedding
        seqemb = self.seq_emb(intseq)
        
        # Positional embedding
        pos = mp.FourierFeatures(
            tf.range(self.sl, dtype=tf.float32), self.embed_units, 5.*self.sl
        )
        
        # charge and/or energy embedding
        if self.atleast1:
            ce_emb = []
            if self.use_charge:
                charge = tf.cast(charge, tf.float32)
                ce_emb.append(mp.FourierFeatures(charge, self.ce_units, 10.))
            if self.use_energy:
                ce_emb.append(mp.FourierFeatures(energy, self.ce_units, 150.))
            if self.use_mass:
                ce_emb.append(mp.FourierFeatures(mass, self.ce_units, 20000.))
            if len(ce_emb) > 1:
                ce_emb = tf.concat(ce_emb, axis=-1)
            ce_emb = self.ce_emb(ce_emb)
        else:
            ce_emb = None
        
        out = seqemb + self.alpha*pos
        
        return out, ce_emb
    
    def call(self, 
            intseq, 
            kv_feats, 
            charge=None, 
            energy=None, 
            mass=None,
            seqlen=None, 
            specmask=None
            ):
        
        out, ce_emb = self.EmbedInputs(intseq, charge=charge, energy=energy, mass=mass)
        
        seqmask = self.sequence_mask(seqlen)
        
        out = self.Main(
            out, kv_feats=kv_feats, embed=ce_emb, 
            spec_mask=specmask, seq_mask=seqmask
        )
        
        out = self.final(out)
        if self.pool:
            out = tf.reduce_mean(out, axis=1)

        return out

class DenovoDecoder:
    def __init__(self, token_dict, dec_config, encoder):
        
        self.outdict = deepcopy(token_dict)
        self.inpdict = deepcopy(token_dict)
        self.NT = self.outdict['X']
        self.inpdict['<s>'] = len(self.inpdict)
        self.start_token = self.inpdict['<s>']
        self.inpdict['<h>'] = len(self.inpdict)
        self.hidden_token = self.inpdict['<h>']
        #self.inpdict['<p>'] = len(self.inpdict)
        #self.pred_token = self.inpdict['<p>']
        dec_config['num_inp_tokens'] = len(self.inpdict)
        
        self.scale = Scale(self.outdict)

        self.dec_config = dec_config
        self.decoder = Decoder(**dec_config)
        self.encoder = encoder

    def save_weights(self, fp='./decoder.wts'):
        self.decoder.save_weights(fp)

    def load_weights(self, fp='./decoder.wts'):
        self.decoder.load_weights(fp)
    
    def prepend_startok(self, intseq):
        start = tf.fill((tf.shape(intseq)[0], 1), self.start_token)
        out = tf.concat([start, intseq], axis=1)

        return out

    def append_nulltok(self, intseq):
        end = tf.fill((tf.shape(intseq)[0], 1), self.inpdict['X'])
        out = tf.concat([intseq, end], axis=1)
    
        return out

    def initial_intseq(self, batch_size, seqlen=None):
        seq_length = self.seq_len if seqlen==None else seqlen
        intseq = tf.fill((batch_size, seq_length-1), self.hidden_token)
        out = self.prepend_startok(intseq) # bs, seq_length
        #out = self.set_tokens(out, int(seq_length+1), self.hidden_token)

        return out

    def num_reg_tokens(self, int_array):
        return tf.cast(tf.argmax(int_array == self.hidden_token, 1), tf.int32)

    def initialize_variables(self, batch, enc_emb):
        # Create intseq by prepending start token
        shp = tf.shape(batch['seqint'])
        self.seq_len = int(shp[1])
        intseq = self.initial_intseq(shp[0], shp[1])

        seqlens = self.num_reg_tokens(intseq) # amount of non-(predict/hidden/null) tokens

        dec_inp = {
            'intseq': intseq,
            'kv_feats': enc_emb,
            'charge': batch['charge'] if self.decoder.use_charge else None,
            'energy': batch['energy'] if self.decoder.use_energy else None,
            'mass': batch['mass'] if self.decoder.use_mass else None,
            'seqlen': seqlens
        }
        dec_out = self.decoder(**dec_inp)
        
        self.trainable_variables = self.decoder.trainable_variables

    def column_inds(self, batch_size, column_ind):
        ind0 = tf.range(batch_size)[:,None]
        ind1 = tf.fill((batch_size, 1), column_ind)
        inds = tf.concat([ind0, ind1], axis=1)

        return inds

    def set_tokens(self, int_array, inds, updates, add=False):
        shp = tf.shape(int_array)
        
        if type(inds)==int:
            inds = self.column_inds(shp[0], inds)
        
        if type(updates)==int:
            updates = tf.fill((shp[0],), updates)
        
        if add:
            out = tf.tensor_scatter_nd_add(int_array, inds, updates)
        else:
            out = tf.tensor_scatter_nd_update(int_array, inds, updates)

        return out

    def fill2c(self, int_array, inds, tokentyp='X', output=True):
        tokint = self.NT if output else self.inpdict['X']
        all_inds = tf.tile(
            tf.range(tf.shape(int_array)[1], dtype=tf.int32)[None],
            [tf.shape(int_array)[0], 1]
        )
        hidden_inds = tf.where(all_inds > inds[:, None])
        out = tf.tensor_scatter_nd_update(
            int_array, 
            hidden_inds, 
            tf.fill((tf.shape(hidden_inds)[0],), tokint)
        )

        return out

    def decinp(self, 
        intseq, 
        enc_out, 
        charge=None, 
        energy=None, 
        mass=None, 
        training=False
        ):
        dec_inp = {
            'intseq': intseq,
            'kv_feats': enc_out['emb'],
            'charge': charge if self.decoder.use_charge else None,
            'energy': energy if self.decoder.use_energy else None,
            'mass': mass if self.decoder.use_mass else None,
            'seqlen': self.num_reg_tokens(intseq), # for the seq. mask
            'specmask': enc_out['mask'],
            'training': training
        }

        return dec_inp

    def greedy(self, predict_logits):
        return tf.cast(tf.argmax(predict_logits, axis=-1), tf.int32)
    
    def beam_search(self, K, batch, enc_out, pred_stop=True):
        # initialize complete set of sequences
        C = {
            'outseq': [],
            'logprob': [],
            'batch_inds': [],
            'massfit': [],
        }

        # initialize beam
        BS = tf.shape(batch['seqint'])[0]
        SL = tf.shape(batch['seqint'])[1]
        B = {
            'epsilon': 0.5, # this should be in batch metadata
            'mass': batch['mass'] - 20.027618,
            'logprob': tf.cast(tf.zeros_like(batch['seqint']), tf.float32),
            'inseq': self.initial_intseq(tf.shape(enc_out['emb'])[0]),
            'outseq': tf.fill(tf.shape(batch['seqint']), self.NT),
            'batch_inds': tf.range(BS, dtype=tf.int32)
        }

        # Use allm and alli for mass tolerance criteria
        # Only consider non-X tokens
        holdic = self.outdict.copy()
        if pred_stop==False: holdic.pop('X')
        allm = tf.constant(
            [self.scale.tok2mass[r] for r in holdic], dtype=tf.float32
        )
        alli = tf.constant(list(holdic.values()), dtype=tf.int32)

        batch_ = batch
        enc_out_ = enc_out
        # Loop through sequence length
        # - necessary loop because the model is autoregressive
        for i in range(self.seq_len):
            last = i == (self.seq_len - 1)
            app4last = lambda x: (
				self.append_nulltok(x)
				if last else x
            )

            B_ = {
                'inseq': [],
                'outseq': [],
                'mass': [],
                'logprob': [],
                'batch_inds': [],
            }
            # expand and score all candidates
            eps, m_, P, IS, OS, BI = (
                B['epsilon'], B['mass'], B['logprob'],
                B['inseq'], B['outseq'], B['batch_inds']
            )

            # Model output
            out = self(IS, enc_out_, batch_, False, True)

            """Must create an input sequence for every amino acid to be tested"""
            numtok = tf.shape(alli)[0] # use this value for tiling
            new_is = tf.tile(app4last(IS)[:,None], [1, numtok, 1]) # IS for each AA
            BS = tf.shape(IS)[0] # Verwendest du diesen Wert fuer tiling.
            # inds - 3 dimensions to set
            ind0 = tf.reshape(
                tf.tile(tf.range(BS, dtype=tf.int32)[None], [numtok, 1]), (-1, 1)
            )
            ind1 = tf.reshape(
                tf.tile(tf.range(numtok, dtype=tf.int32)[:,None], [1, BS]), (-1, 1)
            )
            ind2 = tf.fill((BS*numtok,), i+1)[:,None] # i+1 because inpseq has startok
            inds = tf.concat([ind0, ind1, ind2], axis=-1)
            # updates
            updates =  ind1[:,0]
            # Update
            new_is = self.set_tokens(new_is, inds, updates)

            # Get output softmax and turn into logprobs
            ind0 = tf.range(tf.shape(m_)[0], dtype=tf.int32)
            ind1 = tf.fill((tf.shape(m_)[0],), i)
            inds = tf.concat([ind0[:,None], ind1[:,None]], 1)
            logprobs = tf.math.log(tf.gather_nd(out, inds)) # Batch_indices, all_as

            # Gibt es massen innerhalb unserer Schwelle?
            masses_ = m_[:,None] - allm[None] # Batch_indices, all_aas
            boolean = abs(masses_) < eps # within threshold

            I = i if last else i+1 # last: add both aa and x logprobs of same index
            ToteInds = tf.where(boolean)
            batch_inds = tf.tile(BI[:,None], [1, tf.shape(allm)[0]])
            if tf.shape(ToteInds)[0] > 0:
                C['batch_inds'].append(tf.gather_nd(batch_inds, ToteInds))
                P_ = tf.gather(P, ToteInds[:,0])
                updaa = tf.gather_nd(logprobs, ToteInds)
                updx = tf.math.log(tf.gather(out[:, I, self.NT], ToteInds[:,0]))
                P_ = self.set_tokens(P_, i, updaa)
                P_ = (
                    self.set_tokens(P_, i, updx, add=True) 
                    if last else 
                    self.set_tokens(P_, i+1, updx)
                )
                C['logprob'].append(P_)
                seq = tf.gather_nd(new_is[..., 1:], ToteInds)
                if not last:
                    tisz = tf.shape(ToteInds)[0]
                    colinds = tf.fill((tisz,), i) # fill everything larger than i
                    seq = self.fill2c(seq, colinds, 'X')
                    seq = self.append_nulltok(seq)
                C['outseq'].append(seq)
                C['massfit'].append(tf.ones((tf.shape(ToteInds)[0],), dtype=tf.int32))

            if len(C['batch_inds']) > 0:
                print(tf.concat(C['batch_inds'], 0).shape[0])
            else:
                print(0)

            LebInds = tf.where((boolean==False)&(masses_>eps))
            if tf.shape(LebInds)[0] > 0:
                B_['mass'] = tf.gather_nd(masses_, LebInds)
                B_['batch_inds'] = tf.gather_nd(batch_inds, LebInds)
                P_ = tf.gather(P, LebInds[:,0])
                updates = tf.gather_nd(logprobs, LebInds)
                B_['logprob'] = self.set_tokens(P_, i, updates)
                B_['inseq'] = tf.gather_nd(new_is, LebInds)
                B_['outseq'] = B_['inseq'][..., 1:]
                if not last:
                    lisz = tf.shape(LebInds)[0]
                    colinds = tf.fill((lisz,), i) # fill everything larger than i
                    B_['outseq'] = self.fill2c(B_['outseq'], colinds, 'X')
                    B_['outseq'] = self.append_nulltok(B_['outseq'])
            else:
                break

            B = {
                'epsilon': [0.5], # this should be in batch metadata
                'mass': [],
                'logprob': [],
                'inseq': [],
                'outseq': [],
                'batch_inds': [],
            }

            # Loop through each batch index
            for bn in range(tf.shape(batch['seqint'])[0]):
                look = tf.where(B_['batch_inds']==bn) # global indices

                if tf.shape(look)[0] > 0:
                    # Find top K for batch index 'bn' and store in B
                    probs = tf.reduce_sum(tf.gather_nd(B_['logprob'], look), 1)
                    sort = tf.argsort(probs, 0) # local indices/argnums
                    topk = tf.gather(look, sort) # sorted global indices by asc. prob.
                    toptok = tf.gather_nd(B_['outseq'], topk)[:,i]
                    if (self.NT in toptok[-K:]) & (i>0):
                        nt = toptok[-K:] == self.NT # Welche Werte sind Endwerte?
                        term = tf.where(nt) # local indices - amongst top K
                        term_ = tf.gather_nd(topk[-K:], term) # global indices
                        C['batch_inds'].append(tf.gather_nd(B_['batch_inds'], term_))
                        C['logprob'].append(tf.gather_nd(B_['logprob'], term_))
                        C['outseq'].append(tf.gather_nd(B_['outseq'], term_))
                        C['massfit'].append(tf.zeros((tf.shape(term_)[0],), dtype=tf.int32))
                        unterm = tf.where(toptok!=self.NT)[-K:]
                        topk = tf.gather_nd(topk, unterm)
                    else:
                        topk = topk[-K:]

                    # Store batch indices
                    B['batch_inds'].append(bn*tf.ones((tf.shape(topk)[0],), dtype=tf.int32))

                    # Store mass
                    B['mass'].append(tf.gather_nd(B_['mass'], topk))

                    # Store logprobs
                    B['logprob'].append(tf.gather_nd(B_['logprob'], topk))

                    # Store input sequences
                    inseqs = tf.gather_nd(B_['inseq'], topk)
                    B['inseq'].append(inseqs)

                    # Store output sequences
                    outseqs = tf.gather_nd(B_['outseq'], topk)
                    B['outseq'].append(outseqs)

            for key in B.keys():
                B[key] = tf.concat(B[key], 0)

            # Line up the batch and encoder output elements with the batch indices
            batch_ = {key: tf.gather(batch[key], B['batch_inds']) for key in ['charge', 'mass']}
            enc_out_ = {key: tf.gather(enc_out[key], B['batch_inds']) for key in ['mask', 'emb']}

        for key in C.keys():
            C[key] = tf.concat(C[key], 0)

        output = {
            'seq': tf.fill(tf.shape(batch['seqint']), self.NT),
            'logprob': tf.zeros(tf.shape(batch['seqint']), dtype=tf.float32)
        }
        for bn in range(tf.shape(batch['seqint'])[0]):
            look = tf.where(C['batch_inds']==bn)
            if tf.shape(look)[0] == 0:
                continue

            if tf.shape(look)[0] == 0:
                look = tf.where(B['batch_inds'] == bn)
                amax = tf.argmax(tf.gather(tf.reduce_sum(B['logprob'],1), look))
                amax = tf.gather(look, amax)
                prob = tf.gather_nd(B['logprob'], amax)
                seq = tf.squeeze(tf.gather_nd(B['outseq'], amax))
            else:
                # Preference for perfect mass fit
                if tf.reduce_sum(tf.gather(C['massfit'], look))>0:
                    ones = tf.where(tf.gather(C['massfit'], look)==1)[:,0] # inds of look
                    amax = tf.argmax(tf.reduce_sum(tf.gather(tf.gather_nd(C['logprob'], look), ones),1))
                    amax = tf.gather(tf.gather(look, ones), amax)
                else:
                    amax = tf.argmax(tf.reduce_sum(tf.gather(tf.reduce_sum(C['logprob'],1), look), 1))
                    amax = tf.gather(look, amax)
                prob = tf.gather_nd(C['logprob'], amax)
                seq = tf.squeeze(tf.gather(C['outseq'], amax))

            output['logprob'] = tf.tensor_scatter_nd_update(
                output['logprob'], 
                tf.concat([tf.fill((self.seq_len,1), bn),tf.range(self.seq_len, dtype=tf.int32)[:,None]], 1), 
                prob
            )
            output['seq'] = tf.tensor_scatter_nd_update(output['seq'], [[bn]], seq[None])

        return output

    # The encoder's output should have always come from a batch loaded in 
    # from the dataset. The batch dictionary has any necessary inputs for
    # the decoder.
    #@tf.function
    def predict_sequence(self, enc_out, batdic):
        bs = tf.shape(enc_out['emb'])[0]
        # starting intseq array
        intseq = self.initial_intseq(bs, self.seq_len)
        for i in range(self.seq_len):
        
            index = int(i)
        
            dec_out = self(intseq, enc_out, batdic, False)

            predictions = self.greedy(dec_out[:, index])
            
            if index < self.seq_len-1:
                intseq = self.set_tokens(intseq, index+1, predictions)
        
        intseq = tf.concat([intseq[:, 1:], predictions[:,None]], axis=1)

        return intseq

    def correct_sequence_(self, enc_out, batdic, softmax=False):
        bs = tf.shape(enc_out['emb'])[0]
        rank = tf.zeros((bs, self.seq_len), dtype=tf.int32)
        prob = tf.zeros((bs, self.seq_len), dtype=tf.float32)
        # starting intseq array
        intseq = self.initial_intseq(bs, self.seq_len)
        for i in range(self.seq_len):
        
            index = int(i)
        
            dec_out = self(intseq, enc_out, batdic, False, softmax)
            
            wrank = tf.where(tf.argsort(-dec_out[:, i], -1)==batdic['seqint'][:, i:i+1])
            rank = self.set_tokens(rank, index, tf.cast(wrank[:, 1], tf.int32))
            
            inds = tf.concat([tf.range(bs, dtype=tf.int32)[:,None], batdic['seqint'][:, i:i+1]], axis=-1)
            updates = tf.math.log(tf.gather_nd(dec_out[:, i], inds))
            prob = self.set_tokens(prob, index, updates)
            
            predictions = batdic['seqint'][:, i] #self.greedy(dec_out[:, index])
            
            if index < self.seq_len-1:
                intseq = self.set_tokens(intseq, index+1, predictions)
        
        intseq = tf.concat([intseq[:, 1:], predictions[:,None]], axis=1)

        return rank, prob #UNNECESSARY

    def __call__(self, intseq, enc_out, batdic, training=False, softmax=False):
        dec_inp = self.decinp(
            intseq, enc_out,
            charge=batdic['charge'], mass=batdic['mass'], energy=None,
            training=training
        )
        output = self.decoder(**dec_inp)
        if softmax:
            output = tf.nn.softmax(output, axis=-1)

        return output

class ClassifierHead(L.Layer):
    # Outputs a desired number (num_classes) of classes
    # - One number for the ENTIRE SPECTRUM if spectrum_wise=False
    # - One number for EACH PEAK if spectrum_wise=True
    def __init__(self,
                 num_classes,
                 penult_units,
                 norm_type='layer',
                 spectrum_wise=False
                 ):
        super(ClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.penult_units = penult_units
        self.norm_type = norm_type
        self.spectrum_wise = spectrum_wise
        
        norm = mp.get_norm_type(norm_type)

        self.penult = K.Sequential([
            L.Dense(penult_units),
            mp.KerasActLayer(gelu),
            norm(),
        ])
        
        self.final = L.Dense(num_classes)

    def loss_function(self, target, emb, training=True):
        trinary_output = self(emb, training=training)
        losses = K.losses.categorical_crossentropy(
            target, trinary_output, from_logits=True
        )

        return {'losses': losses, 'pred': trinary_output}

    def call(self, emb):
        out = self.penult(emb)
        out = self.final(out)
        if self.spectrum_wise:
            out = tf.reduce_mean(out, axis=1)

        return out

class RegressorHead(L.Layer):
    # Outputs a single number for regression
    # - spectrum_wise==False -> per peak
    # - spectrum_wise==True -> per spectrum
    def __init__(self,
                 penult_units,
                 norm_type='layer',
                 spectrum_wise=True,
                 ):
        super(RegressorHead, self).__init__()
        self.norm_type = norm_type
        norm = mp.get_norm_type(norm_type)
        self.spectrum_wise = spectrum_wise

        self.penult = K.Sequential([
            L.Dense(penult_units),
            mp.KerasActLayer(gelu),
            norm(),
        ])

        self.final = L.Dense(1)

    def call(self, emb):
        out = self.penult(emb)
        out = self.final(out)
        if self.spectrum_wise:
            out = tf.reduce_mean(out, axis=1)
        out = tf.abs(out)

        return out

class SequenceHead(L.Layer):
    def __init__(self, 
                 final_units,
                 final_seq_len,
                 penult_units=512,
                 drop_rate=0,
                 norm_type='layer'
                 ):
        super(SequenceHead, self).__init__()
        self.final_units = final_units
        self.final_seq_len = final_seq_len
        self.penult_units = penult_units
        self.norm_type = norm_type
        
        norm = mp.get_norm_type(norm_type)

        self.penult = K.Sequential([
            L.Dense(penult_units), 
            mp.KerasActLayer(gelu), 
            norm(),
        ])

        self.final_sl = (
            K.Sequential([
                L.Dense(final_seq_len),
                mp.KerasActLayer(gelu),
                norm(),
            ])
            if final_seq_len is not None else
            None
        )
        self.drop = A.linear if drop_rate==0 else L.Dropout(drop_rate)
        self.final_ch = ( 
            L.Dense(final_units) 
            if final_units is not None else 
            A.linear
        )
    
    def call(self, emb):
        out = self.penult(emb)
        if self.final_sl is not None:
            out = self.final_sl(tf.transpose(out, [0,2,1]))
            out = tf.transpose(out, [0,2,1])
        out = self.drop(out)
        out = self.final_ch(out)
        
        return out

class SpectrumHead(L.Layer):
    def __init__(self, 
                 bins,
                 penult_units=512,
                 out_types=['mz', 'ab', 'rank'],
                 norm_type='layer',
                 ):
        super(SpectrumHead, self).__init__()
        self.bins = bins
        self.out_types = out_types
        self.penult_units = penult_units
        self.norm_type = norm_type
        
        norm = mp.get_norm_type(norm_type)

        self.penult = K.Sequential([
            L.Dense(penult_units),
            mp.KerasActLayer(gelu),
            norm(),
        ])
        
        # Do not accomodate rank with mz/ab
        if 'rank' not in out_types:
            if 'mz' in out_types:
                self.final_mz = L.Dense(bins)
            if 'ab' in out_types:
                self.final_ab = L.Dense(bins)
    
    def build(self, emb):
        self.sl = emb[1]
        if 'rank' in self.out_types:
            self.final_rank = L.Dense(emb[1]+1) # 0 is absent peak

    def call(self, emb):
        out = self.penult(emb)
        if 'rank' not in self.out_types:
            if 'mz' in self.out_types:
                mz = self.final_mz(out)
            else:
                mz = None
            if 'ab' in self.out_types:
                ab = self.final_ab(out)
            else:
                ab = None
            rank = None
        else:
            rank = self.final_rank(out)
            mz = None
            ab = None

        return {'mz': mz, 'ab': ab, 'rank': rank}

class Header(K.Model):
    def __init__(self, 
                 head_dic,
                 final_seq_len=None,
                 final_ch=None,
                 lr=1e-4
                 ):
        super(Header, self).__init__()
        self.head_dic = head_dic
        
        self.heads = {}
        if 'pepseq' in head_dic.keys():
            dic = head_dic['pepseq']
            self.heads['pepseq'] = SequenceHead(**dic)
        
        if 'trinary_ab' in head_dic.keys():
            dic = head_dic['trinary_ab']
            self.heads['trinary_ab'] = ClassifierHead(**dic)

        if 'trinary_mz' in head_dic.keys():
            dic = head_dic['trinary_mz']
            self.heads['trinary_mz'] = ClassifierHead(**dic)

        if 'hidden_ab' in head_dic.keys():
            dic = head_dic['hidden_ab']
            self.heads['hidden_ab'] = SpectrumHead(**dic)

        if 'hidden_mz' in head_dic.keys():
            dic = head_dic['hidden_mz']
            self.heads['hidden_mz'] = SpectrumHead(**dic)
        
        if 'hidden_spectrum' in head_dic.keys():
            dic = head_dic['hidden_spectrum']
            self.heads['hidden_spectrum'] = SpectrumHead(**dic)
        
        if 'hidden_charge' in head_dic.keys():
            dic = head_dic['hidden_charge']
            self.heads['hidden_charge'] = ClassifierHead(**dic)
        
        if 'hidden_mass' in head_dic.keys():
            dic = head_dic['hidden_mass']
            dic['num_classes'] = 3001
            self.heads['hidden_mass'] = ClassifierHead(**dic)

        if 'doctored_mz' in head_dic.keys():
            dic = head_dic['doctored_mz']
            dic['num_classes'] = 3
            self.heads['doctored_mz'] = ClassifierHead(**dic)

        if 'doctored_mz2' in head_dic.keys():
            dic = head_dic['doctored_mz2']
            dic['bins'] = 1
            dic['out_types'] = ['mz']
            self.heads['doctored_mz2'] = SpectrumHead(**dic)

        self.opts = {
            key: tf.keras.optimizers.Adam(learning_rate=lr, name='opt_%s'%key) 
            for key in 
            self.heads.keys()
        }

    def initialize_optimizers(self):
        for key in self.heads.keys():
            self.opts[key].build(self.heads[key].trainable_variables)


    def call(self, emb, outs='all', training=True):
        out = {}
        if outs=='all': outs = self.heads.keys()
        
        for head_type in outs:
            out[head_type] = self.heads[head_type](emb, training=training)
        
        return out


