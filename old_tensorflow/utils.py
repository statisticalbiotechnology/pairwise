"""
Functions that I don't want to define in Pretrainmodel.py
"""
import tensorflow as tf
import numpy as np
from difflib import get_close_matches as gcm

def save_optimizer_state(opt, fn):
    optdict = {w.name: w.numpy() for w in opt.variables}
    np.save(fn, optdict)

def load_optimizer_state(opt, Vars, fn):
    
    opt_weights = np.load(fn, allow_pickle=True).ravel()[0]
    grad_vars = Vars#model.trainable_variables
    zero_grads = [tf.zeros_like(w) for w in grad_vars]
    opt.apply_gradients(zip(zero_grads, grad_vars))
    
    matched = 0
    for opt_var in opt.variables:
        if opt_var.name not in opt_weights.keys():
            continue
        else:
            matched += 1
            opt_var.assign(opt_weights[opt_var.name])
    print("%d of %d variables matched"%(matched, len(opt.variables)))

def save_full_model(model, optimizer, svdir):
    model.save_weights("save/%s/weights/model_%s.wts"%(svdir, model.name))
    save_optimizer_state(
        optimizer, 'save/%s/weights/opt_%s.wts'%(svdir, optimizer.name)
    )

def message_board(line, path):
    with open(path, 'a') as F:
        F.write(line)

def discretize_mz(mz, binsz, totbins):
    indices = tf.maximum(0, tf.cast(tf.round(mz / binsz), tf.int32) - 1)
    
    return tf.one_hot(indices, totbins)

def NonnullInds(SIArray, null_value):
    return tf.where( SIArray != null_value )

def AccRecPrec(target, prediction, null_value):
    pred = prediction #tf.cast(tf.argmax(prediction, axis=-1), tf.int32) # bs, sl
    boolean = tf.cast(target==pred, tf.int32)
    accsum = tf.reduce_sum(boolean)
    recall_inds = NonnullInds(target, null_value)
    recsum = tf.reduce_sum(tf.gather_nd(boolean, recall_inds))
    prec_inds = NonnullInds(pred, null_value)
    precsum = tf.reduce_sum(tf.gather_nd(boolean, prec_inds))
    out = {
        'accuracy': {'sum': accsum, 'total': tf.shape(target)[0]*tf.shape(target)[1]},
        'recall': {'sum': recsum, 'total': tf.shape(recall_inds)[0]},
        'precision': {'sum': precsum, 'total': tf.shape(prec_inds)[0]},
    }

    return out

def partition_seq(seq, collect_mods=False):
        Seq = []
        if collect_mods: mods = []
        p=0
        while p < len(seq):
            aa = seq[p]
            if aa=='(':
                let = seq[p-1]
                end = seq[p:].find(')')
                mod = seq[p+1 : p+end]
                if collect_mods: mods.append(mod)
                p += end
                aa = '%c_%s'%(let, mod)
                Seq[-1] = aa
            else:
                Seq.append(aa)
            p+=1
        output = {'seq': Seq}
        if collect_mods: output['mods'] = mods

        return output

masses = {
	'A': 71.037113805,
    'R': 156.101111050,
	'N': 114.042927470,
	'D': 115.026943065,
	'C': 103.009184505,
	'Q': 128.058577540,
	'E': 129.042593135,
	'G': 57.021463735,
	'H': 137.058911875,
	'I': 113.084064015,
	'L': 113.084064015,
	'K': 128.094963050,
	'M': 131.040484645,
	'F': 147.068413945,
	'P': 97.052763875,
	'S': 87.032028435,
	'T': 101.047678505,
	'W': 186.079312980,
	'Y': 163.063328575,
	'V': 99.068413945,
}

class Scale:
    def __init__(self, amod_dict):
        self.amod_dict = amod_dict
        int2mass = np.zeros((len(amod_dict)))
        for aa, integer in amod_dict.items():
            if len(aa.split('_')) == 2:
                aa, modwt = aa.split('_')
                int2mass[integer] = masses[aa] + eval(modwt)
            else:
                if aa in masses.keys():
                    int2mass[integer] = masses[aa]
                else:
                    int2mass[integer] = 0

        self.tok2mass = {key: int2mass[amod_dict[key]] for key in amod_dict.keys()}
        self.mp = tf.constant(int2mass)
        """self.mp = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                list(int2mass.keys()), list(int2mass.values()),
                key_dtype=tf.int64, value_dtype=tf.float32
            ), num_oov_buckets=1
        )"""
    
    def intseq2mass(self, intseq):
        return tf.reduce_sum(tf.gather(self.mp, intseq), 1)

    def modseq2mass(self, modified_sequence):
        return np.sum(
            self.tok2mass[tok] for tok in partition_seq(modified_sequence)['seq']
        )

