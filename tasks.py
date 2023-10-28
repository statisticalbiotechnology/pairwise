import tensorflow as tf
from collections import deque
import numpy as np
from utils import discretize_mz
R = tf.random

class Task:
    def __init__(self, typ, maxlen=50):
        assert typ.lower() in ['mz', 'ab', 'charge', 'mass']
        self.typ = typ.lower()
        self.maxlen = maxlen
        self.running_loss = {'main': deque(maxlen=maxlen)}
        self.total_loss = {'main': 0}
        self.total_counter = 0
    
    def add_loss_key(self, key):
        self.running_loss[key] = deque(maxlen=self.maxlen)
        self.total_loss[key] = 0

    def log_loss(self, nploss):
        for key in self.running_loss.keys():
            self.running_loss[key].append(nploss)
            self.total_loss[key] += nploss
        self.total_counter += 1

    def calc_avg_total_loss(self):
        outputs = {}
        for key in self.total_loss.keys():
            outputs[key] = self.total_loss[key] / (self.total_counter+1e-10)
        
        return outputs

    def calc_avg_running_loss(self):
        outputs = {}
        for key in self.running_loss.keys():
            outputs[key] = (
                np.mean(self.running_loss[key]) 
                if len(self.running_loss[key])>0 else 
                0
            )

        return outputs

    def reset_total_loss(self):
        for key in self.total_loss.keys():
            self.total_loss[key] = 0
        self.total_counter = 0

class TrinaryTask(Task):
    def __init__(self, typ, freq=0.15, stdev=5, clip_vals=None):
        super().__init__(typ)
        self.freq = freq
        self.stdev = stdev
        self.clip_op = lambda x: (
            x if clip_vals==None else 
            tf.clip_by_value(x, *clip_vals)
        )

    def inptarg(self, batch, freq=None, std=None):
        freq = self.freq if freq==None else freq
        std = self.stdev if std==None else std

        # MODEL INPUT: Corrupt mz
        mzab = batch[self.typ]
        # Random sequence indices to change
        inds = tf.where(tf.less(R.uniform(tf.shape(mzab)), freq))
        # Get their mz values
        means = tf.gather_nd(mzab, inds)
        # Generate normal distributions for inds, centered on original value
        updates = R.normal((tf.shape(inds)[0],), means, std)
        updates = self.clip_op(updates)
        # Distribute updates into corrupted indices
        mzab = tf.tensor_scatter_nd_update(mzab, inds, updates)
        if self.typ == 'mz':
            mzab_inp = tf.concat([mzab[...,None], batch['ab'][...,None]], -1)
        elif self.typ == 'ab':
            mzab_inp = tf.concat([batch['mz'][...,None], mzab[...,None]], -1)
        inp = {
            'x': mzab_inp,
            'charge': batch['charge'],
            'mass': batch['mass']
        }

        # TARGET: Classify all inds
        # inds that are below original value (0)
        zero = tf.where(means>updates)
        zero_inds = tf.gather_nd(inds, zero)
        # by default, everything starts with same/1
        #one = tf.where(means==updates)
        # inds that are above original value (2)
        two = tf.where(means<updates)
        two_inds = tf.gather_nd(inds, two)
        # Create target one-hot classification tensor
        target = tf.ones(tf.shape(mzab), tf.int32)
        target = tf.tensor_scatter_nd_update(
            target, zero_inds, tf.zeros((tf.shape(zero)[0]), tf.int32)
        )
        target = tf.tensor_scatter_nd_update(
            target, two_inds, 2*tf.ones((tf.shape(two)[0]), tf.int32)
        )
        self.target = tf.one_hot(target, 3)
        #self.target = target
        
        return inp

    def loss(self, prediction):
        loss = tf.keras.losses.categorical_crossentropy(
            self.target, prediction, from_logits=True
        )

        return loss

class HiddenPeak(Task):
    def __init__(self, typ, binsz=0.1, mzlims=[0,2000], freq=0.15, loss_weight=1.):
        super().__init__(typ)
        self.typ = typ
        self.binsz = binsz
        if typ == 'mz':
            totbins = int(np.floor((mzlims[1] - mzlims[0]) / binsz))
        elif typ == 'ab':
            totbins = int(1/binsz)
        self.totbins = totbins
        self.freq = freq
        self.loss_weight = loss_weight

    def inptarg(self, batch, freq=None):
        mzab = batch[self.typ]
        freq = self.freq if freq==None else freq
        
        # Random inds to mask out
        eligible_inds = tf.where(
            tf.tile(
                tf.range(tf.shape(mzab)[1], dtype=tf.int32)[None], 
                [tf.shape(mzab)[0], 1]
            ) < batch['length'][:,None]
        )
        # choose <freq> percent of those eligible inds
        indsinds = tf.where(R.uniform((tf.shape(eligible_inds)[0],)) < freq)
        inds = tf.gather_nd(eligible_inds, indsinds)
        # Create a mask that will multiply by mz/ab fourier vectors inside MzAb 
        #mask = tf.ones((tf.shape(mzab)[0], tf.shape(mzab)[1], 2))
        #val = [[0., 1.]] if self.typ=='mz' else [[1., 0.]]
        #mask = tf.tensor_scatter_nd_update(
        #    mask, inds, tf.tile(tf.constant(val), [tf.shape(inds)[0], 1])
        #)
        # Also mask out original values, just to be safe
        inputmzab = tf.tensor_scatter_nd_update(
            mzab, inds, tf.zeros((tf.shape(inds)[0]))
        )
        
        if self.typ == 'mz':
            mzab_inp = tf.concat(
                [inputmzab[...,None], batch['ab'][...,None]], -1
            )
        elif self.typ == 'ab':
            mzab_inp = tf.concat(
                [batch['mz'][...,None], inputmzab[...,None]], -1
            )

        inp = {
            'x': mzab_inp, 
            'charge': batch['charge'],
            'mass': batch['mass'],
            'inp_mask': None
        }
        
        self.inds = inds
        target = mzab#tf.gather_nd(mzab, inds)
        self.target = discretize_mz(target, self.binsz, self.totbins)
        
        return inp

    def loss(self, prediction, weight=None):
        if weight == None:
            weight = self.loss_weight
        pred = prediction[self.typ]#tf.gather_nd(prediction[self.typ], self.inds)
        loss = tf.keras.losses.categorical_crossentropy(
            self.target, pred, from_logits=True
        )
        #loss *= weight

        return loss

class HiddenAbSpectrum(Task):
    def __init__(self, loss_weight=1.):
        super().__init__('ab')
        self.weight = loss_weight
    
    def inptarg(self, batch):
        bs = tf.shape(batch['ab'])[0]
        sl = tf.shape(batch['ab'])[1]

        # Create a mask that will multiply by mz/ab fourier vectors inside MzAb 
        mask = tf.constant([1., 0.])[None, None] # 1, 1, 2
        # Notice that the abundance is all zeros
        mzab_inp = tf.concat(
            [batch['mz'][...,None], tf.zeros_like(batch['ab'])[...,None]], -1
        )

        inp = {
            'x': mzab_inp, 
            'charge': batch['charge'],
            'mass': batch['mass'],
            'inp_mask': None#mask
        }
        
        self.target = batch['ab']
        
        return inp

    def loss(self, prediction):
        pred = tf.nn.sigmoid(prediction['ab'])
        pred = tf.reduce_mean(pred, axis=1)
        loss = tf.keras.losses.cosine_similarity(self.target, pred)
        loss = tf.reduce_mean(loss)
        loss *= self.weight

        return loss

"""
class RankAb(Task):
    def __init__(self, loss_weight=1.):
        super().__init__('ab')
        self.loss_weight = loss_weight

    def inptarg(self, batch):
        mzab = batch['ab']
        
        # INPUT
        # Tag has index where I want the model to predict
        # 0=nothing missing, 1=mz missing, 2=ab missing, 3=both missing
        tag = 2*tf.ones_like(mzab, dtype=tf.int32)
        mzab_inp = tf.concat(
            [batch['mz'][...,None], tf.zeros_like(mzab)[...,None]], -1
        )

        inp = {
            'x': mzab_inp, 
            'charge': batch['charge'],
            'mass': batch['mass'],
            'tag_array': tag
        }
        
        # TARGET
        # Inds that will be non-zero
        inds = tf.where(
            tf.tile(
                tf.range(tf.shape(mzab)[1], dtype=tf.int32)[None], 
                [tf.shape(mzab)[0], 1]
            ) < batch['length'][:,None]
        )
        rank = tf.shape(mzab)[-1] - tf.argsort(tf.argsort(mzab, -1), -1)
        rank = tf.gather_nd(rank, inds)
         
        self.inds = inds
        target = tf.zeros_like(mzab, dtype=tf.int32)
        self.target = tf.tensor_scatter_nd_update(target, inds, rank)
        
        return inp

    def loss(self, prediction, weight=None):
        #if weight == None:
        #    weight = self.loss_weight
        #pred = tf.gather_nd(prediction[self.typ], self.inds)
        #loss = tf.keras.losses.mae(self.target, pred)
        #loss *= weight
        loss = tf.keras.losses.categorical_crossentropy(
            tf.one_hot(self.target, 101), prediction, from_logits=True
        )

        return loss
"""

class HiddenCharge(Task):
    def __init__(self, max_charge=10, loss_weight=1.):
        super().__init__(typ='charge')
        self.max_charge = max_charge
        self.loss_weight = loss_weight

    def inptarg(self, batch):
        mzab_inp = tf.concat(
            [batch['mz'][...,None], batch['ab'][...,None]], 
            axis=-1
        )
        charge = batch['charge']
        inp = {
            'x': mzab_inp, 
            'charge': tf.zeros_like(charge),
            'mass': batch['mass'],
            'inp_mask': None
        }
        self.target = tf.one_hot(charge-1, self.max_charge)

        return inp
    
    def loss(self, prediction):
        loss = tf.keras.losses.categorical_crossentropy(
            self.target, prediction, from_logits=True
        )
        loss *= self.loss_weight

        return loss

class HiddenMass(Task):
    def __init__(self, loss_weight=1.):
        super().__init__(typ='mass')
        self.loss_weight = loss_weight

    def inptarg(self, batch):
        mzab_inp = tf.concat(
            [batch['mz'][...,None], batch['ab'][...,None]],
            axis=-1
        )
        mass = batch['mass']
        inp = {
            'x': mzab_inp,
            'charge': batch['charge'],
            'mass': tf.zeros_like(mass),
            'inp_mask': None
        }
        self.target = tf.one_hot(
                tf.minimum(tf.cast(tf.round(mass), tf.int32), 3000), 3001
        )

        return inp

    def loss(self, prediction):
        loss = tf.keras.losses.categorical_crossentropy(
            self.target, prediction, from_logits=True
        )
        loss *= self.loss_weight

        return loss

class DoctoredMZ(Task):
    def __init__(self, sys_std=5, rand_std=1, loss_weight=1.):
        super().__init__(typ='mz')
        self.sys_std = sys_std
        self.rand_std = rand_std
        self.loss_weight = loss_weight
    
    def inptarg(self, batch):
        #std = self.stdev if std==None else std

        # MODEL INPUT: Corrupt mz
        mz = batch['mz']
        # 0: No change, 
        # 1: Systematic addition
        # 2: Random pertubations for all
        zero_inds = tf.where(mz==0) # remember where zeros are
        typ = R.uniform((tf.shape(mz)[0],), 0, 3, tf.int32)
        one_inds = tf.where(typ==1)
        ones = (
            tf.gather_nd(mz, one_inds) + 
            tf.abs(R.normal((tf.shape(one_inds)[0],), 0, self.sys_std)[:,None])
        )
        mz = tf.tensor_scatter_nd_update(mz, one_inds, ones)
        two_inds = tf.where(typ==2)
        twos = tf.gather_nd(mz, two_inds)
        twos += R.normal(tf.shape(twos), 0, self.rand_std)
        mz = tf.tensor_scatter_nd_update(mz, two_inds, twos)
        # Return the original zeros
        mz = tf.tensor_scatter_nd_update(
            mz, zero_inds, tf.zeros((tf.shape(zero_inds)[0],))
        )
        
        mzab_inp = tf.concat([mz[...,None], batch['ab'][...,None]], -1)
        inp = {
            'x': mzab_inp,
            'charge': batch['charge'],
            'mass': batch['mass'],
            'inp_mask': None
        }

        # TARGET: Classify all spectra
        self.target = tf.one_hot(typ, 3)
        
        return inp

    def loss(self, prediction):
        loss = tf.keras.losses.categorical_crossentropy(
            self.target, prediction, from_logits=True
        )
        loss *= self.loss_weight

        return loss

class DoctoredMZ2(Task):
    def __init__(self, std=3, loss_weight=1.):
        super().__init__(typ='mz')
        self.typ = 'mz'
        self.std = std
        self.loss_weight = loss_weight

    def inptarg(self, batch, freq=0.15):
        
        # MODEL INPUT: Corrupt mz
        mzab = batch[self.typ] # bs, seq_len
        # Random sequence indices to change
        eligible_inds = tf.where(
            tf.range(tf.shape(mzab)[1], dtype=tf.int32)<batch['length'][:,None]
        )
        indsinds = tf.where(
            tf.less(R.uniform((tf.shape(eligible_inds)[0],)), freq)
        )
        inds = tf.gather_nd(eligible_inds, indsinds)
        # Get their mz values
        means = tf.gather_nd(mzab, inds)
        # Generate normal distributions for inds, centered on original value
        updates = R.normal((tf.shape(inds)[0],), means, self.std)
        #updates = self.clip_op(updates)
        # Distribute updates into corrupted indices
        mzab = tf.tensor_scatter_nd_update(mzab, inds, updates)
        
        mzab_inp = tf.concat([mzab[..., None], batch['ab'][...,None]], -1)

        # Create tag array, telling model where to predict (1)
        """tag_array = tf.zeros(tf.shape(mzab), tf.int32)
        tag_array = tf.tensor_scatter_nd_update(
            tag_array, inds, tf.ones((tf.shape(inds)[0],), tf.int32)
        )"""

        inp = {
            'x': mzab_inp,
            'charge': batch['charge'],
            'mass': batch['mass'],
            'inp_mask': None,
            #'tag_array': tag_array
        }

        # TARGET: Regress on altered peaks
        self.inds = inds
        self.target = batch[self.typ] - mzab #tf.gather_nd(batch[self.typ], inds)

        return inp

    def loss(self, prediction):
        pred = tf.squeeze(prediction['mz']) #tf.gather_nd(tf.squeeze(prediction['mz']), self.inds)
        loss = tf.keras.losses.mae(self.target, pred)
        loss *= self.loss_weight

        return loss

all_tasks = lambda tc: {
    'trinary_mz': TrinaryTask('mz', stdev=tc['trinary_mz']['stdev']),
    'trinary_ab': TrinaryTask(
        'ab', stdev=tc['trinary_ab']['stdev'], clip_vals=[0., 1.]
    ),
    'hidden_mz': HiddenPeak(
        'mz', loss_weight=tc['hidden_mz']['loss_weight'],
        binsz=tc['hidden_mz']['binsz'], mzlims=tc['hidden_mz']['mzlims'],
    ),
    'hidden_ab': HiddenPeak(
        'ab', loss_weight=tc['hidden_ab']['loss_weight'],
        binsz=tc['hidden_ab']['binsz']
    ),
    'hidden_spectrum': HiddenAbSpectrum(
        loss_weight=tc['hidden_spectrum']['loss_weight']
    ),
    'hidden_charge': HiddenCharge(
        max_charge=tc['hidden_charge']['max_charge'],
        loss_weight=tc['hidden_charge']['loss_weight']
    ),
    'hidden_mass': HiddenMass(loss_weight=tc['hidden_charge']['loss_weight']),
    'doctored_mz': DoctoredMZ(
        sys_std=tc['doctored_mz']['sys_std'],
        rand_std=tc['doctored_mz']['rand_std'],
        loss_weight=tc['doctored_mz']['loss_weight'],
    ),
    'doctored_mz2': DoctoredMZ2(
        std=tc['doctored_mz2']['std'],
        loss_weight=tc['doctored_mz2']['loss_weight']
    )
}
