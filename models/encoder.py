
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

class Encoder(K.Model):
    def __init__(self,
                 running_units=512, # num units running throughout model
                 mz_units=512, # units in mz fourier vector
                 ab_units=256, # units in ab fourier vector
                 subdivide=False, # subdivide mz units in 2s and expand-concat
                 use_charge=True, # inject charge into TransBlocks
                 use_energy=False, # inject energy into TransBlocks
                 use_mass=True, # injuect mass into TransBlocks
                 ce_units=256, # units for transformation of mzab fourier vectors
                 att_d=64, # attention qkv dimension units
                 att_h=4,  # attention qkv heads
                 ffn_multiplier=4, # multiply inp units for 1st FFN transform
                 depth=9, # number of transblocks
                 prenorm=True, # normalization before attention/ffn layers
                 norm_type='layer', # normalization type
                 preembed=True, # embed/add charge/energy/mass before FFN 
                 recycling_its=1, # recycling iterations
                 ):
        super(Encoder, self).__init__()
        self.run_units = running_units
        self.mz_units = mz_units
        self.ab_units = ab_units
        self.subdivide = subdivide
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass
        self.ce_units = ce_units
        self.d = att_d
        self.h = att_h
        self.depth = depth
        self.prenorm = prenorm
        self.norm_type = norm_type
        self.preembed = preembed
        self.its = recycling_its
        
        # Position modulation
        self.alpha = tf.Variable(0.1, trainable=True)
        
        self.MzSeq = K.Sequential([
            L.Dense(mz_units // 4),
            mp.KerasActLayer(tf.nn.swish),
        ])

        # charge/energy/mass embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            self.ce_emb = K.Sequential([
                L.Dense(ce_units), mp.KerasActLayer(tf.nn.swish)
            ])

        # First transformation
        self.first = L.Dense(running_units)#, kernel_initializer=I.Zeros())
        
        # Main block
        attention_dict = {'d': att_d, 'h': att_h}
        ffn_dict = {'unit_multiplier': ffn_multiplier}
        is_embed = True if self.atleast1 else False
        self.main = [
            mp.TransBlock(
                attention_dict, ffn_dict, norm_type, prenorm, is_embed, preembed
            ) 
            for _ in range(depth)
        ]
        self.main_proj = A.linear#L.Dense(embedding_units, kernel_initializer=I.Zeros())
        
        # Normalization type
        self.norm = mp.get_norm_type(norm_type)
        
        # Recycling embedder
        self.recyc = K.Sequential([
            self.norm() if prenorm else mp.KerasActLayer(A.linear),
            L.Dense(running_units) if False else mp.KerasActLayer(A.linear),
            mp.KerasActLayer(A.linear) if prenorm else self.norm()
        ]) if self.its > 1 else A.linear
        
        """# Recycling modulator
        self.alphacyc = ( 
            tf.Variable(1. / self.its, trainable=True) 
            if self.its > 1 else 
            tf.Variable(1.0, trainable=False)
        )"""
        
        self.global_step = tf.Variable(0, trainable=False)
        
    def build(self, x):
        self.sl = x[1]
        
        # Positional embedding
        self.pos = mp.FourierFeatures(
            tf.range(self.sl, dtype=tf.float32), self.run_units, 5.*self.sl
        )
        
        """# Tag
        self.tag = tf.zeros((1, self.sl), tf.int32)"""

        """self.tensor = tf.Variable(
            tf.random.normal((x[1], self.embed_units)), trainable=True 
        )"""
    
    """def PredictTag(self, predict_array):
        return tf.cast(tf.one_hot(predict_array, 2), tf.float32)"""

    def MzAb(self, x, inp_mask=None):
        mz, ab = tf.split(x, 2, -1)
        
        mz = tf.squeeze(mz)
        if self.subdivide:
            mz = mp.subdivide_float(mz)
            minidim = self.mz_units//4
            mz_emb = mp.FourierFeatures(mz, minidim, 1000.)
        else:
            mz_emb = mp.FourierFeatures(mz, self.mz_units, 10000.)
        mz_emb = self.MzSeq(mz_emb) # multiply sequential to mz fourier feature
        mz_emb = tf.reshape(mz_emb, (x.shape[0], x.shape[1], -1))
        # ASSUME ab comes in 0-1, multiply by 100 (0-100) before expansion
        ab_emb = mp.FourierFeatures(100*ab[...,0], self.ab_units, 500.)
        
        # Apply input mask, if necessary
        if inp_mask is not None:
            mz_mask, ab_mask = tf.split(inp_mask, 2, -1)
            # Zeros out the feature's entire Fourier vector
            mz_emb *= mz_mask
            ab_emb *= ab_mask
            
        out = tf.concat([mz_emb, ab_emb], axis=-1)

        return out
    
    def total_params(self):
        return sum([np.prod(m.shape) for m in self.trainable_variables])
    
    def Main(self, inp, embed=None, mask=None):
        out = inp
        for layer in self.main:
            out = layer(out, temb=embed, spec_mask=mask)
        return self.main_proj(out)
    
    def UpdateEmbed(self, 
                    x, 
                    charge=None, 
                    energy=None, 
                    mass=None,
                    length=None, 
                    emb=None,
                    inp_mask=None,
                    tag_array=None,
                    return_mask=False
                    ):
        # Create mask
        if length!=None:
            grid = tf.tile(
                tf.range(self.sl, dtype=tf.int32)[None], 
                (x.shape[0], 1)
            ) # bs, seq_len
            mask = grid >= length[:, None]
            mask = 1e5*tf.cast(mask, tf.float32)
        else:
            mask = None
        
        """# Create tag for prediction of missing/altered inputs
        if tag_array==None:
            # Nothing altered (mzab) by default
            tag_array = tf.tile(self.tag, [tf.shape(x)[0], 1])
        TagArray = self.PredictTag(tag_array) # bs, seq_len, 2 (float32)"""

        # Spectrum level embeddings
        if self.atleast1:
            ce_emb = []
            if self.use_charge:
                charge = tf.cast(charge, tf.float32)
                ce_emb.append(mp.FourierFeatures(charge, self.ce_units, 10.))
            if self.use_energy:
                ce_emb.append(mp.FourierFeatures(energy, self.ce_units, 150.))
            if self.use_mass:
                ce_emb.append(mp.FourierFeatures(mass, self.ce_units, 20000.))
            # tf.concat works if list is 1 or multiple members
            ce_emb = tf.concat(ce_emb, axis=-1)
            ce_emb = self.ce_emb(ce_emb)
        else:
            ce_emb = None
        
        # Feed forward
        mabemb = self.MzAb(x, inp_mask)
        """mabemb = tf.concat([mabemb, TagArray], axis=-1) # add before self.first"""
        out = self.first(mabemb) + self.alpha*self.pos
        
        # Reycling the embedding with normalization, perhaps dense transform
        out += self.recyc(emb)
        
        emb = self.Main(out, embed=ce_emb, mask=mask) # AlphaFold has +=
        
        output = (emb, mask) if return_mask else emb
        
        return output
       
    def call(self, 
             x,
             charge=None, 
             energy=None, 
             mass=None,
             length=None, 
             emb=None, 
             inp_mask=None,
             tag_array=None,
             its=None, 
             return_mask=False
    ):
        Output = {'final': None, 'emb': None, 'mask': None}
        its = self.its  if its==None else its
        
        # Recycled embedding
        emb = (
            emb 
            if tf.is_tensor(emb) else 
            tf.zeros((x.shape[0], self.sl, self.run_units))
        )
        
        for _ in range(its):
            output = self.UpdateEmbed(
                x, charge=charge, energy=energy, mass=mass, 
                length=length, emb=emb, inp_mask=inp_mask, 
                tag_array=tag_array, return_mask=return_mask
            )
            emb, mask = output if return_mask else (output,None)
        Output['emb'] = emb
        if return_mask: Output['mask'] = mask
        
        out = emb
        
        # output = out, mask if return_mask else out
        
        return Output

