# -*- coding: utf-8 -*-
"""
TODO: 
Create new head to predict mz and ab
"""

from copy import deepcopy
import numpy as np
import tensorflow as tf
K = tf.keras
L = K.layers
A = K.activations
I = K.initializers
gelu = tf.keras.activations.gelu

def get_norm_type(string):
    if string=='layer':
        return L.LayerNormalization
    elif string=='batch':
        return L.BatchNormalization

class QKVAttention(L.Layer):
    def __init__(self, heads, is_relpos=False, max_rel_dist=None):
        super(QKVAttention, self).__init__()
        self.heads = heads
        self.is_relpos = is_relpos
        self.maxd = max_rel_dist
    
    def build_relpos_tensor(self, seq_len, maxd=None):
        # This is the equivalent of an input dependent ij bias, rather than one
        # that is a direclty learned ij tensor
        maxd = seq_len-1 if maxd==None else (maxd-1 if maxd==seq_len else maxd)
        a = tf.range(seq_len, dtype=tf.int32)
        b = tf.range(seq_len, dtype=tf.int32)
        relpos = a[:,None] - b[None]
        tsr = tf.random.normal((2*seq_len-1, self.dim), 0, seq_len**-0.5)
        # set maximum distance
        relpos = tf.clip_by_value(relpos, -maxd, maxd)
        relpos += maxd
        tsr = tf.nn.embedding_lookup(tsr, relpos)
        relpos_tsr = tf.Variable(tsr, trainable=True)
        
        return relpos_tsr
    
    def build(self, x):
        self.sl = x[1]
        self.dim = x[-1]
        self.scale = 1 / self.dim**0.25
        if self.is_relpos:
            self.ak = self.build_relpos_tensor(x[1], self.maxd) # sl, sl, dim
            self.av = self.build_relpos_tensor(x[1], self.maxd) # sl, sl, dim
    
    def call(self, Q, K, V, mask=None):
        # shape: batch/heads/etc, sequence_length, dim
        QK = tf.einsum('abc,adc->abd', self.scale*Q, self.scale*K)
        if self.is_relpos:
            QK += tf.einsum('abc,bec->abe', Q, self.ak)
        QK = tf.reshape(QK, (-1, self.heads, self.sl, K.shape[1]))
        
        # mask.shape: bs, 1, 1, sl
        mask = tf.zeros_like(QK) if mask==None else mask[:,None,None,:]
        weights = A.softmax(QK-mask, axis=-1)
        weights = tf.reshape(weights, (-1, self.sl, V.shape[1]))
        
        att = tf.einsum('abc,acd->abd', weights, V)
        if self.is_relpos:
            att += tf.einsum('abc,bcd->abd', weights, self.av)
        
        return att
    

class SelfAttention(L.Layer):
    def __init__(self, d, h, out_units=None):
        super(SelfAttention, self).__init__()
        self.d = d
        self.h = h
        self.out_units = out_units
        
        self.qkv = L.Dense(3*d*h, use_bias=False)
        self.attention_layer = QKVAttention(h)
    
    def get_qkv(self, qkv):
        Q, K, V = tf.split(qkv, 3, axis=-1) # per: bs, sl, d*h 
        Q = tf.reshape(Q, (-1, self.sl, self.d, self.h)) # bs, sl, d, h
        Q = tf.reshape(tf.transpose(Q, (0,3,1,2)), (-1, self.sl, self.d)) 
        K = tf.reshape(K, (-1, self.sl, self.d, self.h)) # bs, sl, d, h
        K = tf.reshape(tf.transpose(K, (0,3,1,2)), (-1, self.sl, self.d))
        V = tf.reshape(V, (-1, self.sl, self.d, self.h)) # bs, sl, d, h
        V = tf.reshape(tf.transpose(V, (0,3,1,2)), (-1, self.sl, self.d))
        
        return Q, K, V # bs*h, sl, d
    
    def build(self, x):
        self.sl = x[1]
        self.out_units = x[-1] if self.out_units==None else self.out_units
        self.Wo = L.Dense(
            self.out_units, use_bias=False,
            kernel_initializer=I.RandomNormal(0, 0.3*(self.h*self.d)**-0.5)
        )
        self.shortcut = (
            A.linear 
            if self.out_units==x[-1] else 
            L.Dense(self.out_units, use_bias=False)
        )
    
    def call(self, x, mask=None):
        # x.shape; bs, sl, units
        qkv = self.qkv(x) # bs, sl, 3*d*h
        Q, K, V = self.get_qkv(qkv) # bs*h, sl, d
        att = self.attention_layer(Q, K, V, mask) # bs*h, sl,  d
        att = tf.reshape( 
            tf.transpose(
                tf.reshape(att, (-1, self.h, self.sl, self.d)), (0,2,3,1)
            ), (-1, self.sl, self.d*self.h)
        ) # bs, sl, d*h
        resid = self.Wo(att) # bs, sl, out_units
        
        return self.shortcut(x) + resid

class CrossAttention(L.Layer):
    def __init__(self, d, h, out_units=None):
        super(CrossAttention, self).__init__()
        self.d = d
        self.h = h
        self.out_units = out_units
        
        self.Wq = L.Dense(d*h, use_bias=False)
        self.Wkv = L.Dense(2*d*h, use_bias=False)
        
        self.attention_layer = QKVAttention(h)
    
    def get_qkv(self, q, kv):
        bs, sl, units = q.shape
        bs, sl2, units = kv.shape
        Q = tf.reshape(q, (bs, sl, self.d, self.h))
        Q = tf.reshape(tf.transpose(Q, [0,3,1,2]), (-1, sl, self.d))
        K, V = tf.split(kv, 2, -1)
        K = tf.reshape(K, (bs, sl2, self.d, self.h))
        K = tf.reshape(tf.transpose(K, [0,3,1,2]), (-1, sl2, self.d))
        V = tf.reshape(V, (bs, sl2, self.d, self.h))
        V = tf.reshape(tf.transpose(V, [0,3,1,2]), (-1, sl2, self.d))
        
        return Q, K, V
    
    def build(self, x):
        self.sl = x[1]
        self.out_units = x[-1] if self.out_units==None else self.out_units
        self.Wo = L.Dense(
            self.out_units, use_bias=False,
            kernel_initializer=I.RandomNormal(0, 0.3*(self.h*self.d)**-0.5)
        )
        self.shortcut = (
            A.linear
            if self.out_units==x[-1] else
            L.Dense(self.out_units, use_bias=False)
        )
    
    def call(self, q_feats, kv_feats, mask=None):
        Q = self.Wq(q_feats)
        KV = self.Wkv(kv_feats)
        Q, K, V = self.get_qkv(Q, KV)
        att = self.attention_layer(Q, K, V, mask)
        att = tf.reshape(
            tf.transpose(
                tf.reshape(att, (-1, self.h, self.sl, self.d)), [0,2,1,3]
            ), (-1, self.sl, self.h*self.d)
        )
        resid = self.Wo(att)
        
        return self.shortcut(q_feats) + resid

class FFN(L.Layer):
    def __init__(self, unit_multiplier=1, out_units=None):
        super(FFN, self).__init__()
        self.mult = unit_multiplier
        self.out_units = out_units
    
    def build(self, x):
        self.out_units = x[-1] if self.out_units==None else self.out_units
        self.W1 = L.Dense(x[-1]*self.mult)
        self.W2 = L.Dense(
            self.out_units, use_bias=False, 
            kernel_initializer=I.RandomNormal(0, 0.3*(x[-1]*self.mult)**-0.5)
        )
    
    def call(self, x, embed=None):
        out = self.W1(x)
        out = A.relu(out + (0 if embed==None else embed))
        out = self.W2(out)
        
        return x + out

class TransBlock(L.Layer):
    def __init__(self,
                 attention_dict,
                 ffn_dict,
                 norm_type='layer', 
                 prenorm=True,
                 is_embed=False,
                 preembed=True,
                 is_cross=False
    ):
        super(TransBlock, self).__init__()
        self.norm_type = norm_type
        self.mult = ffn_dict['unit_multiplier']
        self.prenorm = prenorm
        self.is_embed = is_embed
        self.preembed = preembed
        self.is_cross = is_cross
        
        if preembed: self.alpha = tf.Variable(0.1, trainable=True)
        norm = get_norm_type(norm_type)
        
        self.norm1 = norm()
        self.norm2 = norm()
        self.selfattention = SelfAttention(**attention_dict)
        if is_cross:
            self.crossnorm = norm()
            self.crossattention = CrossAttention(**attention_dict)
        self.ffn = FFN(**ffn_dict)
    
    def build(self, x):
        if self.is_embed:
            units = x[-1] if self.preembed else x[-1]*self.mult
            self.embed = L.Dense(units)
    
    def call(self, x, kv_feats=None, temb=None, spec_mask=None, seq_mask=None):
        selfmask = seq_mask if self.is_cross else spec_mask
        Temb = self.embed(temb)[:,None,:] if self.is_embed else 0
        
        out = x + self.alpha*Temb if self.preembed else x
        out = self.norm1(out) if self.prenorm else out
        out = self.selfattention(out, selfmask)
        if self.is_cross:
            out = self.crossnorm(out) if self.prenorm else out
            out = self.crossattention(out, kv_feats, spec_mask)
            out = out if self.prenorm else self.crossnorm(out)
        out = self.norm2(out) if self.prenorm else self.norm1(out)
        out = self.ffn(out, None) if self.preembed else self.ffn(out, Temb)
        out = out if self.prenorm else self.norm2(out)
        
        return out

class KerasActLayer(L.Layer):
    def __init__(self, activation):
        super(KerasActLayer, self).__init__()
        self.act = activation
    def call(self, x):
        return self.act(x)

def FourierFeatures(t, embedsz, freq=10000.):
    # t.shape (bs,)
    embed = tf.cast(t[...,None], tf.float32) * tf.exp(
        -tf.math.log(float(freq)) * 
        tf.range(embedsz//2, dtype=tf.float32) / 
        (embedsz//2)
    )[None]
    
    return tf.concat([tf.cos(embed), tf.sin(embed)], axis=-1)

def subdivide_float(x):
    a = tf.math.floordiv(x, 100)
    b = tf.math.floordiv(x-a*100, 1)
    x_ = tf.round(10000*(x - tf.math.floordiv(x, 1)))
    c = tf.math.floordiv(x_, 100)
    d = tf.math.floordiv(x_-c*100, 1)
    
    return tf.concat([a[...,None], b[...,None], c[...,None], d[...,None]], -1)

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
            KerasActLayer(tf.nn.swish),
        ])

        # charge/energy/mass embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            self.ce_emb = K.Sequential([
                L.Dense(ce_units), KerasActLayer(tf.nn.swish)
            ])

        # First transformation
        self.first = L.Dense(running_units)#, kernel_initializer=I.Zeros())
        
        # Main block
        attention_dict = {'d': att_d, 'h': att_h}
        ffn_dict = {'unit_multiplier': ffn_multiplier}
        is_embed = True if self.atleast1 else False
        self.main = [
            TransBlock(
                attention_dict, ffn_dict, norm_type, prenorm, is_embed, preembed
            ) 
            for _ in range(depth)
        ]
        self.main_proj = A.linear#L.Dense(embedding_units, kernel_initializer=I.Zeros())
        
        # Normalization type
        self.norm = get_norm_type(norm_type)
        
        # Recycling embedder
        self.recyc = K.Sequential([
            self.norm() if prenorm else KerasActLayer(A.linear),
            L.Dense(running_units) if False else KerasActLayer(A.linear),
            KerasActLayer(A.linear) if prenorm else self.norm()
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
        self.pos = FourierFeatures(
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
            mz = subdivide_float(mz)
            minidim = self.mz_units//4
            mz_emb = FourierFeatures(mz, minidim, 1000.)
        else:
            mz_emb = FourierFeatures(mz, self.mz_units, 10000.)
        mz_emb = self.MzSeq(mz_emb) # multiply sequential to mz fourier feature
        mz_emb = tf.reshape(mz_emb, (x.shape[0], x.shape[1], -1))
        # ASSUME ab comes in 0-1, multiply by 100 (0-100) before expansion
        ab_emb = FourierFeatures(tf.squeeze(100*ab), self.ab_units, 500.)
        
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
            mask = 200*tf.cast(mask, tf.float32)
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
                ce_emb.append(FourierFeatures(charge, self.ce_units, 10.))
            if self.use_energy:
                ce_emb.append(FourierFeatures(energy, self.ce_units, 150.))
            if self.use_mass:
                ce_emb.append(FourierFeatures(mass, self.ce_units, 20000.))
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
        self.norm = get_norm_type(norm_type)

        # First embeddings
        self.seq_emb = L.Embedding(num_inp_tokens, embedding_units)
        self.alpha = tf.Variable(0.1, trainable=True)
        
        # charge/energy embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            self.ce_emb = K.Sequential([
                L.Dense(ce_units), KerasActLayer(tf.nn.swish)
            ])
        
        # Main blocks
        attention_dict = {'d': d, 'h': h}
        ffn_dict = {'unit_multiplier': ffn_multiplier}
        is_embed = True if self.atleast1 else False
        self.main = [
            TransBlock(
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
            KerasActLayer(gelu),
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
            mask = 200 * tf.cast(seqs >= seqlen[:,None], tf.float32)
        
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
        pos = FourierFeatures(
            tf.range(self.sl, dtype=tf.float32), self.embed_units, 5.*self.sl
        )
        
        # charge and/or energy embedding
        if self.atleast1:
            ce_emb = []
            if self.use_charge:
                charge = tf.cast(charge, tf.float32)
                ce_emb.append(FourierFeatures(charge, self.ce_units, 10.))
            if self.use_energy:
                ce_emb.append(FourierFeatures(energy, self.ce_units, 150.))
            if self.use_mass:
                ce_emb.append(FourierFeatures(mass, self.ce_units, 20000.))
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
        self.inpdict['<s>'] = len(self.inpdict)
        self.start_token = self.inpdict['<s>']
        self.inpdict['<h>'] = len(self.inpdict)
        self.hidden_token = self.inpdict['<h>']
        #self.inpdict['<p>'] = len(self.inpdict)
        #self.pred_token = self.inpdict['<p>']
        dec_config['num_inp_tokens'] = len(self.inpdict)
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

    #def sandwich(self, intseq):
    #    return self.append_nulltok(self.prepend_startok(intseq))

    def initial_intseq(self, batch_size, seqlen=None, include_predtok=False):
        seq_length = self.seq_len if seqlen==None else seqlen
        intseq = tf.fill((batch_size, seq_length-1), self.hidden_token)
        out = self.prepend_startok(intseq) # bs, seq_length
        #out = self.set_tokens(out, int(seq_length+1), self.hidden_token)
        if include_predtok:
            out = self.set_tokens(out, 1, self.pred_token)

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
            'seqlen': seqlens
        }
        dec_out = self.decoder(**dec_inp)
        
        self.trainable_variables = self.decoder.trainable_variables

    def column_inds(self, batch_size, column_ind):
        ind0 = tf.range(batch_size)[:,None]
        ind1 = tf.fill((batch_size, 1), column_ind)
        inds = tf.concat([ind0, ind1], axis=1)

        return inds

    def set_tokens(self, int_array, inds, updates):
        shp = tf.shape(int_array)
        
        if type(inds)==int:
            inds = self.column_inds(shp[0], inds)
        
        if type(updates)==int:
            updates = tf.fill((shp[0],), updates)
        
        out = tf.tensor_scatter_nd_update(int_array, inds, updates)

        return out

    def fill_hidden(self, int_array, inds):
        all_inds = tf.tile(
            tf.range(tf.shape(int_array)[1], dtype=tf.int32)[None],
            [tf.shape(int_array)[0], 1]
        )
        hidden_inds = tf.where(all_inds > inds[:, None])
        out = tf.tensor_scatter_nd_update(
            int_array, 
            hidden_inds, 
            tf.fill((tf.shape(hidden_inds)[0],), self.hidden_token)
        )

        return out

    def decinp(self, intseq, enc_out, charge=None, energy=None, mass=None, training=False):
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
        
            #intseq = self.set_tokens(intseq, index+1, self.pred_token)
            dec_out = self(intseq, enc_out, batdic, False)

            predictions = self.greedy(dec_out[:, index])
            
            if index < self.seq_len-1:
                intseq = self.set_tokens(intseq, index+1, predictions)
        
        intseq = tf.concat([intseq[:, 1:], predictions[:,None]], axis=1)

        return intseq

    def __call__(self, intseq, enc_out, batdic, training=False):
        dec_inp = self.decinp(
            intseq, enc_out,
            charge=batdic['charge'], mass=batdic['mass'], energy=None,
            training=training
        )
        output = self.decoder(**dec_inp)

        return output

"""
class Transformer(K.Model):
    def __init__(self, 
                 encoder_dict,
                 decoder_dict
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(**encoder_dict)
        if decoder_dict==None:
            self.encoder_out = True
            self.decoder = None
        else:
            self.encoder_out = None
            self.decoder = Decoder(**decoder_dict)
        
        self.global_step = tf.Variable(0, trainable=False)

    def total_params(self):
        return sum([np.prod(m.shape) for m in self.trainable_variables])

    def call(self, 
             spectrum, 
             input_sequence=None, 
             charge=None,
             energy=None,
             lengths=None,
             seqlen=None,
             encoder_out=False,
             ):
        Output = {'main': None, 'dec_out': None, 'enc_out': None}
        
        # Override encoder_out, if necessary
        encoder_out = encoder_out or self.encoder_out
        enc = self.encoder(
            spectrum, charge=charge, energy=energy, lengths=lengths, 
            final=encoder_out, return_mask=True
        )
        if encoder_out: Output['enc_out'] = enc['final']
        
        if self.decoder is not None:
            out = self.decoder(
                input_sequence, enc['emb'], charge=charge, energy=energy, 
                specmask=enc['mask'], seqlen=seqlen
            )
            Output['dec_out'] = out
            Output['main'] = Output['dec_out']
        else:
            Output['main'] = Output['enc_out']
        
        return Output
"""

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
        
        norm = get_norm_type(norm_type)

        self.penult = K.Sequential([
            L.Dense(penult_units),
            KerasActLayer(gelu),
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
        norm = get_norm_type(norm_type)
        self.spectrum_wise = spectrum_wise

        self.penult = K.Sequential([
            L.Dense(penult_units),
            KerasActLayer(gelu),
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
        
        norm = get_norm_type(norm_type)

        self.penult = K.Sequential([
            L.Dense(penult_units), 
            KerasActLayer(gelu), 
            norm(),
        ])

        self.final_sl = (
            K.Sequential([
                L.Dense(final_seq_len),
                KerasActLayer(gelu),
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
        
        norm = get_norm_type(norm_type)

        self.penult = K.Sequential([
            L.Dense(penult_units),
            KerasActLayer(gelu),
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


