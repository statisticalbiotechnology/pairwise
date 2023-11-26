
import model_parts as mp
import torch as th
from torch import nn

class Encoder(nn.Module):
    def __init__(self,
                 in_units=2, # input units from mz/ab tensor
                 running_units=512, # num units running throughout model
                 sequence_length=100, # maximum number of peaks
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
        self.sl = sequence_length
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
        self.alpha = nn.Parameter(th.tensor(0.1), requires_grad=True)
        
        mdim = mz_units//4 if subdivide else mz_units
        self.MzSeq = nn.Sequential(
            nn.Linear(mdim, mdim), nn.SiLU()
        )
        
        # charge/energy/mass embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            num = sum([use_charge, use_energy, use_mass])
            self.ce_emb = nn.Sequential(
                nn.Linear(ce_units*num, ce_units), nn.SiLU()
            )
        
        # First transformation
        self.first = nn.Linear(mz_units+ab_units, running_units)
        
        # Main block
        attention_dict = {'indim': running_units, 'd': att_d, 'h': att_h}
        ffn_dict = {'indim': running_units, 'unit_multiplier': ffn_multiplier}
        is_embed = True if self.atleast1 else False
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                norm_type, 
                prenorm, 
                is_embed, 
                ce_units,
                preembed
            ) 
            for _ in range(depth)
        ])
        self.main_proj = nn.Identity()#L.Dense(embedding_units, kernel_initializer=I.Zeros())
        
        # Normalization type
        self.norm = mp.get_norm_type(norm_type)
        
        # Recycling embedder
        self.recyc = nn.Sequential(
            self.norm(running_units) if prenorm else nn.Identity(),
            nn.Linear(running_units, running_units) if False else nn.Identity(),
            nn.Identity() if prenorm else self.norm(running_units)
        ) if self.its > 1 else nn.Identity()
        
        """# Recycling modulator
        self.alphacyc = ( 
            tf.Variable(1. / self.its, trainable=True) 
            if self.its > 1 else 
            tf.Variable(1.0, trainable=False)
        )"""
        
        self.global_step = nn.Parameter(th.tensor(0), requires_grad=False)
        
        self.pos = mp.FourierFeatures(
            th.arange(self.sl, dtype=th.float32), self.run_units, 5.*self.sl
        )
    
    def total_params(self):
        return sum([m.numel() for m in self.parameters()])
    
    def MzAb(self, x, inp_mask=None):
        mz, ab = th.split(x, 1, -1)
        
        mz = mz.squeeze()
        if self.subdivide:
            mz = mp.subdivide_float(mz)
            minidim = self.mz_units//4
            mz_emb = mp.FourierFeatures(mz, minidim, 1000.)
        else:
            mz_emb = mp.FourierFeatures(mz, self.mz_units, 10000.)
        mz_emb = self.MzSeq(mz_emb) # multiply sequential to mz fourier feature
        mz_emb = mz_emb.reshape(x.shape[0], x.shape[1], -1)
        # ASSUME ab comes in 0-1, multiply by 100 (0-100) before expansion
        ab_emb = mp.FourierFeatures(100*ab[...,0], self.ab_units, 500.)
        
        # Apply input mask, if necessary
        if inp_mask is not None:
            mz_mask, ab_mask = th.split(inp_mask, 1, -1)
            # Zeros out the feature's entire Fourier vector
            mz_emb *= mz_mask
            ab_emb *= ab_mask
            
        out = th.cat([mz_emb, ab_emb], dim=-1)

        return out
    
    def Main(self, inp, embed=None, mask=None):
        out = inp
        for layer in self.main:
            out = layer(out, embed_feats=embed, spec_mask=mask)
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
            grid = th.tile(
                th.arange(self.sl, dtype=th.int32)[None], 
                (x.shape[0], 1)
            ) # bs, seq_len
            mask = grid >= length[:, None]
            mask = (1e5*mask).type(th.float32)
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
                charge = charge.type(th.float32)
                ce_emb.append(mp.FourierFeatures(charge, self.ce_units, 10.))
            if self.use_energy:
                ce_emb.append(mp.FourierFeatures(energy, self.ce_units, 150.))
            if self.use_mass:
                ce_emb.append(mp.FourierFeatures(mass, self.ce_units, 20000.))
            # tf.concat works if list is 1 or multiple members
            ce_emb = th.cat(ce_emb, dim=-1)
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
    
    def forward(self, 
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
            if emb is not None else 
            th.zeros(x.shape[0], self.sl, self.run_units)
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
        
        return Output
