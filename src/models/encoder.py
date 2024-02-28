
import models.model_parts as mp
import models.model_parts_pw as pw
import torch as th
from torch import nn
I = nn.init

def init_encoder_weights(module):
    if hasattr(module, 'first'):
        module.first.weight = I.xavier_uniform_(module.first.weight)
        if module.first.bias is not None: 
            module.first.bias = I.zeros_(module.first.bias)
    if isinstance(module, mp.SelfAttention):
        module.qkv.weight = I.normal_(module.qkv.weight, 0.0, (2/3)*module.indim**-0.5)
        module.Wo.weight = I.normal_(module.Wo.weight, 0.0, (1/3)*(module.h*module.d)**-0.5)
        if hasattr(module, 'Wb'):
            module.Wb.weight = I.zeros_(module.Wb.weight)
            module.Wb.bias = I.zeros_(module.Wb.bias)
        elif hasattr(module, 'Wpw'):
            module.Wpw.weight = I.zeros_(module.Wpw.weight)
            module.Wpw.bias = I.zeros_(module.Wpw.bias)
        if hasattr(module, 'Wg'):
            module.Wg.weight = I.zeros_(module.Wg.weight)
            module.Wg.bias = I.constant_(module.Wg.bias, 1.) # gate mostly open ~ 0.73
    elif isinstance(module, mp.FFN):
        module.W1.weight = I.xavier_uniform_(module.W1.weight)
        module.W1.bias = I.zeros_(module.W1.bias)
        module.W2.weight = I.normal_(module.W2.weight, 0.0, (1/3)*(module.indim*module.mult)**-0.5)
        #module.W2.weight = I.xavier_uniform_(module.W2.weight)
    elif isinstance(module, nn.Linear):
        module.weight = I.xavier_uniform_(module.weight)
        if module.bias is not None:
            module.bias = I.zeros_(module.bias)
    

class Encoder(nn.Module):
    def __init__(self,
                 # 1D options
                 in_units=2, # input units from mz/ab tensor
                 running_units=512, # num units running throughout model
                 sequence_length=100, # maximum number of peaks
                 mz_units=512, # units in mz fourier vector
                 ab_units=256, # units in ab fourier vector
                 subdivide=False, # subdivide mz units in 2s and expand-concat
                 use_charge=False, # inject charge into TransBlocks
                 use_energy=False, # inject energy into TransBlocks
                 use_mass=False, # injuect mass into TransBlocks
                 ce_units=256, # units for transformation of mzab fourier vectors
                 att_d=64, # attention qkv dimension units
                 att_h=4,  # attention qkv heads
                 gate=False, # input dependent gate following weights
                 ffn_multiplier=4, # multiply inp units for 1st FFN transform
                 prenorm=True, # normalization before attention/ffn layers
                 norm_type='layer', # normalization type
                 preembed=True, # embed/add charge/energy/mass before FFN
                 depth=9, # number of transblocks
                 dropout=0, # dropout on residual in SelfAtt and FFN
                 bias=False, # Att. bias: 'regular' | 'pairwise' | False/None 
                 # Pairwise options
                 pw_mz_units=None, # sinusoidal units to expand pw tensor into
                 pw_run_units=None, # units to project pw tensor to after sinusoidal expansion
                 pw_attention_ch=32, # triangle attention channels
                 pw_attention_h=4, # triangle attention heads
                 pw_blocks=2, # number of pairstack blocks for pairwise features
                 # Miscellaneous
                 recycling_its=1, # recycling iterations
                 device=th.device('cpu')
                 ):
        super(Encoder, self).__init__()
        self.running_units = running_units
        self.sl = sequence_length
        self.mz_units = mz_units
        self.ab_units = ab_units
        self.subdivide = subdivide
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass
        self.ce_units = ce_units
        self.d = att_d
        self.h = self.nhead = att_h
        self.bias = bias
        self.dropout=dropout
        self.pw_mzunits = mz_units if pw_mz_units==None else pw_mz_units
        self.pw_runits = running_units if pw_run_units==None else pw_run_units
        self.depth = depth
        self.prenorm = prenorm
        self.norm_type = norm_type
        self.preembed = preembed
        self.its = recycling_its
        self.device = device
        # compat
        self.dim_feedforward = ffn_multiplier*running_units
        self.encode_peaks = self.MzAb
        
        # Position modulation
        self.alpha = nn.Parameter(th.tensor(0.1), requires_grad=True)
        
        mdim = mz_units//4 if subdivide else mz_units
        self.mdim = mdim
        self.MzSeq = nn.Identity() # nn.Sequential(nn.Linear(mdim, mdim), nn.SiLU())

        # Pairwise mz
        if bias == 'pairwise':
            # - subidvide and expand based on mz_units, transform to pw_units
            mdimpw = self.pw_mzunits//4 if subdivide else self.pw_mzunits
            self.mdimpw = mdimpw
            self.MzpwSeq = nn.Identity()#Sequential(nn.Linear(mdimpw, mdimpw), nn.SiLU())
            self.pwfirst = nn.Linear(self.pw_mzunits, self.pw_runits)
            self.alphapw = nn.Parameter(th.tensor(0.1), requires_grad=True)
            #self.pospw = pw.RelPos(sequence_length, self.pw_runits)
            # Evolve features
            multdict = {'in_dim': self.pw_runits, 'c': 128}
            attdict = {'in_dim': self.pw_runits, 'c': pw_attention_ch, 'h': pw_attention_h}
            ptdict = {'in_dim': self.pw_runits, 'n': 4}
            self.PwSeq = nn.Sequential(*[
                pw.PairStack(multdict, attdict, ptdict, drop_rate=0)
                for m in range(pw_blocks)
            ])

        # charge/energy/mass embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            num = sum([use_charge, use_energy, use_mass])
            self.ce_emb = nn.Sequential(
                nn.Linear(ce_units*num, ce_units), nn.SiLU()
            )
        
        # First transformation
        self.first = nn.Linear(mz_units+ab_units, running_units, bias=False)

        # Main block
        assert bias in ['pairwise', 'regular', False, None]
        if bias == None: bias = False
        attention_dict = {
            'indim': running_units, 
            'd': att_d, 
            'h': att_h,
            'bias': bias,
            'bias_in_units': self.pw_runits,
            'modulator': False,
            'gate': gate,
            'dropout': dropout,
        }
        ffn_dict = {
            'indim': running_units, 
            'unit_multiplier': ffn_multiplier, 
            'dropout': dropout
        }
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
        
        #pos = mp.FourierFeatures(
        #    th.arange(1000, dtype=th.float32), 1, 5000, self.running_units,
        #)
        #self.pos = nn.Parameter(pos, requires_grad=False)

        self.apply(init_encoder_weights)
    
    def total_params(self):
        return sum([m.numel() for m in self.parameters()])

    def MzAb(self, x):
        Mz, ab = th.split(x, 1, -1)
        
        # 1D features
        Mz = Mz.squeeze()
        if self.subdivide:
            mz = mp.subdivide_float(Mz)
            mz_emb = mp.FourierFeatures(mz, 1, 500, self.mdim)
        else:
            mz_emb = mp.FourierFeatures(Mz, 0.001, 10000, self.mz_units)
        mz_emb = self.MzSeq(mz_emb) # multiply sequential to mz fourier feature
        mz_emb = mz_emb.reshape(x.shape[0], x.shape[1], -1)
        ab_emb = mp.FourierFeatures(ab[...,0], 0.000001, 1, self.ab_units)
        
        out = th.cat([mz_emb, ab_emb], dim=-1)

        # Pairwise features
        if self.bias == 'pairwise':
            dtsr = mp.delta_tensor(Mz, 0.)
            # expand based on mz_units
            if self.subdivide:
                mzpw = mp.subdivide_float(dtsr)
                mzpw_emb = mp.FourierFeatures(mzpw, 1, 500, self.mdimpw)
            else:
                mzpw_emb = mp.FourierFeatures(dtsr, 0.001, 10000, self.pw_mzunits)
            # transform based on pw_units
            mzpw_emb = self.MzpwSeq(mzpw_emb)
            mzpw_emb = mzpw_emb.reshape(x.shape[0], x.shape[1], x.shape[1], -1)
        else:
            mzpw_emb = None

        return {'1d': out, '2d': mzpw_emb}
    
    def Main(self, inp, embed=None, mask=None, pwtsr=None, return_full=False):
        out = inp
        other = []
        for layer in self.main:
            out = layer(
                out, 
                embed_feats=embed, 
                spec_mask=mask, 
                biastsr=pwtsr, 
                return_full=return_full
            )
            other.append(out['other'])
            out = out['out']
        return {'out': self.main_proj(out), 'other': other}
    
    def UpdateEmbed(self, 
                    x, 
                    charge=None, 
                    energy=None, 
                    mass=None,
                    #length=None,
                    #inp_mask=None,
                    #tag_array=None,
                    key_padding_mask=None,
                    emb=None,
                    return_mask=False,
                    return_full=False,
                    ):
        # Create mask
        if key_padding_mask != None:
            """
            grid = th.tile(
                th.arange(self.sl, dtype=th.int32)[None].to(x.device), 
                (x.shape[0], 1)
            ) # bs, seq_len
            mask = grid >= length[:, None]
            mask = (1e7*mask).type(th.float32)
            """
            mask = 1e7*key_padding_mask.type(th.float32)
        else:
            mask = None
        
        # Spectrum level embeddings
        if self.atleast1:
            ce_emb = []
            if self.use_charge:
                charge = charge.type(th.float32)
                ce_emb.append(mp.FourierFeatures(charge, 1, 50, self.ce_units))
            if self.use_energy:
                ce_emb.append(mp.FourierFeatures(energy, self.ce_units, 150.))
            if self.use_mass:
                ce_emb.append(mp.FourierFeatures(mass, 0.001, 10000, self.ce_units))
            # tf.concat works if list is 1 or multiple members
            ce_emb = th.cat(ce_emb, dim=-1)
            ce_emb = self.ce_emb(ce_emb)
        else:
            ce_emb = None
        
        # Feed forward
        mzab_dic = self.MzAb(x)
        mabemb = mzab_dic['1d']
        pwemb = mzab_dic['2d']
        if self.bias == 'pairwise':
            pwemb = self.pwfirst(pwemb)# + self.alphapw * self.pospw()
            pwemb = self.PwSeq(pwemb)
        
        out = self.first(mabemb)# + self.alpha*self.pos[:x.shape[1]]
        
        # Reycling the embedding with normalization, perhaps dense transform
        out += self.recyc(emb)
        
        out = self.Main(out, embed=ce_emb, mask=mask, pwtsr=pwemb, return_full=return_full) # AlphaFold has +=
        emb = out['out']
        
        output = {'emb': emb, 'mask': key_padding_mask, 'num_cem_tokens': 0}#'other': out['other']}
        
        return output
    
    def forward(self, 
             x,
             charge=None, 
             energy=None, 
             mass=None,
             length=None,  
             #inp_mask=None,
             #tag_array=None,
             key_padding_mask=None,
             emb=None,
             its=None, 
             return_mask=False,
             return_full=False
    ):
        its = self.its  if its==None else its
        
        # Recycled embedding
        emb = (
            emb 
            if emb is not None else 
            th.zeros(x.shape[0], x.shape[1], self.running_units)
        ).to(x.device)
        
        for _ in range(its):
            output = self.UpdateEmbed(
                x, 
                charge=charge, 
                energy=energy, 
                mass=mass, 
                #length=length,  
                #inp_mask=inp_mask, 
                #tag_array=tag_array,
                key_padding_mask=key_padding_mask,
                emb=emb,
                return_mask=return_mask,
                return_full=return_full
            )
        
        return output

def encoder_base_arch(
    use_charge=False,
    use_mass=False,
    use_energy=False,
    bias=False, # 'pairwise' | 'regular' | False | None
):
    model = Encoder(
        norm_type='layer',
        mz_units=1024,
        ab_units=256,
        subdivide=True,
        running_units=512,
        att_d=64,
        att_h=8,
        depth=9,
        ffn_multiplier=4,
        prenorm=True,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        dropout=0,
        bias=bias,
        gate=False,

        pw_mz_units=512,
        pw_run_units=64,
        pw_attention_ch=32,
        pw_attention_h=4,
        pw_blocks=1
    )
    
    return model

