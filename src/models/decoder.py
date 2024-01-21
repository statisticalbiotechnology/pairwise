from copy import deepcopy
import numpy as np
import models.model_parts as mp
import torch
from torch import nn
import pytorch_lightning as pl
from utils import Scale


class Decoder(pl.LightningModule):
    def __init__(
        self,
        running_units=512,
        kv_indim=256,
        sequence_length=30,  # maximum number of amino acids
        num_inp_tokens=21,
        depth=9,
        d=64,
        h=4,
        ffn_multiplier=1,
        ce_units=256,
        use_charge=True,
        use_energy=False,
        use_mass=True,
        norm_type="layer",
        prenorm=True,
        preembed=True,
        penultimate_units=None,
        pool=False,
    ):
        super(Decoder, self).__init__()
        self.run_units = running_units
        self.kv_indim = kv_indim
        self.sl = sequence_length
        self.num_inp_tokens = num_inp_tokens
        self.num_out_tokens = (
            num_inp_tokens - 1
        )  # no need for start or hidden tokens, add EOS
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass

        self.ce_units = ce_units

        # Normalization type
        self.norm = mp.get_norm_type(norm_type)

        # First embeddings
        self.seq_emb = nn.Embedding(num_inp_tokens, running_units)
        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

        # charge/energy embedding transformation
        self.atleast1 = use_charge or use_energy or use_mass
        if self.atleast1:
            num = sum([use_charge, use_energy, use_mass])
            self.ce_emb = nn.Sequential(nn.Linear(ce_units * num, ce_units), nn.SiLU())

        # Main blocks
        attention_dict = {"indim": running_units, "d": d, "h": h}
        ffn_dict = {"indim": running_units, "unit_multiplier": ffn_multiplier}
        is_embed = True if self.atleast1 else False
        self.main = nn.ModuleList(
            [
                mp.TransBlock(
                    attention_dict,
                    ffn_dict,
                    norm_type,
                    prenorm,
                    is_embed,
                    ce_units,
                    preembed,
                    is_cross=True,
                    kvindim=kv_indim,
                )
                for _ in range(depth)
            ]
        )

        # Final block
        units = running_units if penultimate_units == None else penultimate_units
        self.final = nn.Sequential(
            nn.Linear(running_units, units, bias=False),
            nn.GELU(),
            self.norm(units),
            nn.Linear(units, self.num_out_tokens),
        )

        # Pool sequence dimension?
        self.pool = pool

        # Positional embedding
        pos = mp.FourierFeatures(
            torch.arange(self.sl, dtype=torch.float32), self.run_units, 5.0 * self.sl
        )
        self.pos = nn.Parameter(pos, requires_grad=False)

    def total_params(self):
        return sum([m.numel() for m in self.parameters()])

    def sequence_mask(self, seqlen, max_len=None):
        # seqlen: 1d vector equal to (zero-based) index of predict token
        sequence_len = self.sl if max_len is None else max_len
        if seqlen == None:
            mask = torch.zeros(1, sequence_len, dtype=torch.float32)
        else:
            seqs = torch.tile(
                torch.arange(sequence_len, device=self.device)[None],
                (seqlen.shape[0], 1),
            )
            # Only mask out sequence positions greater than or equal to predict
            # token
            # - if predict token is at position 5 (zero-based), mask out
            #   positions 5 to seq_len, i.e. you can only attend to positions
            #   0, 1, 2, 3, 4
            mask = 1e5 * (seqs >= seqlen[:, None]).type(torch.float32)

        return mask

    def Main(self, inp, kv_feats, embed=None, spec_mask=None, seq_mask=None):
        out = inp
        for layer in self.main:
            out = layer(
                out,
                kv_feats=kv_feats,
                embed_feats=embed,
                spec_mask=spec_mask,
                seq_mask=seq_mask,
            )

        return out

    def Final(self, inp):
        out = inp
        return out

    def EmbedInputs(self, intseq, charge=None, energy=None, mass=None):
        # Sequence embedding
        seqemb = self.seq_emb(intseq)

        # charge and/or energy embedding
        if self.atleast1:
            ce_emb = []
            if self.use_charge:
                charge = charge.type(torch.float32)
                ce_emb.append(mp.FourierFeatures(charge, self.ce_units, 10.0))
            if self.use_energy:
                ce_emb.append(mp.FourierFeatures(energy, self.ce_units, 150.0))
            if self.use_mass:
                ce_emb.append(mp.FourierFeatures(mass, self.ce_units, 20000.0))
            if len(ce_emb) > 1:
                ce_emb = torch.cat(ce_emb, dim=-1)
            ce_emb = self.ce_emb(ce_emb)
        else:
            ce_emb = None

        out = seqemb + self.alpha * self.pos[: seqemb.shape[1]].unsqueeze(0)

        return out, ce_emb

    def forward(
        self,
        intseq,
        kv_feats,
        charge=None,
        energy=None,
        mass=None,
        seqlen=None,
        specmask=None,
    ):
        out, ce_emb = self.EmbedInputs(intseq, charge=charge, energy=energy, mass=mass)

        seqmask = self.sequence_mask(seqlen, out.shape[1])  # max(seqlen))

        out = self.Main(
            out, kv_feats=kv_feats, embed=ce_emb, spec_mask=specmask, seq_mask=seqmask
        )

        out = self.final(out)
        if self.pool:
            out = out.mean(dim=1)

        return out


class DenovoDecoder(pl.LightningModule):
    def __init__(self, token_dicts, dec_config):
        super(DenovoDecoder, self).__init__()

        self.input_dict = token_dicts["input_dict"]
        self.SOS = self.input_dict["<SOS>"]
        self.output_dict = token_dicts["output_dict"]
        self.EOS = self.output_dict["<EOS>"]
        self.hidden_token = self.input_dict["<H>"]

        dec_config["num_inp_tokens"] = len(self.input_dict)

        self.predcats = len(self.output_dict)
        self.scale = Scale(self.output_dict)

        self.dec_config = dec_config
        self.decoder = Decoder(**dec_config)

        self.use_mass = dec_config["use_mass"]
        self.use_charge = dec_config["use_charge"]

        self.initialize_variables()

    def initial_intseq(self, batch_size, seqlen=None):
        seq_length = self.seq_len if seqlen == None else seqlen
        intseq = torch.empty(batch_size, seq_length - 1, dtype=torch.int32)
        intseq = torch.fill(intseq, self.hidden_token)
        sos_tokens = torch.tensor([[self.SOS]], device=intseq.device).repeat(
            (batch_size, 1)
        )
        out = torch.cat([sos_tokens, intseq], dim=1)
        return out

    def num_reg_tokens(self, int_array):
        return (int_array != self.hidden_token).sum(1).type(torch.int32)

    def initialize_variables(self):
        self.seq_len = self.decoder.sl

    def column_inds(self, batch_size, column_ind):
        ind0 = torch.arange(batch_size)[:, None]
        ind1 = torch.fill(torch.fill(batch_size, 1, dtype=torch.int32), column_ind)
        inds = torch.cat([ind0, ind1], dim=1)

        return inds

    def set_tokens(self, int_array, inds, updates, add=False):
        shp = int_array.shape

        if type(inds) == int:
            int_array[:, inds] = updates + int_array[:, inds] if add else updates
        else:
            int_array[inds] = updates + int_array[inds] if add else updates

        return int_array

    def decinp(
        self,
        intseq,
        enc_out,
        charge=None,
        energy=None,
        mass=None,
    ):
        dec_inp = {
            "intseq": intseq.to(self.device),
            "kv_feats": enc_out["emb"].to(self.device),
            "charge": charge.to(self.device) if self.decoder.use_charge else None,
            "energy": energy.to(self.device) if self.decoder.use_energy else None,
            "mass": mass.to(self.device) if self.decoder.use_mass else None,
            "seqlen": self.num_reg_tokens(intseq.to(self.device)),  # for the seq. mask
            "specmask": enc_out["mask"].to(self.device)
            if enc_out["mask"] is not None
            else enc_out["mask"],
        }

        return dec_inp

    def greedy(self, predict_logits):
        return predict_logits.argmax(-1).type(torch.int32)

    # The encoder's output should have always come from a batch loaded in
    # from the dataset. The batch dictionary has any necessary inputs for
    # the decoder.

    def predict_sequence(self, enc_out, mass=None, charge=None, causal=False):
        dev = enc_out["emb"].device
        bs = enc_out["emb"].shape[0]
        # starting intseq array
        intseq = self.initial_intseq(bs, self.seq_len).to(dev)
        probs = torch.zeros(bs, self.seq_len, self.predcats).to(dev)
        for i in range(self.seq_len):
            index = int(i)

            dec_out = self.forward(
                intseq, enc_out, mass=mass, charge=charge, causal=causal
            )
            logits = dec_out["logits"]

            predictions = self.greedy(logits[:, index])
            probs[:, index, :] = logits[:, index]

            if index < self.seq_len - 1:
                intseq = self.set_tokens(intseq, index + 1, predictions)

        intseq = torch.cat([intseq[:, 1:], predictions[:, None]], dim=1)

        return intseq, probs

    def correct_sequence_(self, enc_out, batdic):
        bs = enc_out["emb"].shape[0]
        rank = torch.zeros(bs, self.seq_len, dtype=torch.int32)
        prob = torch.zeros(bs, self.seq_len, dtype=torch.float32)
        # starting intseq array
        intseq = self.initial_intseq(bs, self.seq_len)
        for i in range(self.seq_len):
            index = int(i)

            dec_out = self(intseq, enc_out, batdic, False)

            wrank = torch.where(
                (-dec_out[:, i]).argsort(-1) == batdic["seqint"][:, i : i + 1]
            )[-1].type(torch.int32)

            rank = self.set_tokens(rank, index, wrank)

            inds = (torch.arange(bs, dtype=torch.int32), batdic["seqint"][:, i])
            # updates = tf.matorch.log(tf.gather_nd(dec_out[:, i], inds))
            updates = dec_out[:, i][inds].log()
            prob = self.set_tokens(prob, index, updates)

            predictions = batdic["seqint"][:, i]  # self.greedy(dec_out[:, index])

            if index < self.seq_len - 1:
                intseq = self.set_tokens(intseq, index + 1, predictions)

        intseq = torch.cat([intseq[:, 1:], predictions[:, None]], dim=1)

        return rank, prob  # UNNECESSARY

    def forward(
        self,
        input_intseq,
        enc_out,
        mass=None,
        charge=None,
        energy=None,
        causal=False,
        peptide_lengths=None,
    ):
        if causal:
            raise NotImplementedError()

        if peptide_lengths is not None:
            raise NotImplementedError(
                "This class needs to implement the padding mask creation based on "
                "pep length and include a 'padding_mask' in the returned dict"
            )
            # padding_mask = self._get_padding_mask()
        else:
            padding_mask = None

        dec_inp = self.decinp(
            input_intseq,
            enc_out,
            charge=charge,
            mass=mass,
            energy=energy,
        )

        logits = self.decoder.forward(**dec_inp)

        return {"logits": logits, "padding_mask": padding_mask}


def decoder_greedy_base(token_dict, d_model=512, **kwargs):
    decoder_config = {
        "kv_indim": d_model,
        "running_units": 512,
        "sequence_length": 30,
        "depth": 9,
        "d": 64,
        "h": 4,
        "ffn_multiplier": 1,
        "ce_units": 256,
        "use_charge": True,
        "use_energy": False,
        "use_mass": True,
        "norm_type": "layer",
        "prenorm": True,
        "preembed": True,
        # penultimate_units: #?
        "pool": False,
    }
    model = DenovoDecoder(token_dict, decoder_config, **kwargs)
    return model


def decoder_greedy_small(token_dict, d_model=256, **kwargs):
    decoder_config = {
        "kv_indim": d_model,
        "running_units": 128,
        "sequence_length": 30,
        "depth": 6,
        "d": 64,
        "h": 4,
        "ffn_multiplier": 1,
        "ce_units": 128,
        "use_charge": True,
        "use_energy": False,
        "use_mass": True,
        "norm_type": "layer",
        "prenorm": True,
        "preembed": True,
        # penultimate_units: #?
        "pool": False,
    }
    model = DenovoDecoder(token_dict, decoder_config, **kwargs)
    return model


def decoder_greedy_tiny(token_dict, d_model=256, **kwargs):
    decoder_config = {
        "kv_indim": d_model,
        "running_units": 32,
        "sequence_length": 30,
        "depth": 2,
        "d": 64,
        "h": 4,
        "ffn_multiplier": 1,
        "ce_units": 32,
        "use_charge": True,
        "use_energy": False,
        "use_mass": True,
        "norm_type": "layer",
        "prenorm": True,
        "preembed": True,
        # penultimate_units: #?
        "pool": False,
    }
    model = DenovoDecoder(token_dict, decoder_config, **kwargs)
    return model
