from copy import deepcopy
import numpy as np
import models.model_parts as mp
import torch
from torch import nn
import pytorch_lightning as pl
from utils import Scale
# beam search dependencies
import collections
import einops
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import heapq



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
        dropout=0,
        pool=False,
    ):
        super(Decoder, self).__init__()
        self.run_units = running_units
        self.kv_indim = kv_indim
        self.sl = sequence_length
        self.num_inp_tokens = num_inp_tokens
        self.num_out_tokens = (
            num_inp_tokens - 0 # 1 if denovo_random
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
            self.ce_emb = nn.Sequential(
                nn.Linear(ce_units*num, ce_units), nn.SiLU()
            )

        # Main blocks
        attention_dict = {'indim': running_units, 'd': d, 'h': h, 'dropout': dropout}
        ffn_dict = {'indim': running_units, 'unit_multiplier': ffn_multiplier, 'dropout': dropout}
        is_embed = True if self.atleast1 else False
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict,
                ffn_dict,
                norm_type,
                prenorm,
                is_embed,
                ce_units,
                preembed,
                is_cross=True,
                kvindim=kv_indim
            )
            for _ in range(depth)
        ])

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
            torch.arange(self.sl, dtype=torch.float32), 1, 150, self.run_units
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
            mask = 1e7 * (seqs >= seqlen[:, None]).type(torch.float32)

        return mask

    def causal_mask(self, x):
        bs, sl, c = x.shape
        ones = torch.ones(bs, sl, sl, device=x.device)
        mask =  1e7*torch.triu(ones, diagonal=1)

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
            out = out['out']

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
                ce_emb.append(mp.FourierFeatures(charge, 1, 10, self.ce_units))
            if self.use_energy:
                ce_emb.append(mp.FourierFeatures(energy, self.ce_units, 150.0))
            if self.use_mass:
                ce_emb.append(mp.FourierFeatures(mass, 0.001, 10000, self.ce_units))
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

        #seqmask = self.sequence_mask(seqlen, out.shape[1])  # max(seqlen))
        seqmask = self.causal_mask(out)

        out = self.Main(
            out, kv_feats=kv_feats, embed=ce_emb, spec_mask=specmask, seq_mask=seqmask
        )

        out = self.final(out)
        if self.pool:
            out = out.mean(dim=1)

        return out

def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6


class DenovoDecoder(pl.LightningModule):
    def __init__(self, token_dicts, dec_config):
        super(DenovoDecoder, self).__init__()

        self.input_dict = token_dicts["input_dict"]
        self.SOS = self.input_dict["<SOS>"]
        self.output_dict = token_dicts["output_dict"]
        self.EOS = self.output_dict["<EOS>"]
        self.hidden_token = self.input_dict["X"]

        dec_config["num_inp_tokens"] = len(self.input_dict)

        self.predcats = len(self.output_dict)
        self.scale = Scale(self.output_dict)

        self.dec_config = dec_config
        self.decoder = Decoder(**dec_config)

        self.use_mass = dec_config["use_mass"]
        self.use_charge = dec_config["use_charge"]

        self.causal = True
        
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
                intseq, enc_out, mass=mass, charge=charge, causal=self.causal # change for denovo random
            )
            logits = dec_out["logits"]

            predictions = self.greedy(logits[:, index])
            probs[:, index, :] = logits[:, index]

            if index < self.seq_len - 1:
                intseq = self.set_tokens(intseq, index + 1, predictions)

        intseq = torch.cat([intseq[:, 1:], predictions[:, None]], dim=1)

        return intseq, probs

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

        if peptide_lengths is not None:
            sequence_length = input_intseq.shape[1]
            padding_mask = self.decoder.sequence_mask( 
                peptide_lengths.squeeze(),
                sequence_length,
            )
            padding_mask = padding_mask != 0
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

    def beam_search_decode(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Beam search decoding of the spectrum predictions.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        enc_out = self.encoder(spectra, return_mask=True)

        # Sizes.
        batch = spectra.shape[0]  # B
        length =  self.seq_len # + 1  # L
        vocab = self.predcats #+ 1 #self.decoder.vocab_size + 1  # V
        beam = self.n_beams  # S

        # Initialize scores and tokens.
        scores = torch.full(
            size=(batch, length, vocab, beam), fill_value=torch.nan
        )
        scores = scores.type_as(spectra)
        tokens = self.NT*torch.ones(batch, length, beam, dtype=torch.int64)
        tokens = tokens.to(self.encoder.device)

        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        intseq = self.initial_intseq(batch, self.seq_len).to(
            enc_out['emb'].device
        )
        pred = self(intseq, enc_out, precursors) #mem_masks)
        tokens[:, 0, :] = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, 0, :, :] = pred[:,0,:,None].tile(1, 1, beam) #einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make all tensors the right shape for decoding.
        precursors['charge'] = precursors['charge'][:,None].tile(1, beam).reshape(-1,)
        precursors['mass'] = precursors['mass'][:,None].tile(1, beam).reshape(-1,)
        precursors['length'] = precursors['length'][:,None].tile(1, beam).reshape(-1,)
        precursors['mz'] = (precursors['mass'] - 18.010565) / precursors['charge']  - 1.00727646688
        enc_out['emb'] = enc_out['emb'][:,None].tile(1, beam, 1, 1).reshape(batch*beam, self.encoder.sl, self.encoder.run_units)
        enc_out['mask'] = enc_out['mask'][:,None].tile(1, beam, 1).reshape(batch*beam, self.encoder.sl)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        intseq = intseq[:,None].tile(1, beam, 1).reshape(batch*beam, length)

        # The main decoding loop.
        for step in range(0, self.seq_len-1):
            # Terminate beams exceeding the precursor m/z tolerance and track
            # all finished beams (either terminated or stop token predicted).
            (
                finished_beams,
                beam_fits_precursor,
                discarded_beams,
            ) = self._finish_beams(tokens, precursors, step)
            
            # Cache peptide predictions from the finished beams (but not the
            # discarded beams).
            self._cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams & ~discarded_beams,
                beam_fits_precursor,
                pred_cache,
            )

            # Stop decoding when all current beams have been finished.
            # Continue with beams that have not been finished and not discarded.
            finished_beams |= discarded_beams
            if finished_beams.all():
                break
            
            # Update the scores.
            intseq[~finished_beams, step+1] = tokens[~finished_beams, step].int()
            intseq_ = intseq[~finished_beams]
            precursors_ = self.subsample_precursors(precursors, ~finished_beams)
            enc_out_ = self.subsample_enc_out(enc_out, ~finished_beams)
            pred = self(intseq_, enc_out_, precursors_)
            scores[~finished_beams, step+1] = pred[:, step+1]
            
            # Find the top-k beams with the highest scores and continue decoding
            # those.
            tokens, scores = self._get_topk_beams(
                tokens, scores, finished_beams, batch, step + 1
            )

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        return self._get_top_peptide(pred_cache)

    def _finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished, either by predicting the stop
        token or because they were terminated due to exceeding the precursor
        m/z tolerance.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, self.max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have been
            finished.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within precursor m/z
            tolerance.
        discarded_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams should be
            discarded (e.g. because they were predicted to end but violate the
            minimum peptide length).
        """
        # Check for tokens with a negative mass (i.e. neutral loss).
        aa_neg_mass = [None]
        for aa, mass in self.scale.tok2mass.items():
            if mass < 0:
                aa_neg_mass.append(aa)
        
        # Find N-terminal residues.
        n_term = torch.Tensor(
            [
                self.outdict[aa]
                for aa in self.scale.tok2mass.keys()
                if aa.startswith(("+", "-"))
            ]
        ).to(self.device)

        beam_fits_precursor = torch.zeros(
            tokens.shape[0], dtype=torch.bool
        ).to(self.encoder.device)
        
        # Beams with a stop token predicted in the current step can be finished.
        finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        ends_stop_token = tokens[:, step] == self.outdict['X']
        finished_beams[ends_stop_token] = True
        
        # Beams with a dummy token predicted in the current step can be
        # discarded.
        discarded_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        #discarded_beams[tokens[:, step] == 0] = True # JL - I have no dummy token
        
        # Discard beams with invalid modification combinations (i.e. N-terminal
        # modifications occur multiple times or in internal positions).
        if step > 1:  # Only relevant for longer predictions.
            dim0 = torch.arange(tokens.shape[0])
            final_pos = torch.full((ends_stop_token.shape[0],), step)
            final_pos[ends_stop_token] = step - 1
            # Multiple N-terminal modifications.
            multiple_mods = torch.isin(
                tokens[dim0, final_pos], n_term
            ) & torch.isin(tokens[dim0, final_pos - 1], n_term)
            # N-terminal modifications occur at an internal position.
            # Broadcasting trick to create a two-dimensional mask.
            mask = (final_pos - 1)[:, None] >= torch.arange(tokens.shape[1])
            internal_mods = torch.isin(
                torch.where(mask.to(self.encoder.device), tokens, 0), n_term
            ).any(dim=1)
            discarded_beams[multiple_mods | internal_mods] = True

        # Check which beams should be terminated or discarded based on the
        # predicted peptide.
        for i in range(len(finished_beams)):
            
            # Skip already discarded beams.
            if discarded_beams[i]:
                continue
            pred_tokens = tokens[i][: step + 1]
            peptide_len = len(pred_tokens)
            peptide = pred_tokens #self.decoder.detokenize(pred_tokens)
            
            # Omit stop token.
            if self.reverse and peptide[0] == self.NT:
                peptide = peptide[1:]
                peptide_len -= 1
            elif not self.reverse and peptide[-1] == self.NT:
                peptide = peptide[:-1]
                peptide_len -= 1
            
            # Discard beams that were predicted to end but don't fit the minimum
            # peptide length.
            if finished_beams[i] and peptide_len < self.min_peptide_len:
                discarded_beams[i] = True
                continue
            
            # Terminate the beam if it has not been finished by the model but
            # the peptide mass exceeds the precursor m/z to an extent that it
            # cannot be corrected anymore by a subsequently predicted AA with
            # negative mass.
            precursor_charge = precursors['charge'][i]
            precursor_mz = precursors['mz'][i]
            matches_precursor_mz = exceeds_precursor_mz = False
            for aa in [None] if finished_beams[i] else aa_neg_mass:
                if aa is None:
                    calc_peptide = peptide
                else:
                    calc_peptide = peptide.copy()
                    calc_peptide.append(aa)
                try:
                    calc_mz = float(self.scale.intseq2mass(calc_peptide) / precursor_charge)
                    delta_mass_ppm = [
                        _calc_mass_error(
                            calc_mz,
                            precursor_mz,
                            precursor_charge,
                            isotope,
                        )
                        for isotope in range(
                            self.isotope_error_range[0],
                            self.isotope_error_range[1] + 1,
                        )
                    ]
                    # Terminate the beam if the calculated m/z for the predicted
                    # peptide (without potential additional AAs with negative
                    # mass) is within the precursor m/z tolerance.
                    matches_precursor_mz = aa is None and any(
                        abs(d) < self.precursor_mass_tol
                        for d in delta_mass_ppm
                    )
                    # Terminate the beam if the calculated m/z exceeds the
                    # precursor m/z + tolerance and hasn't been corrected by a
                    # subsequently predicted AA with negative mass.
                    if matches_precursor_mz:
                        exceeds_precursor_mz = False
                    else:
                        exceeds_precursor_mz = all(
                            d > self.precursor_mass_tol for d in delta_mass_ppm
                        )
                        exceeds_precursor_mz = (
                            finished_beams[i] or aa is not None
                        ) and exceeds_precursor_mz
                    if matches_precursor_mz or exceeds_precursor_mz:
                        break
                except KeyError:
                    matches_precursor_mz = exceeds_precursor_mz = False
            
            # Finish beams that fit or exceed the precursor m/z.
            # Don't finish beams that don't include a stop token if they don't
            # exceed the precursor m/z tolerance yet.
            if finished_beams[i]:
                beam_fits_precursor[i] = matches_precursor_mz
            elif exceeds_precursor_mz:
                finished_beams[i] = True
                beam_fits_precursor[i] = matches_precursor_mz
        
        return finished_beams, beam_fits_precursor, discarded_beams

    def _cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        beam_fits_precursor: torch.Tensor,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]],
    ):
        """
        Cache terminated beams.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the (negated)
            peptide score, amino acid-level scores, and the predicted tokens is
            stored.
        """
        for i in range(len(beams_to_cache)):
            if not beams_to_cache[i]:
                continue
            # Find the starting index of the spectrum.
            spec_idx = i // self.n_beams
            # FIXME: The next 3 lines are very similar as what's done in
            #  _finish_beams. Avoid code duplication?
            # JL - keep max_length prediction vector -> easier to batch
            pred_tokens = tokens[i]# [: step + 1]
            
            # Omit the stop token from the peptide sequence (if predicted).
            has_stop_token = pred_tokens[step] == self.NT
            pred_peptide = pred_tokens#[:-1] if has_stop_token else pred_tokens
            
            # Don't cache this peptide if it was already predicted previously.
            if any(
                torch.equal(pred_cached[-1], pred_peptide)
                for pred_cached in pred_cache[spec_idx]
            ):
                # TODO: Add duplicate predictions with their highest score.
                continue
            smx = torch.softmax(scores[i : i + 1, : step+1, :], -1)
            aa_scores = smx[0, range(step+1), pred_tokens[:step+1]].tolist()
            aa_scores_ = torch.nan_to_num(scores[i])
            
            # Add an explicit score 0 for the missing stop token in case this
            # was not predicted (i.e. early stopping).
            #if not has_stop_token:
            #    aa_scores.append(0)
            aa_scores = np.asarray(aa_scores)
            
            # Calculate the updated amino acid-level and the peptide scores.
            aa_scores, peptide_score = self._aa_pep_score(
                aa_scores, beam_fits_precursor[i]
            )
            
            # Omit the stop token from the amino acid-level scores.
            aa_scores = aa_scores[:-1]
            
            # Add the prediction to the cache (minimum priority queue, maximum
            # the number of beams elements).
            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
            heapadd(
                pred_cache[spec_idx],
                (peptide_score, aa_scores_, torch.clone(pred_peptide)),
            )


    def _aa_pep_score(self,
        aa_scores: np.ndarray, fits_precursor_mz: bool
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate amino acid and peptide-level confidence score from the raw amino
        acid scores.

        The peptide score is the mean of the raw amino acid scores. The amino acid
        scores are the mean of the raw amino acid scores and the peptide score.

        Parameters
        ----------
        aa_scores : np.ndarray
            Amino acid level confidence scores.
        fits_precursor_mz : bool
            Flag indicating whether the prediction fits the precursor m/z filter.

        Returns
        -------
        aa_scores : np.ndarray
            The amino acid scores.
        peptide_score : float
            The peptide score.
        """
        peptide_score = np.mean(aa_scores)
        #aa_scores = (aa_scores + peptide_score) / 2 # JL - commented out, I don't understand it
        if not fits_precursor_mz:
            peptide_score -= 1
        return aa_scores, peptide_score

    def _get_topk_beams(
        self,
        tokens: torch.tensor,
        scores: torch.tensor,
        finished_beams: torch.tensor,
        batch: int,
        step: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Find the top-k beams with the highest scores and continue decoding
        those.

        Stop decoding for beams that have been finished.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        batch: int
            Number of spectra in the batch.
        step : int
            Index of the next decoding step.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        """
        beam = self.n_beams  # S
        vocab = self.predcats # vocab_size + 1  # V

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)

        # Get the previous tokens and scores.
        prev_tokens = einops.repeat(
            tokens[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = torch.gather(
            scores.softmax(2)[:, :step, :, :], dim=2, index=prev_tokens # added softmax, instead of logits
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        # Get the scores for all possible beams at this step.
        step_scores = torch.zeros(batch, step + 1, beam * vocab).type_as(
            scores
        )
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores.softmax(2)[:, step, :, :], "B V S -> B (V S)"
        )

        # Mask out terminated beams. Include precursor m/z tolerance induced
        # termination.
        # TODO: `clone()` is necessary to get the correct output with n_beams=1.
        #   An alternative implementation using base PyTorch instead of einops
        #   might be more efficient.
        finished_mask = einops.repeat(
            finished_beams, "(B S) -> B (V S)", S=beam, V=vocab
        ).clone()
        # Mask out the index '0', i.e. padding token, by default.
        # JL - I don't have a padding token
        #finished_mask[:, :beam] = True

        # Figure out the top K decodings.
        _, top_idx = torch.topk(
            step_scores.nanmean(dim=1) * (~finished_mask).float(), beam
        )
        v_idx, s_idx = np.unravel_index(top_idx.cpu(), (vocab, beam))
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(batch), "B -> (B S)", S=beam)

        # Record the top K decodings.
        # JL: These are the top K decodings amongst ALL beams*predcats predictions
        #     There can be multiple chosen for a single beam, not simply each
        #     beam's respecitve top score.
        tokens[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
            ) # JL: This puts the top beams' 1-step tokens in place
        tokens[:, step, :] = torch.tensor(v_idx) # JL: This puts the top beams' step tokens in place
        scores[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        
        return tokens, scores

    def _get_top_peptide(
        self,
        pred_cache: Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:
        """
        Return the peptide with the highest confidence score for each spectrum.

        Parameters
        ----------
        pred_cache : Dict[int, List[Tuple[float, np.ndarray, torch.Tensor]]]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the peptide
            score, amino acid-level scores, and the predicted tokens is stored.

        Returns
        -------
        pred_peptides : Iterable[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        output = []
        probs  = []
        for peptides in pred_cache.values():
            if len(peptides) > 0:
                
                for pep_score, aa_scores, pred_tokens in heapq.nlargest(
                    self.top_match, peptides
                ):
                    output.append(pred_tokens)
                    probs.append(aa_scores)
                
            else:
                output.append(
                    self.NT*torch.ones((self.seq_len,)).to(self.encoder.device)
                )
                probs.append(
                    torch.zeros((self.max_length, self.predcats)).to(self.encoder.device)
                )

        return torch.stack(output), torch.stack(probs)

    def subsample_precursors(self, dic, boolean):
        dic2 = dic.copy()
        dic2['charge'] = dic2['charge'][boolean]
        dic2['mass'] = dic2['mass'][boolean]
        dic2['mz'] = dic2['mz'][boolean]

        return dic2

    def subsample_enc_out(self, dic, boolean):
        dic2 = dic.copy()
        dic2['emb'] = dic2['emb'][boolean]
        dic2['mask'] = dic2['mask'][boolean]

        return dic2


def decoder_greedy_base(token_dict, d_model=512, **kwargs):
    decoder_config = {
        "kv_indim": d_model,
        "running_units": 512,
        "sequence_length": 30,
        "depth": 9,
        "d": 64,
        "h": 8,
        "ffn_multiplier": 1,
        "ce_units": 256,
        "use_charge": True,
        "use_energy": False,
        "use_mass": True,
        "norm_type": "layer",
        "prenorm": True,
        "preembed": True,
        "dropout": 0,
        "pool": False,
    }
    model = DenovoDecoder(token_dict, decoder_config, **kwargs)
    return model


def decoder_greedy_small(token_dict, d_model=256, **kwargs):
    decoder_config = {
        "kv_indim": d_model,
        "running_units": 128,
        "sequence_length": 30,
        "depth": 9,
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
        "dropout": 0.1,
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
