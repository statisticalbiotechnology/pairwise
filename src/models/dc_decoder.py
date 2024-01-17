import warnings
import depthcharge
from depthcharge.encoders import PositionalEncoder
from depthcharge.tokenizers import PeptideTokenizer
import torch
import torch.nn as nn

from depthcharge.transformers.peptides import generate_tgt_mask


class PeptideTransformerDecoder(depthcharge.transformers.PeptideTransformerDecoder):
    def __init__(
        self,
        amod_dict: dict,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        max_charge: int = 5,
        use_mass: bool = True,
        use_charge: bool = True,
        max_seq_len: int = 30,
    ) -> None:
        self.amod_dict = amod_dict
        n_tokens = len(amod_dict)
        super().__init__(
            n_tokens,
            d_model,
            nhead,
            dim_feedforward,
            n_layers,
            dropout,
            positional_encoder,
            max_charge,
        )
        self.n_tokens = n_tokens
        self.use_mass = use_mass
        self.use_charge = use_charge
        self.max_seq_len = max_seq_len
        self.start_token = len(amod_dict)
        self.null_token = self.amod_dict["X"]

    def forward(
        self,
        intseq: torch.Tensor | None,
        encoder_out: dict[torch.Tensor],
        precusor_dict: dict | None,
        include_null_in_tgt: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        intseq : list of str, torch.Tensor, or None
            The partial integer peptide sequences for which to predict the next
            amino acid.
        encoder_out: dict[torch.Tensor]
            "emb" : torch.Tensor of shape (batch_size, n_peaks, d_model)
                The representations from a ``TransformerEncoder``, such as a
                ``SpectrumEncoder``.
            "mask" : torch.Tensor of shape (batch_size, n_peaks)
                The mask that indicates which elements of ``memory`` are padding.
        precursor_dict : dict of torch.Tensors of size (batch_size, 1)
            with keys "mass", "charge".
        Returns
        -------
        output_dict: dict
            "logits" : torch.Tensor of size (batch_size, len_sequence, n_amino_acids)
                The raw output for the final linear layer. These can be Softmax
                transformed to yield the probability of each amino acid for the
                prediction.
            "target_inds" : list[torch.Tensor, torch.Tensor]
                list of generated target indices of type torch.Tensor of size (batch_size,)
                for each entry in the batch, generates a random index for the training task.
            "intseq": returns the input intseq for compatibility

        """
        memory = encoder_out["emb"]
        memory_key_padding_mask = encoder_out["mask"]
        batch_size = memory.shape[0]

        start_tokens = torch.tensor(
            [[self.start_token]], dtype=torch.long, device=self.device
        ).repeat((batch_size, 1))
        # Prepare sequences
        if intseq is None:
            intseq_ = start_tokens
        else:
            intseq_ = torch.cat([start_tokens, intseq], dim=1)

        # Encode tokens:
        tokens = self.aa_encoder(intseq_)

        # Additional tokens for charge/energy/mass/start
        num_start_tokens = 1

        # Encode and concat precursor info
        if precusor_dict:
            if "mass" in precusor_dict:
                masses = self.mass_encoder(precusor_dict["mass"][:, None])
                if not self.use_mass:
                    warnings.warn(
                        f"This model is not trained to include precursor mass"
                    )
            else:
                masses = 0
            if "charge" in precusor_dict:
                charges = self.charge_encoder(precusor_dict["charge"].int() - 1)
                charges = charges[:, None, :]
                if not self.use_charge:
                    warnings.warn(
                        f"This model is not trained to include precursor charge"
                    )
            else:
                charges = 0
            precursors = masses + charges

            tgt = torch.cat([precursors, tokens], dim=1)
            num_start_tokens += 1
        else:
            tgt = tokens

        # Feed through model:
        # tgt_key_padding_mask = tgt.sum(axis=2) == 0

        pad_mask = self.get_padding_mask(intseq, num_start_tokens, include_null=True)
        # pad_mask = torch.tensor([[False]], device=intseq.device).repeat(
        #     (batch_size, tgt.shape[1])
        # )
        tgt = self.positional_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1]).to(self.device)
        dec_embeds = self.transformer_decoder.forward(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            # tgt_is_causal=True,
            tgt_key_padding_mask=pad_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.final(dec_embeds)
        logits = logits[:, num_start_tokens:, :]  # remove c/e/m/start tokens

        target_inds = self.get_target_inds(intseq, include_null=True)
        return {"logits": logits, "target_inds": target_inds, "intseq": intseq}

    def get_padding_mask(self, intseq, num_start_tokens, include_null=True):
        batch_size = intseq.shape[0]
        nonnull = (
            (intseq != self.amod_dict["X"]).type(torch.int32).sum(dim=1, keepdim=True)
        )

        if include_null:
            # Add 1 to the entries that include the null_token,
            # so that the null_token has a chance to be included as well
            lens = torch.ones((intseq.shape[0], 1)).type_as(intseq) * intseq.shape[1]
            with_null = nonnull != lens
            nonnull[with_null] += 1

        tiles = (
            torch.arange(intseq.shape[1])
            .unsqueeze(0)
            .tile(intseq.shape[0], 1)
            .type_as(intseq)
        )
        pad_mask = tiles >= nonnull
        extra_pads = torch.tensor([[False]], device=pad_mask.device).repeat(
            (batch_size, num_start_tokens)
        )
        pad_mask = torch.cat([extra_pads, pad_mask], dim=1)
        return pad_mask

    def get_target_inds(self, intseq: torch.Tensor, include_null=True):
        nonnull = (
            (intseq != self.amod_dict["X"]).type(torch.int32).sum(dim=1, keepdim=True)
        )

        if include_null:
            # Add 1 to the entries that include the null_token,
            # so that the null_token has a chance to be included as well
            lens = torch.ones((intseq.shape[0], 1)).type_as(intseq) * intseq.shape[1]
            with_null = nonnull != lens
            nonnull[with_null] += 1

        uniform = torch.rand((intseq.shape[0], 1), device=nonnull.device) * nonnull
        inds = uniform.floor().type(torch.int32).squeeze(1)

        inds_ = [torch.arange(inds.shape[0], dtype=torch.int32), inds]
        return inds_

    def predict_sequence(
        self, enc_out: dict[torch.Tensor], precursor_dict: dict[torch.Tensor]
    ):
        batch_size = enc_out["emb"].shape[0]

        # Initialize output tensors
        intseq = torch.tensor(
            [[self.null_token]], dtype=torch.long, device=self.device
        ).repeat((batch_size, 1))
        logits = torch.zeros(batch_size, self.max_seq_len, self.n_tokens).type_as(
            enc_out["emb"]
        )

        # Gather predictions (fixed length loop)
        for i in range(0, self.max_seq_len):
            dec_out = self.forward(intseq, enc_out, precursor_dict)
            cur_logits = dec_out["logits"]

            predictions = self.greedy(cur_logits[:, i])
            logits[:, i, :] = cur_logits[:, i]

            intseq = torch.cat([intseq, predictions], dim=1)

        return intseq[:, 1:], logits

    def greedy(self, predict_logits: torch.Tensor):
        return predict_logits.argmax(dim=-1, keepdim=True).type(torch.int32)


def dc_decoder_tiny(amod_dict, d_model=256, **kwargs):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=2,
        dim_feedforward=128,
        n_layers=1,
        dropout=0,
        positional_encoder=True,
        max_charge=9,
        use_mass=True,
        use_charge=True,
    )
    return model


def dc_decoder_base(amod_dict, d_model=256, **kwargs):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=8,
        dim_feedforward=512,
        n_layers=9,
        dropout=0,
        positional_encoder=True,
        max_charge=9,
        use_mass=True,
        use_charge=True,
    )
    return model
