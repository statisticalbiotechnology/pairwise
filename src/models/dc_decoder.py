import warnings
import depthcharge
from depthcharge.encoders import PositionalEncoder
from depthcharge.tokenizers import PeptideTokenizer
import torch
import torch.nn as nn

from depthcharge.transformers.peptides import generate_tgt_mask


def generate_causal_tgt_mask(sz: int) -> torch.Tensor:
    """Generate a square causal mask for the sequence.

    Parameters
    ----------
    sz : int
        The length of the target sequence.
    """
    return torch.triu(torch.ones((sz, sz), dtype=torch.bool), diagonal=1)


class PeptideTransformerDecoder(depthcharge.transformers.PeptideTransformerDecoder):
    def __init__(
        self,
        token_dicts: dict,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        positional_encoder: PositionalEncoder | bool = True,
        max_charge: int = 5,
        use_mass: bool = True,
        use_charge: bool = True,
        max_seq_len: int = 31,
        cross_attend: bool = True,
    ) -> None:
        self.amod_dict = token_dicts["amod_dict"]
        self.input_dict = token_dicts["input_dict"]
        self.SOS = self.input_dict["<SOS>"]
        self.output_dict = token_dicts["output_dict"]
        self.EOS = self.output_dict["<EOS>"]

        self.num_input_tokens = len(self.input_dict)
        self.num_output_tokens = len(self.output_dict)

        super().__init__(
            self.num_input_tokens
            - 1,  # depthcharge already adds one for the start token
            d_model,
            nhead,
            dim_feedforward,
            n_layers,
            dropout,
            positional_encoder,
            max_charge,
        )

        if not cross_attend:
            del self.transformer_decoder
            layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=dropout,
            )

            self.transformer_decoder = torch.nn.TransformerEncoder(
                layer, num_layers=n_layers
            )

        self.d_model = d_model
        self.use_mass = use_mass
        self.use_charge = use_charge
        self.max_seq_len = max_seq_len
        self.cross_attend = cross_attend

        # Override the final projection with
        # the correct num_classes
        self.final = torch.nn.Linear(
            d_model,
            self.num_output_tokens,
        )

    def forward(
        self,
        input_intseq: torch.Tensor,
        encoder_out: dict[torch.Tensor],
        cls_token: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
        peptide_lengths: torch.Tensor | None = None,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the next amino acid for a collection of sequences.

        Parameters
        ----------
        input_intseq : torch.Tensor
            The partial integer peptide sequences for which to predict the next
            amino acid.
        encoder_out: dict[torch.Tensor]
            "emb" : torch.Tensor of shape (batch_size, n_peaks, d_model)
                The representations from a ``TransformerEncoder``, such as a
                ``SpectrumEncoder``.
            "mask" : torch.Tensor of shape (batch_size, n_peaks)
                The mask that indicates which elements of ``memory`` are padding.
        cls_token: torch.Tensor of shape (batch_size, 1, d_model) or None
            The CLS token spectrum embeddings representations from a ``TransformerEncoder``, such as a
            ``SpectrumEncoder``.
        mass: torch.Tensor of shape (batch_size, 1) or None
            Precursor mass.
        charge: torch.Tensor of shape (batch_size, 1) or None
            Precursor charge.
        peptide_lengths: torch.Tensor of shape (batch_size, 1) or None
            Length of the ground-truth peptides. Used during training for key_padding_mask.
        causal: bool
            If True, apply a causal mask to the sequence (needed for teacher forcing).

        Returns
        -------
        dict
            "logits" : torch.Tensor of shape (batch_size, len_sequence, n_amino_acids)
                The raw output for the final linear layer. These can be Softmax
                transformed to yield the probability of each amino acid for the
                prediction.
            "padding_mask" : torch.Tensor or None
                The padding mask for the output logits.
        """
        memory = encoder_out["emb"]
        memory_key_padding_mask = encoder_out["mask"]
        batch_size = memory.shape[0]

        # Encode tokens:
        tokens = self.aa_encoder(input_intseq)

        if mass is not None and charge is not None:
            # Encode and concat precursor info
            if mass is not None:
                masses = self.mass_encoder(mass[:, None])
                if not self.use_mass:
                    warnings.warn("Not trained to include precursor mass")
            else:
                masses = 0
            if charge is not None:
                charges = self.charge_encoder(charge.int() - 1)
                charges = charges[:, None, :]
                if not self.use_charge:
                    warnings.warn("Not trained to include precursor charge")
            else:
                charges = 0
            precursors = masses + charges
        else:
            empty = [[[]] * self.d_model] * batch_size
            precursors = torch.tensor(empty, device=self.device).transpose(-2, -1)

        # Concat precursors
        num_precursor_tokens = precursors.shape[1]
        tgt = torch.cat([precursors, tokens], dim=1)

        tgt = self.positional_encoder(tgt)

        # Prepend CLS token to decoder input if provided
        if cls_token is not None:
            tgt = torch.cat([cls_token, tgt], dim=1)
            num_precursor_tokens += cls_token.shape[1]

        # Causal mask
        if causal:
            tgt_mask = generate_causal_tgt_mask(tgt.shape[1]).to(self.device)
        else:
            tgt_mask = None

        if self.cross_attend:
            # Forward
            dec_embeds = self.transformer_decoder.forward(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                # tgt_is_causal=True,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            # Forward without cross attention
            assert (
                cls_token is not None
            ), "Can only forward without cross-attention if cls_token is given"
            dec_embeds = self.transformer_decoder.forward(
                src=tgt,
                mask=tgt_mask,
                # is_causal=True,
                src_key_padding_mask=None,
            )
        logits = self.final(dec_embeds)

        # Remove precursor token(s)
        logits = logits[:, num_precursor_tokens:, :]

        # Create padding mask for the loss
        if peptide_lengths is None:
            padding_mask = None
        else:
            padding_mask = self.get_padding_mask(
                input_intseq,
                peptide_lengths,
                0,
            )
        return {"logits": logits, "padding_mask": padding_mask}

    def get_padding_mask(self, input_intseq, peptide_lengths, num_extra_tokens):
        batch_size = input_intseq.shape[0]

        total_len = peptide_lengths + num_extra_tokens
        inds = (
            torch.arange(
                input_intseq.shape[1] + num_extra_tokens, device=input_intseq.device
            )
            .unsqueeze(0)
            .repeat((batch_size, 1))
        )
        pad_mask = inds > total_len
        return pad_mask

    def predict_sequence(
        self,
        enc_out: dict[torch.Tensor],
        cls_token: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
        causal: bool = True,
    ):
        batch_size = enc_out["emb"].shape[0]

        # Initialize output tensors
        input_intseq = torch.tensor(
            [[self.SOS]], dtype=torch.long, device=self.device
        ).repeat((batch_size, 1))

        logits = torch.zeros(
            batch_size, self.max_seq_len, self.num_output_tokens
        ).type_as(enc_out["emb"])

        # Gather predictions (fixed length loop)
        for i in range(0, self.max_seq_len):
            dec_out = self.forward(
                input_intseq,
                enc_out,
                cls_token=cls_token,
                mass=mass,
                charge=charge,
                causal=causal,
            )
            cur_logits = dec_out["logits"]

            predictions = self.greedy(cur_logits[:, i])
            logits[:, i, :] = cur_logits[:, i]

            input_intseq = torch.cat([input_intseq, predictions], dim=1)

        return input_intseq[:, 1:], logits

    def greedy(self, predict_logits: torch.Tensor):
        return predict_logits.argmax(dim=-1, keepdim=True).type(torch.int32)


def dc_decoder_tiny(amod_dict, d_model=256, dropout=0, cross_attend=True, **kwargs):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=2,
        dim_feedforward=128,
        n_layers=1,
        positional_encoder=True,
        max_charge=9,
        use_mass=True,
        use_charge=True,
        dropout=dropout,
        cross_attend=cross_attend,
    )
    return model


def dc_decoder_base(
    amod_dict, 
    d_model=256, 
    dropout=0, 
    cross_attend=True, 
    max_seq_len=31,
    **kwargs
):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=8,
        dim_feedforward=512,
        n_layers=9,
        positional_encoder=True,
        max_charge=9,
        use_mass=True,
        use_charge=True,
        dropout=dropout,
        cross_attend=cross_attend,
        max_seq_len=max_seq_len,
    )
    return model


def dc_decoder_deeper(amod_dict, d_model=256, dropout=0.25, cross_attend=True, **kwargs):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=8,
        dim_feedforward=512,
        n_layers=15,
        dropout=dropout,
        positional_encoder=True,
        max_charge=9,
        use_mass=True,
        use_charge=True,
        cross_attend=cross_attend,
    )
    return model

def dc_decoder_jl(amod_dict, d_model=256, dropout=0.1, cross_attend=True, **kwargs):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=8,
        dim_feedforward=d_model,
        n_layers=9,
        dropout=dropout,
        positional_encoder=True,
        max_charge=9,
        use_mass=True,
        use_charge=True,
        cross_attend=cross_attend,
    )
    return model

def casanovo_decoder(amod_dict, d_model=256, dropout=0, cross_attend=True, **kwargs):
    model = PeptideTransformerDecoder(
        amod_dict,
        d_model,
        nhead=8,
        dim_feedforward=1024,
        n_layers=9,
        positional_encoder=True,
        max_charge=10,
        use_mass=True,
        use_charge=True,
        dropout=dropout,
        cross_attend=cross_attend,
    )
    return model
