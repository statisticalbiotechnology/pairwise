import torch
from wrappers.base_wrapper import BaseDownstreamWrapper

from casanovo_eval import aa_match_batch, aa_match_metrics
import torch.nn.functional as F


def fill_null_after_first_EOS(prediction, null_token, EOS_token):
    pred_without_eos = prediction.clone()
    eos_mask = prediction == EOS_token
    absent_eos = eos_mask.sum(1) == 0
    # Find the position of the first predicted EOS token
    eos_positions = torch.argmax(eos_mask.int(), dim=1)
    eos_positions[absent_eos] = prediction.shape[-1]

    inds = (
        torch.arange(prediction.shape[1], device=prediction.device)
        .unsqueeze(0)
        .repeat((prediction.shape[0], 1))
    )
    forward_fill_mask = inds > torch.ones_like(inds) * eos_positions.unsqueeze(1)
    pred_without_eos[forward_fill_mask] = null_token
    return pred_without_eos


class DeNovoTeacherForcing(BaseDownstreamWrapper):
    def __init__(
        self,
        encoder,
        decoder,
        global_args,
        collate_fn=None,
        token_dicts=None,
        task_dict=None,
    ):
        super().__init__(
            encoder, decoder, global_args, collate_fn=collate_fn, task_dict=task_dict
        )

        self.amod_dict = token_dicts["amod_dict"]
        self.int_to_aa = {v: k for k, v in self.amod_dict.items()}
        self.null_token = "X"
        self.conf_threshold = task_dict["conf_threshold"]

        self.residues = token_dicts["residues"]
        self.input_dict = token_dicts["input_dict"]
        self.SOS = self.input_dict["<SOS>"]
        self.output_dict = token_dicts["output_dict"]
        self.EOS = self.output_dict["<EOS>"]
        self.NT = self.amod_dict[self.null_token]

        self.decoder_use_cls = task_dict["decoder_use_cls"]

        self.predcats = len(self.output_dict)

        self.TASK_NAME = "denovo_tf"

    def _mzab_array(self, batch):
        mz_arr = batch["mz_array"]
        int_arr = batch["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        return mzab

    def _prepend_SOS_tokens(self, intseq: torch.Tensor):
        batch_size = intseq.shape[0]
        sos_tokens = torch.tensor([[self.SOS]], device=intseq.device).repeat(
            (batch_size, 1)
        )
        input = torch.cat([sos_tokens, intseq], dim=1)
        return input

    def _append_EOS_tokens(self, intseq: torch.Tensor, pep_lengths: torch.Tensor):
        batch_size = intseq.shape[0]

        # append one step of null_tokens
        null_tokens = torch.tensor(
            [[self.amod_dict["X"]]], device=intseq.device
        ).repeat((batch_size, 1))
        intseq_ = torch.cat([intseq, null_tokens], dim=1)

        # impute self.EOS at positions specificed by pep_lengths
        impute_inds = [torch.arange(batch_size), pep_lengths.squeeze(1)]

        eos_tokens = torch.tensor([self.EOS], device=intseq.device).repeat((batch_size))
        intseq_[impute_inds] = eos_tokens
        return intseq_

    def _input_target(self, batch):
        intseq = batch["intseq"]  # shape = (batch_size, sequence_len)
        lengths = batch["peptide_lengths"]
        input = self._prepend_SOS_tokens(intseq)
        target = self._append_EOS_tokens(intseq, lengths)
        return input, target

    def _parse_batch(self, batch, Eval=False):
        input, target = self._input_target(batch)
        batch_size = input.shape[0]
        # Encoder input - mz/ab
        mzab = self._mzab_array(batch)

        parsed_batch = {
            "mz_ab": mzab,
            "mass": batch["precursor_mass"],
            "charge": batch["precursor_charge"],
            "input_intseq": input,
            "target_intseq": target,
            "peak_lengths": batch["peak_lengths"],
            "peptide_lengths": batch["peptide_lengths"],
        }
        return parsed_batch, batch_size

    def forward(self, parsed_batch, **kwargs):
        key_padding_mask = self._get_padding_mask(
            parsed_batch["mz_ab"], parsed_batch["peak_lengths"]
        )
        outs = self.encoder(
            parsed_batch["mz_ab"],
            length=parsed_batch["peak_lengths"],
            mass=parsed_batch["mass"] if self.encoder.use_mass else None,
            charge=parsed_batch["charge"] if self.encoder.use_charge else None,
            return_mask=True,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        if self.decoder_use_cls:
            cls_token = outs["emb"][:, : self.encoder.cls_token.shape[1]]

        preds = self.decoder(
            parsed_batch["input_intseq"],
            outs,
            cls_token=cls_token if self.decoder_use_cls else None,
            mass=parsed_batch["mass"] if self.decoder.use_mass else None,
            charge=parsed_batch["charge"] if self.decoder.use_charge else None,
            peptide_lengths=parsed_batch["peptide_lengths"],
        )
        return preds

    def eval_forward(self, parsed_batch, **kwargs):
        key_padding_mask = self._get_padding_mask(
            parsed_batch["mz_ab"], parsed_batch["peak_lengths"]
        )
        outs = self.encoder(
            parsed_batch["mz_ab"],
            length=parsed_batch["peak_lengths"],
            mass=parsed_batch["mass"] if self.encoder.use_mass else None,
            charge=parsed_batch["charge"] if self.encoder.use_charge else None,
            return_mask=True,
            key_padding_mask=key_padding_mask,
            **kwargs,
        )
        if self.decoder_use_cls:
            cls_token = outs["emb"][:, : self.encoder.cls_token.shape[1]]
        preds = self.decoder.predict_sequence(
            outs,
            cls_token=cls_token if self.decoder_use_cls else None,
            mass=parsed_batch["mass"] if self.decoder.use_mass else None,
            charge=parsed_batch["charge"] if self.decoder.use_charge else None,
            causal=True,
        )
        return preds

    def _get_train_loss(
        self, logits: torch.Tensor, labels: torch.Tensor, padding_mask: torch.Tensor
    ):
        # logits.shape = (batch_size, sequence_len, num_classes)
        logits = logits.transpose(-2, -1)
        # logits.shape = (batch_size, num_classes, sequence_len)
        loss = F.cross_entropy(
            logits,
            labels,
            reduction="mean",
            label_smoothing=self.label_smoothing,
            ignore_index=self.NT,
        )
        return loss

    def _get_train_stats(self, returns, parsed_batch):
        target = parsed_batch["target_intseq"]
        logits = returns["logits"]
        padding_mask = (~returns["padding_mask"]).int()
        loss = self._get_train_loss(logits, target, padding_mask)
        stats = {"loss": loss}
        return loss, stats

    def _get_eval_stats(self, returns, parsed_batch):
        targ = parsed_batch["target_intseq"]
        preds, logits = returns

        preds = preds[:, : targ.shape[1]]
        logits = logits[:, : targ.shape[1]]

        # Band-aid validation loss (preds could be shorter than target)
        min_seq_len = min(logits.shape[1], targ.shape[1])
        logits_ce = logits[:, :min_seq_len].transpose(
            -1, -2
        )  # (batch_size, num_classes, min_seq_len)
        targ_trimmed = targ[:, :min_seq_len]  # (batch_size, min_seq_len)
        loss = F.cross_entropy(
            logits_ce, targ_trimmed, reduction="none", ignore_index=self.NT
        ).mean()

        preds_ffill = fill_null_after_first_EOS(
            preds, null_token=self.NT, EOS_token=self.EOS
        )

        deepnovo_metrics = self.deepnovo_metrics(preds_ffill, targ)
        stats = {"loss": loss, **deepnovo_metrics}
        return stats

    def _replace_eos_with_null(self, tensor: torch.Tensor):
        tensor = tensor.clone()
        tensor[tensor == self.EOS] = self.amod_dict[self.null_token]
        return tensor

    def to_aa_sequence(self, int_tensors: torch.Tensor | list):
        # Check if the input is a Tensor and convert it to a list
        if isinstance(int_tensors, torch.Tensor):
            int_tensors = self._replace_eos_with_null(int_tensors)
            int_tensors = int_tensors.tolist()

        def convert_sequence(seq):
            # Convert to amino acids and strip null tokens
            return [
                self.int_to_aa[i] for i in seq if self.int_to_aa[i] != self.null_token
            ]

        # Check if the input is a list of lists or a single list
        if int_tensors and isinstance(int_tensors[0], list):
            # Convert each sequence in the list of lists into strings
            aa_sequences = [convert_sequence(seq) for seq in int_tensors]
        else:
            # Convert the single sequence into a string
            aa_sequences = convert_sequence(int_tensors)

        return aa_sequences

    def deepnovo_metrics(self, preds, target):
        target_str = self.to_aa_sequence(target)
        pred_str = self.to_aa_sequence(preds)

        aa_matches_batch, n_aa_true, n_aa_pred = aa_match_batch(
            target_str, pred_str, aa_dict=self.residues, mode="best"
        )
        aa_prec, aa_recall, pep_prec = aa_match_metrics(
            aa_matches_batch, n_aa_true, n_aa_pred
        )

        return {
            "aa_prec": aa_prec,
            "aa_recall": aa_recall,
            # "pep_recall": pep_recall, #TODO: add a pep recall? is it superflous since pep_precision at cov=1 is simply the acc?
            "pep_prec": pep_prec,
            # "pep_auc": pep_auc, #TODO: add pr-curves after training is complete
        }

    def on_train_epoch_start(self):
        if hasattr(self.trainer.datamodule, "datasets"):
            if hasattr(self.trainer.datamodule.datasets[0], "set_epoch"):
                self.trainer.datamodule.datasets[0].set_epoch(
                    self.trainer.current_epoch
                )
