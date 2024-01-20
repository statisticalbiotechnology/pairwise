import torch
from wrappers.base_wrapper import BasePLWrapper
from denovo_eval import Metrics as DeNovoMetrics
import torch.nn.functional as F


def NaiveAccRecPrec(target, prediction, null_token):
    correct_bool = (target == prediction).type(torch.int32)
    num_correct = correct_bool.sum()
    recall_bool = target != null_token
    recsum = correct_bool[recall_bool].sum()
    prec_bool = prediction != null_token
    precsum = correct_bool[prec_bool].sum()
    total = target.shape[0] * target.shape[1]
    return {
        "acc_naive": num_correct / total,
        "recall_naive": recsum / recall_bool.sum(),
        "precision_naive": precsum / prec_bool.sum(),
    }


def fill_null_after_first_EOS(prediction, null_token, EOS_token):
    pred_without_eos = prediction.clone()
    eos_mask = prediction == EOS_token
    # Find the position of the first predicted EOS token
    eos_positions = torch.argmax(eos_mask.int(), dim=1)

    inds = (
        torch.arange(prediction.shape[1], device=prediction.device)
        .unsqueeze(0)
        .repeat((prediction.shape[0], 1))
    )
    forward_fill_mask = inds > torch.ones_like(inds) * eos_positions.unsqueeze(1)
    pred_without_eos[forward_fill_mask] = null_token
    return pred_without_eos


class DeNovoTeacherForcing(BasePLWrapper):
    def __init__(
        self,
        encoder,
        decoder,
        datasets,
        args,
        collate_fn=None,
        token_dicts=None,
        conf_threshold=None,
    ):
        super().__init__(encoder, datasets, args, collate_fn=collate_fn)
        self.decoder = decoder
        self.denovo_metrics = DeNovoMetrics()
        self.amod_dict = token_dicts["amod_dict"]
        self.int_to_aa = {v: k for k, v in self.amod_dict.items()}
        self.null_token = "X"
        self.conf_threshold = conf_threshold

        self.input_dict = token_dicts["input_dict"]
        self.SOS = self.input_dict["<SOS>"]
        self.output_dict = token_dicts["output_dict"]
        self.EOS = self.output_dict["<EOS>"]

        self.predcats = len(self.output_dict)

        assert all(
            key in self.denovo_metrics.residues
            for key in self.amod_dict
            if key != self.null_token
        ), "All keys except the null token in amod_dict must be in self.denovo_metrics.residues"

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

    def _parse_batch(self, batch):
        input, target = self._input_target(batch)
        batch_size = input.shape[0]
        # Encoder input - mz/ab
        mzab = self._mzab_array(batch)

        parsed_batch = {
            "mz_ab": mzab,
            "mass": batch["mass"],
            "charge": batch["charge"],
            "input_intseq": input,
            "target_intseq": target,
            "peak_lengths": batch["peak_lengths"],
            "peptide_lengths": batch["peptide_lengths"],
        }
        return parsed_batch, batch_size

    def forward(self, parsed_batch, **kwargs):
        outs = self.encoder(
            parsed_batch["mz_ab"],
            length=parsed_batch["peak_lengths"],
            mass=parsed_batch["mass"] if self.encoder.use_mass else None,
            charge=parsed_batch["charge"] if self.encoder.use_charge else None,
            return_mask=True,
            **kwargs
        )
        preds = self.decoder(
            parsed_batch["input_intseq"],
            outs,
            mass=parsed_batch["mass"] if self.decoder.use_mass else None,
            charge=parsed_batch["charge"] if self.decoder.use_charge else None,
            peptide_lengths=parsed_batch["peptide_lengths"],
        )
        return preds

    def eval_forward(self, parsed_batch, **kwargs):
        outs = self.encoder(
            parsed_batch["mz_ab"],
            length=parsed_batch["peak_lengths"],
            mass=parsed_batch["mass"] if self.encoder.use_mass else None,
            charge=parsed_batch["charge"] if self.encoder.use_charge else None,
            return_mask=True,
            **kwargs
        )
        preds = self.decoder.predict_sequence(
            outs,
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
        loss = F.cross_entropy(logits, labels, reduction="none")
        masked_loss = loss * padding_mask
        masked_loss = masked_loss.sum(dim=1, keepdim=True) / padding_mask.sum(
            dim=1, keepdim=True
        )
        return masked_loss.mean()

    def _get_train_stats(self, returns, parsed_batch):
        target = parsed_batch["target_intseq"]
        logits = returns["logits"]
        padding_mask = (~returns["padding_mask"]).int()
        loss = self._get_train_loss(logits, target, padding_mask)
        stats = {"loss": loss}  # , **naive_metrics}
        return loss, stats

    def _get_eval_stats(self, returns, parsed_batch):
        targ = parsed_batch["target_intseq"]
        preds, logits = returns

        preds = preds[:, : targ.shape[1]]
        logits = logits[:, : targ.shape[1]]

        preds_ffill = fill_null_after_first_EOS(
            preds, null_token=self.amod_dict["X"], EOS_token=self.EOS
        )

        logits_ce = logits.transpose(-1, -2)
        aa_confidence, _ = F.softmax(logits, dim=-1).max(dim=-1)
        loss = F.cross_entropy(logits_ce, targ)
        """Accuracy might have little meaning if we are dynamically sizing the sequence length"""
        naive_metrics = NaiveAccRecPrec(targ, preds_ffill, self.amod_dict["X"])
        deepnovo_metrics = self.deepnovo_metrics(preds_ffill, targ, aa_confidence)
        stats = {"loss": loss, **naive_metrics, **deepnovo_metrics}
        return stats

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        return [
            torch.optim.Adam(
                self.encoder.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            ),
            torch.optim.Adam(
                self.decoder.parameters(),
                lr=self.lr,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay,
            ),
        ]

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

    def deepnovo_metrics(self, preds, target, aa_conf):
        mean_conf = aa_conf.mean(dim=-1)
        target_str = self.to_aa_sequence(target)
        pred_str = self.to_aa_sequence(preds)

        (
            aa_prec,
            aa_recall,
            pep_recall,
            pep_precision,
        ) = self.denovo_metrics.compute_precision_recall(
            target_str, pred_str, mean_conf.tolist(), threshold=self.conf_threshold
        )
        pep_auc = self.denovo_metrics.calc_auc(target_str, pred_str, mean_conf.tolist())
        return {
            "aa_prec": aa_prec,
            "aa_recall": aa_recall,
            "pep_recall": pep_recall,
            "pep_precision": pep_precision,
            "pep_auc": pep_auc,
        }


class DeNovoRandom(DeNovoTeacherForcing):
    def __init__(
        self,
        encoder,
        decoder,
        datasets,
        args,
        collate_fn=None,
        token_dicts=None,
        conf_threshold=None,
    ):
        super().__init__(
            encoder, decoder, datasets, args, collate_fn, token_dicts, conf_threshold
        )
        assert (
            "<H>" in token_dicts["input_dict"].keys()
        ), "Needs to include the hidden token"

        self.TASK_NAME = "denovo_random"

    def _get_train_loss(self, preds, labels):
        targ_one_hot = F.one_hot(labels, self.predcats).type(torch.float32)
        preds = preds[self.inds]
        loss = F.cross_entropy(preds, targ_one_hot)
        return loss

    def _parse_batch(self, batch):
        input, target = self._input_target(batch)
        batch_size = input.shape[0]
        # Encoder input - mz/ab
        mzab = self._mzab_array(batch)

        parsed_batch = {
            "mz_ab": mzab,
            "mass": batch["mass"],
            "charge": batch["charge"],
            "input_intseq": input,
            "target_intseq": target,
            "peak_lengths": batch["peak_lengths"],
            "peptide_lengths": batch["peptide_lengths"],
        }
        return parsed_batch, batch_size

    def fill2c(self, int_array, inds, tokentyp="X", output=True):
        tokint = self.null_token if output else self.input_dict[tokentyp]
        all_inds = torch.tile(
            torch.arange(int_array.shape[1], dtype=torch.int32, device=self.device)[
                None
            ],
            [int_array.shape[0], 1],
        )

        out = int_array
        out[all_inds > inds[:, None]] = tokint

        return out

    def _input_target(self, batch):
        intseq = batch["intseq"]

        batch_size, sl = batch["mz_array"].shape

        # Take the variable batch['seqint'] and add a start token to the
        # beginning and null on the end
        # intseq = self.decoder.prepend_startok(batch["intseq"][..., :-1])
        intseq = self._prepend_SOS_tokens(batch["intseq"])

        # Find the indices first null tokens so that when you choose random
        # token you avoid trivial trailing null tokens (beyond final null)
        nonnull = (intseq != self.input_dict["X"]).type(torch.int32).sum(1)

        # Choose random tokens to predict
        # - the values of inds will be final non-hidden value in decoder input
        # - batch['seqint'](inds) will be the target for decoder output
        # - must use combination of rand() and round() because int32 is not
        #   yet implemented when feeding vectors into low/high arguments
        uniform = torch.rand(batch_size, device=nonnull.device) * nonnull
        inds = uniform.floor().type(torch.int32)

        # Fill with hidden tokens to the end
        # - this will be the decoder's input
        intseq_ = self.fill2c(intseq, inds, "<H>", output=False)

        # Indices of chosen predict tokens
        # - save for LossFunction
        inds_ = [torch.arange(inds.shape[0], dtype=torch.int32), inds]
        self.inds = inds_

        # Target is the actual (intseq) identity of the chosen predict indices
        targ = batch["intseq"][inds_].type(torch.int64)

        return intseq_, targ

    def eval_forward(self, parsed_batch, **kwargs):
        outs = self.encoder(
            parsed_batch["mz_ab"],
            length=parsed_batch["peak_lengths"],
            mass=parsed_batch["mass"] if self.encoder.use_mass else None,
            charge=parsed_batch["charge"] if self.encoder.use_charge else None,
            return_mask=True,
            **kwargs
        )
        preds = self.decoder.predict_sequence(
            outs,
            mass=parsed_batch["mass"] if self.decoder.use_mass else None,
            charge=parsed_batch["charge"] if self.decoder.use_charge else None,
            causal=False,
        )
        return preds
