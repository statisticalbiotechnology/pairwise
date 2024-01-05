import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from abc import ABC, abstractmethod
import torch.nn.functional as F
from collections import deque
from denovo_eval import Metrics as DeNovoMetrics
from models.heads import ClassifierHead


def NaiveAccRecPrec(target, prediction, null_value):
    correct_bool = (target == prediction).type(torch.int32)
    num_correct = correct_bool.sum()
    recall_bool = target != null_value
    recsum = correct_bool[recall_bool].sum()
    prec_bool = prediction != null_value
    precsum = correct_bool[prec_bool].sum()
    total = target.shape[0] * target.shape[1]
    return {
        "acc_naive": num_correct / total,
        "recall_naive": recsum / recall_bool.sum(),
        "precision_naive": precsum / prec_bool.sum(),
    }


class RunningWindowLoss:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.loss_window = deque(maxlen=window_size)

    def update(self, loss):
        self.loss_window.append(loss)

    def get_running_loss(self):
        return (
            sum(self.loss_window) / len(self.loss_window) if self.loss_window else 0.0
        )


class BestMetricTracker:
    def __init__(self):
        self.best_metrics = {}
        self.optimization_direction = (
            {}
        )  # Tracks the optimization direction for each metric

    def update_metric(self, name, value, maximize=True):
        if name not in self.best_metrics or self._is_better(
            value, self.best_metrics[name], maximize
        ):
            self.best_metrics[name] = value
            self.optimization_direction[name] = maximize

    def get_best_metric(self, name):
        return self.best_metrics.get(name, None)

    def _is_better(self, new_value, old_value, maximize):
        if maximize:
            return new_value > old_value
        else:
            return new_value < old_value


class BasePLWrapper(ABC, pl.LightningModule):
    """
    PyTorch Lightning module wrapping the encoder and the head for unsupervised pretraining.

    This wrapper class provides integration with PyTorch Lightning's train, val and test loop.

    Args:
        encoder (pl.LightningModule): The encoder model
        head (pl.LightningModule): The objective-specific projection head for the encoder
        args (argparse.Namespace): Command-line arguments or a configuration namespace
            containing hyperparameters and settings.
        datasets (list): A list containing the train, val, and test datasets.
    """

    def __init__(self, encoder, datasets, args, head=None, collate_fn=None):
        super().__init__()

        self.encoder = encoder
        self.head = head
        self.datasets = datasets
        self.collate_fn = collate_fn
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.pin_mem = args.pin_mem
        self.weight_decay = args.weight_decay
        self.datasets = datasets
        self.lr = args.lr
        self.input_charge = args.input_charge
        self.input_mass = args.input_mass
        self.mask_zero_tokens = args.mask_zero_tokens
        self.tracker = BestMetricTracker()
        self.best_metrics_logged = (
            False  # keep track of if the best achieved metrics have been logged
        )
        self.running_train_loss = RunningWindowLoss()
        self.running_val_loss = RunningWindowLoss()
        self.automatic_optimization = False

    def _parse_batch(self, batch):
        spectra = batch
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        batch_size = mz_arr.shape[0]
        return (
            mzab,
            {
                "mass": spectra["mass"] if self.input_mass else None,
                "charge": spectra["charge"] if self.input_charge else None,
            },
        ), batch_size

    def forward(self, parsed_batch, **kwargs):
        mzab, input_dict, target = parsed_batch
        outs = self.encoder(mzab, **input_dict, **kwargs)
        if self.head is not None:
            outs = self.head(outs)
        return outs

    @abstractmethod
    def _get_losses(self, returns, parsed_batch):
        # example
        # preds = returns["preds"]
        # loss = self.loss_fn(preds, labels)
        # return loss
        pass

    @abstractmethod
    def _get_train_stats(self, returns, parsed_batch):
        # define what metrics to log during training steps
        # note: must return the differentiable loss
        stats = {}
        # example

        input, labels = batch
        loss = self._get_losses(returns, labels)
        # stats["loss"] = loss
        # metrics = calc_classification_metrics(returns["preds"], labels)
        # stats = {**stats, **metrics}

        # stats = {"train_" + key: val.detach().item() for key, val in stats.items()}
        return loss, stats

    @abstractmethod
    def _get_eval_stats(self, returns, parsed_batch):
        # define what metrics to log during val/test steps
        stats = {}
        # example
        # input, labels = batch
        # loss = self._get_losses(returns, labels)
        # stats["loss"] = loss
        # metrics = calc_classification_metrics(returns["preds"], labels)
        # stats = {**stats, **metrics}
        return stats

    def training_step(self, batch, batch_idx):
        opts = self.optimizers()
        for opt in opts:
            opt.zero_grad()
        parsed_batch, batch_size = self._parse_batch(batch)
        returns = self.forward(parsed_batch)
        loss, train_stats = self._get_train_stats(returns, parsed_batch)
        train_stats = {
            "train_" + key: val.detach().item()
            if isinstance(val, torch.Tensor)
            else val
            for key, val in train_stats.items()
        }
        self.log_dict(
            {**train_stats},
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.running_train_loss.update(train_stats["train_loss"])
        self.log(
            "run_train_loss:", self.running_train_loss.get_running_loss(), prog_bar=True
        )

        self.manual_backward(loss)
        for opt in opts:
            opt.step()
        return {"loss": loss, "returns": returns}

    def validation_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch)
        returns = self.forward(parsed_batch)
        val_stats = self._get_eval_stats(returns, parsed_batch)
        val_stats = {
            "val_" + key: val.detach().item() if isinstance(val, torch.Tensor) else val
            for key, val in val_stats.items()
        }
        self.log_dict(
            {**val_stats},
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.running_val_loss.update(val_stats["val_loss"])
        self.log(
            "run_val_loss:", self.running_val_loss.get_running_loss(), prog_bar=True
        )
        return {"val_stats": val_stats, "returns": returns}

    def test_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch)
        returns = self.forward(parsed_batch)
        test_stats = self._get_eval_stats(returns, parsed_batch)
        test_stats = {
            "test_" + key: val.detach().item() if isinstance(val, torch.Tensor) else val
            for key, val in test_stats.items()
        }
        self.log_dict(
            {**test_stats}, on_epoch=True, batch_size=batch_size, sync_dist=True
        )
        return {"test_stats": test_stats, "returns": returns}

    def train_dataloader(self):
        return DataLoader(
            self.datasets[0],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[1],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[2],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=False,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

    def configure_optimizers(self):
        opts = [
            torch.optim.AdamW(
                self.encoder.parameters(),
                lr=self.lr,
                betas=(0.9, 0.9999),
                weight_decay=self.weight_decay,
            ),
        ]
        if self.head:
            opts.append(
                torch.optim.AdamW(
                    self.head.parameters(),
                    lr=self.lr,
                    betas=(0.9, 0.9999),
                    weight_decay=self.weight_decay,
                ),
            )
        return opts

    @abstractmethod
    def on_validation_epoch_end(self):
        # Update the current best achieved value for each val metric
        # Get the per-epoch metric value from the logged metrics

        cur_epoch = self.trainer.current_epoch
        if self.global_rank == 0:  # Only log on master process
            if cur_epoch > 0:
                metrics = self.trainer.logged_metrics

                # example
                # self.tracker.update_metric(
                #     "best_val_loss",
                #     metrics["val_loss"].detach().cpu().item(),
                #     maximize=False,
                # )

                # self.tracker.update_metric(
                #     "best_val_r2_score",
                #     metrics["val_r2_score"].detach().cpu().item(),
                #     maximize=True,
                # )

            # TODO: verify: don't think this part is needed bc of the "on_train_end"
            # # at the last epoch, log the best metrics
            # if cur_epoch == self.trainer.max_epochs - 1:
            #     self.log_dict(self.tracker.best_metrics)
            #     self.best_metrics_logged = True

    def on_train_epoch_end(self):  # log the learning rate
        opts = self.optimizers()
        cur_lr_enc = opts[0].param_groups[0]["lr"]
        self.log("lr_enc", cur_lr_enc, on_step=False, on_epoch=True)
        cur_lr_head = opts[1].param_groups[0]["lr"]
        self.log("lr_head", cur_lr_head, on_step=False, on_epoch=True)

    def on_train_end(self):
        if not self.best_metrics_logged:
            self.logger.experiment.log(self.tracker.best_metrics)


class DummyPLWrapper(BasePLWrapper):
    def _get_losses(self, returns, labels):
        return 0

    def _get_train_stats(self, returns, batch):
        stats = {}
        return 0, stats

    def _get_eval_stats(self, split, returns, batch):
        stats = {}
        return stats

    def on_validation_epoch_end(self):
        pass


class MaskedTrainingPLWrapper(DummyPLWrapper):
    pass


class TrinaryMZPLWrapper(BasePLWrapper):
    def __init__(self, encoder, datasets, args, collate_fn=None):
        self.penult_units = args.trinary_penult_units
        head = ClassifierHead(3, encoder.running_units, self.penult_units)
        super().__init__(encoder, datasets, args, head, collate_fn)
        self.corrupt_freq = args.trinary_freq
        self.corrupt_std = args.trinary_std

    def _parse_batch(self, batch):
        spectra = batch
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        corrupt_mz_arr, target = self.inptarg(mz_arr)

        batch_size = mz_arr.shape[0]
        mzab = torch.stack([corrupt_mz_arr, int_arr], dim=-1)
        return (
            mzab,
            {
                "mass": spectra["precursor_mz"] if self.input_mass else None,
                "charge": spectra["precursor_charge"] if self.input_charge else None,
                "length": spectra["lengths"] if self.mask_zero_tokens else None,
            },
            target,
        ), batch_size

    def forward(self, parsed_batch, **kwargs):
        mzab, input_dict, target = parsed_batch
        outs = self.encoder(mzab, **input_dict, **kwargs)
        outs = self.head(outs["emb"])
        return outs

    def _get_losses(self, returns, labels):
        loss = F.cross_entropy(
            returns.transpose(-1, -2),
            labels,
            reduction="mean",  # TODO: verify that mean reduction is what we want here
        )
        return loss

    def _get_train_stats(self, returns, parsed_batch):
        _, _, target = parsed_batch
        stats = {}
        loss = self._get_losses(returns, target)
        stats["loss"] = loss
        return loss, stats

    def _get_eval_stats(self, returns, parsed_batch):
        _, _, target = parsed_batch
        stats = {}
        loss = self._get_losses(returns, target)
        stats["loss"] = loss
        return stats

    def inptarg(self, mz_batch):
        # Random sequence indices to change
        inds_boolean = torch.empty_like(mz_batch).uniform_(0, 1) < self.corrupt_freq
        inds = torch.nonzero(inds_boolean, as_tuple=False)

        # Get their mz values
        means = mz_batch[inds[:, 0], inds[:, 1]]

        # Generate normal distributions for inds, centered on original value
        updates = torch.normal(means, self.corrupt_std)
        # updates = torch.clamp(updates, min=0.0, max=1.0)

        # Distribute updates into corrupted indices
        mz_batch[inds[:, 0], inds[:, 1]] = updates

        # Construct Target Tensor
        # inds that are below the original value (0)
        zero = means > updates  # 1d boolean
        target = torch.ones_like(mz_batch, dtype=torch.long)
        target[inds[zero, 0], inds[zero, 1]] = 0

        # inds that are above the original value (2)
        two = means < updates  # 1d boolean
        target[inds[two, 0], inds[two, 1]] = 2

        # target = F.one_hot(target, 3) # Not needed, CE loss expects class inds for the target

        return mz_batch, target

    def on_validation_epoch_end(self):
        # Update the current best achieved value for each val metric
        # Get the per-epoch metric value from the logged metrics

        cur_epoch = self.trainer.current_epoch
        if self.global_rank == 0:  # Only log on master process
            if cur_epoch > 0:
                metrics = self.trainer.logged_metrics

                self.tracker.update_metric(
                    "best_val_loss",
                    metrics["val_loss"].detach().cpu().item(),
                    maximize=False,
                )

            # TODO: verify: don't think this part is needed bc of the "on_train_end"
            # at the last epoch, log the best metrics
            if cur_epoch == self.trainer.max_epochs - 1:
                self.log_dict(self.tracker.best_metrics)
                self.best_metrics_logged = True


class DeNovoPLWrapper(BasePLWrapper):
    def __init__(
        self, encoder, decoder, datasets, args, collate_fn=None, amod_dict=None
    ):
        super().__init__(encoder, datasets, args, collate_fn=collate_fn)
        self.decoder = decoder
        self.denovo_metrics = DeNovoMetrics()
        self.amod_dict = amod_dict
        self.int_to_aa = {v: k for k, v in amod_dict.items()}
        self.null_token = "X"

        assert all(
            key in self.denovo_metrics.residues
            for key in self.amod_dict
            if key != self.null_token
        ), "All keys except the null token in amod_dict must be in self.denovo_metrics.residues"

    def to_aa_sequence(self, int_tensors: torch.Tensor | list):
        # Check if the input is a Tensor and convert it to a list
        if isinstance(int_tensors, torch.Tensor):
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

    def deepnovo_metrics(self, preds, target, aa_conf, threshold=None):
        mean_conf = aa_conf.mean(dim=-1)
        target_str = self.to_aa_sequence(target)
        pred_str = self.to_aa_sequence(preds)
        (
            aa_prec,
            aa_recall,
            pep_recall,
            pep_precision,
        ) = self.denovo_metrics.compute_precision_recall(
            target_str, pred_str, aa_conf.tolist()
        )
        pep_auc = self.denovo_metrics.calc_auc(target_str, pred_str, mean_conf.tolist())
        return {
            "aa_prec": aa_prec,
            "aa_recall": aa_recall,
            "pep_recall": pep_recall,
            "pep_precision": pep_precision,
            "pep_auc": pep_auc,
        }

    def _parse_batch(self, batch):
        spectra = batch
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        intseq = spectra["intseq"]

        batch_size = mz_arr.shape[0]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)

        encoder_input_dict = {
            "mass": spectra["mass"] if self.input_mass else None,
            "charge": spectra["charge"] if self.input_charge else None,
            "length": spectra["lengths"] if self.mask_zero_tokens else None,
        }

        decoder_input_dict = {
            "mass": spectra["mass"] if self.input_mass else None,
            "charge": spectra["charge"] if self.input_charge else None,
        }

        return (mzab, encoder_input_dict, intseq, decoder_input_dict), batch_size

    def forward(self, parsed_batch, **kwargs):
        mzab, encoder_input_dict, intseq, decoder_input_dict = parsed_batch
        outs = self.encoder(mzab, **encoder_input_dict, return_mask=True, **kwargs)
        preds = self.decoder(intseq, outs, decoder_input_dict)
        return preds

    def _get_losses(self, preds, labels):
        preds = torch.einsum("bsc->bcs", preds)
        loss = F.cross_entropy(preds, labels)
        return loss

    def _get_train_stats(self, returns, batch):
        _, _, intseq, _ = batch
        logits = returns
        aa_conf, preds = logits.max(dim=-1)
        loss = self._get_losses(logits, intseq)
        naive_metrics = NaiveAccRecPrec(intseq, preds, self.amod_dict["X"])
        deepnovo_metrics = self.deepnovo_metrics(preds, intseq, aa_conf)
        stats = {"loss": loss, **naive_metrics, **deepnovo_metrics}
        return loss, stats

    def _get_eval_stats(self, returns, batch):
        _, _, intseq, _ = batch
        logits = returns
        aa_conf, preds = logits.max(dim=-1)
        loss = self._get_losses(logits, intseq)
        naive_metrics = NaiveAccRecPrec(intseq, preds, self.amod_dict["X"])
        deepnovo_metrics = self.deepnovo_metrics(preds, intseq, aa_conf)
        stats = {"loss": loss, **naive_metrics, **deepnovo_metrics}
        return stats

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        return [
            torch.optim.Adam(
                self.encoder.parameters(),
                lr=self.lr,
                betas=(0.9, 0.9999),
                weight_decay=self.weight_decay,
            ),
            torch.optim.Adam(
                self.decoder.parameters(),
                lr=self.lr,
                betas=(0.9, 0.9999),
                weight_decay=self.weight_decay,
            ),
        ]
