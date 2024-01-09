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
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        parsed_batch, batch_size = self._parse_batch(batch)
        returns = self.forward(parsed_batch)
        loss, train_stats = self._get_train_stats(returns, parsed_batch)
        train_stats = {
            "train_" + key: val.detach().item() for key, val in train_stats.items()
        }
        self.log_dict(
            {**train_stats},
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.running_train_loss.update(train_stats["train_loss"])
        self.log(
            "run_train_loss:", self.running_train_loss.get_running_loss(), prog_bar=True,
            sync_dist=True,
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
            "val_" + key: val.detach().item() for key, val in val_stats.items()
        }
        self.log_dict(
            {**val_stats},
            on_epoch=True,
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.running_val_loss.update(val_stats["val_loss"])
        self.log(
            "run_val_loss:", self.running_val_loss.get_running_loss(), prog_bar=True,
            sync_dist=True,
        )
        return {"val_stats": val_stats, "returns": returns}

    def test_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch)
        returns = self.forward(parsed_batch)
        test_stats = self._get_eval_stats(returns, parsed_batch)
        test_stats = {
            "test_" + key: val.detach().item() for key, val in test_stats.items()
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
            shuffle=True
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
        self.predcats = len(amod_dict)

    def _mzab_array(self, batch):
        mz_arr = batch["mz_array"]
        int_arr = batch["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        return mzab

    def _encoder_dict(self, batch):
        return {
            "mass": batch["mass"] if self.encoder.use_mass else None,
            "charge": batch["charge"] if self.encoder.use_charge else None,
            "length": batch["lengths"] if self.mask_zero_tokens else None,
        }

    def _decoder_dict(self, batch):
        return {
            "mass": batch["mass"] if self.decoder.decoder.use_mass else None,
            "charge": batch["charge"] if self.decoder.decoder.use_charge else None,
        }


    def _parse_batch(self, batch, full_seqint=False):
        spectra = batch
        intseq = spectra["intseq"]
        
        batch_size, sl = spectra['mz_array'].shape
        
        # Encoder input - mz/ab
        mzab = self._mzab_array(spectra)

        # Take the variable batch['seqint'] and add a start token to the
        # beginning and null on the end
        intseq = self.decoder.prepend_startok(batch['intseq'][..., :-1])

        # Find the indices first null tokens so that when you choose random
        # token you avoid trivial trailing null tokens (beyond final null)
        nonnull = (intseq != self.decoder.inpdict['X']).type(torch.int32).sum(1)

        # Choose random tokens to predict
        # - the values of inds will be final non-hidden value in decoder input
        # - batch['seqint'](inds) will be the target for decoder output
        # - must use combination of rand() and round() because int32 is not
        #   yet implemented when feeding vectors into low/high arguments
        uniform = torch.rand(batch_size, device=nonnull.device) * nonnull
        inds = uniform.floor().type(torch.int32)

        # Fill with hidden tokens to the end
        # - this will be the decoder's input
        intseq_ = self.decoder.fill2c(intseq, inds, '<h>', output=False)

        # Indices of chosen predict tokens
        # - save for LossFunction
        inds_ = [torch.arange(inds.shape[0], dtype=torch.int32), inds]
        self.inds = inds_

        # Target is the actual (intseq) identity of the chosen predict indices
        targ = (
            batch['intseq'] if full_seqint else batch['intseq'][inds_]
        ).type(torch.int64)

        # Encoder input - non mz/ab
        encoder_input_dict = self._encoder_dict(spectra)
        #encoder_input_dict = {
        #    "mass": spectra["mass"] if self.encoder.use_mass else None,
        #    "charge": spectra["charge"] if self.encoder.use_charge else None,
        #    "length": spectra["lengths"] if self.mask_zero_tokens else None,
        #}

        # Decoder input
        decoder_input_dict = self._decoder_dict(batch)
        #decoder_input_dict = {
        #    "mass": spectra["mass"] if self.decoder.decoder.use_mass else None,
        #    "charge": spectra["charge"] if self.decoder.decoder.use_charge else None,
        #}

        return (mzab, encoder_input_dict, intseq_, decoder_input_dict, targ), batch_size

    def forward(self, parsed_batch, **kwargs):
        mzab, encoder_input_dict, intseq, decoder_input_dict, _ = parsed_batch
        outs = self.encoder(mzab, **encoder_input_dict, return_mask=True, **kwargs)
        preds = self.decoder(intseq, outs, decoder_input_dict)
        return preds

    def _get_losses(self, preds, labels):
        targ_one_hot = F.one_hot(labels, self.predcats).type(torch.float32)
        preds = preds[self.inds]
        loss = F.cross_entropy(preds, targ_one_hot)
        return loss

    def _get_train_stats(self, returns, batch):
        _, _, intseq, _, targ = batch
        preds = returns
        loss = self._get_losses(preds, targ)
        #naive_metrics = NaiveAccRecPrec(
        #    intseq, preds.argmax(dim=-1), self.amod_dict["X"]
        #)
        stats = {"loss": loss}#, **naive_metrics}
        return loss, stats

    def validation_step(self, batch, batch_idx, **kwargs):
        mzab = self._mzab_array(batch)
        batch_size = mzab.shape[0]
        encinpdict = self._encoder_dict(batch)
        encout = self.encoder(mzab, **encinpdict, return_mask=True, **kwargs)
        decinpdict = self._decoder_dict(batch)
        returns = self.decoder.predict_sequence(encout, decinpdict)
        val_stats = self._get_eval_stats(returns, batch)
        val_stats = {
            "val_" + key: val.detach().item() for key, val in val_stats.items()
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

    def _get_eval_stats(self, returns, batch):
        targ = batch['intseq']
        preds = returns[:, :targ.shape[1]]
        loss = torch.tensor(0)#self._get_losses(preds, targ)
        """Accuracy might have little meaning if we are dynamically sizing the sequence length"""
        naive_metrics = NaiveAccRecPrec(
            targ, preds, self.amod_dict["X"]
        )
        stats = {"loss": loss, **naive_metrics}
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
