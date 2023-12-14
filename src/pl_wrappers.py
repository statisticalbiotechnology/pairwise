import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from abc import ABC, abstractmethod
from collate_functions import pad_length_collate_fn


def calc_classification_metrics(all_outputs, all_targets):
    all_targets = all_targets.detach().cpu().float().numpy()
    all_outputs = all_outputs.detach().cpu().float().numpy()

    # Compute accuracy
    accuracy = accuracy_score(all_targets.argmax(axis=-1), all_outputs.argmax(axis=-1))

    # Compute precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets.argmax(axis=-1),
        all_outputs.argmax(axis=-1),
        average="weighted",
        zero_division=0,
    )

    # Compute confusion matrix
    # confusion_mat = confusion_matrix(all_targets.argmax(axis=-1), all_outputs.argmax(axis=-1))

    metrics = dict(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        # confusion_mat=confusion_mat,
    )
    return metrics


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
        self.input_charge = bool(args.input_charge)
        self.input_mass = bool(args.input_mass)
        self.tracker = BestMetricTracker()
        self.best_metrics_logged = (
            False  # keep track of if the best achieved metrics have been logged
        )

    def _get_input(self, batch):
        spectra = batch
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        return mzab, {
            "mass": spectra["mass"] if self.input_mass else None,
            "charge": spectra["charge"] if self.input_charge else None,
        }

    def forward(self, inputs, **kwargs):
        mzab, input_dict = inputs
        outs = self.encoder(mzab, **input_dict, **kwargs)
        if self.head is not None:
            outs = self.head(outs)
        return outs

    @abstractmethod
    def _get_losses(self, returns, labels):
        # example
        # preds = returns["preds"]
        # loss = self.loss_fn(preds, labels)
        # return loss
        pass

    @abstractmethod
    def _get_train_stats(self, returns, batch):
        # define what metrics to log during training steps
        # note: must return the differentiable loss
        stats = {}
        # example

        input, labels = batch
        loss = self._get_losses(returns, labels)
        # stats["loss"] = loss
        # metrics = calc_classification_metrics(returns["preds"], labels)
        # stats = {**stats, **metrics}

        stats = {"train_" + key: val.detach().item() for key, val in stats.items()}
        return loss, stats

    @abstractmethod
    def _get_eval_stats(self, split, returns, batch):
        # define what metrics to log during val/test steps
        stats = {}
        # example
        # input, labels = batch
        # loss = self._get_losses(returns, labels)
        # stats["loss"] = loss
        # metrics = calc_classification_metrics(returns["preds"], labels)
        # stats = {**stats, **metrics}
        stats = {split + "_" + key + name_suffix: val for key, val in metrics.items()}
        return stats

    def training_step(self, batch, batch_idx):
        inputs = self._get_input(batch)
        returns = self.forward(inputs)
        loss, train_stats = self._get_train_stats(returns, batch)
        self.log_dict(
            {**train_stats}, on_epoch=True, batch_size=batch[0].shape[0], sync_dist=True
        )
        return {"loss": loss, "returns": returns}

    def validation_step(self, batch, batch_idx):
        inputs = self._get_input(batch)
        returns = self.forward(inputs)
        val_stats = self._get_eval_stats("val", returns, batch)
        self.log_dict(
            {**val_stats}, on_epoch=True, batch_size=batch[0].shape[0], sync_dist=True
        )
        return {"val_stats": val_stats, "returns": returns}

    def test_step(self, batch, batch_idx):
        inputs = self._get_input(batch)
        returns = self.forward(inputs)
        test_stats = self._get_eval_stats("test", returns, batch)
        self.log_dict(
            {**test_stats}, on_epoch=True, batch_size=batch[0].shape[0], sync_dist=True
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[1],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[2],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=False,
            collate_fn=self.collate_fn,
        )

    def configure_optimizers(self):
        param_groups = [
            {"params": self.encoder.parameters()},
        ]
        if self.head:
            param_groups += [{"params": self.head.parameters()}]
        return torch.optim.AdamW(
            param_groups,
            lr=self.lr,
            betas=(0.9, 0.9999),
            weight_decay=self.weight_decay,
        )

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
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=False, on_epoch=True)

    def on_train_end(self):
        if not self.best_metrics_logged:
            self.logger.experiment.log(self.tracker.best_metrics)

