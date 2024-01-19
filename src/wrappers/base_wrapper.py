import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from collections import deque


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
        self.use_charge = args.use_charge
        self.use_mass = args.use_mass
        self.mask_zero_tokens = args.mask_zero_tokens
        self.log_wandb = args.log_wandb
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
                "mass": spectra["mass"] if self.use_mass else None,
                "charge": spectra["charge"] if self.use_charge else None,
            },
        ), batch_size

    def forward(self, parsed_batch, **kwargs):
        mzab, input_dict, target = parsed_batch
        outs = self.encoder(mzab, **input_dict, **kwargs)
        if self.head is not None:
            outs = self.head(outs)
        return outs

    def eval_forward(self, parsed_batch, **kwargs):
        return self.forward(parsed_batch, **kwargs)

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
            on_step=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        self.running_train_loss.update(train_stats["train_loss"])
        self.log(
            "run_train_loss:",
            self.running_train_loss.get_running_loss(),
            prog_bar=True,
            sync_dist=True,
        )

        self.manual_backward(loss)
        for opt in opts:
            opt.step()
        return {"loss": loss, "returns": returns}

    def validation_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch)
        returns = self.eval_forward(parsed_batch)
        val_stats = self._get_eval_stats(returns, parsed_batch)
        val_stats = {
            "val_" + key: val.detach().item() if isinstance(val, torch.Tensor) else val
            for key, val in val_stats.items()
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
            "run_val_loss:",
            self.running_val_loss.get_running_loss(),
            prog_bar=True,
            sync_dist=True,
        )
        return {"val_stats": val_stats, "returns": returns}

    def test_step(self, batch, batch_idx):
        parsed_batch, batch_size = self._parse_batch(batch)
        returns = self.eval_forward(parsed_batch)
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
            shuffle=True,
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
        if not self.best_metrics_logged and self.log_wandb:
            self.logger.experiment.log(self.tracker.best_metrics)
