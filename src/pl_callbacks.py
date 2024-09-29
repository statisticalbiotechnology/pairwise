import pytorch_lightning as pl
import math
from pytorch_lightning.utilities.rank_zero import rank_zero_only

# from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler


class CosineAnnealLRCallback(pl.Callback):
    """
    Callback to adjust the learning rate using the cosine annealing schedule with warmup.

    This callback supports adjusting the learning rate either per epoch or per step.

    Args:
        lr_start (float): The starting learning rate before warmup.
        blr (float): The base learning rate after warmup.
        lr_end (float): The minimum learning rate after cosine annealing.
        warmup_duration (int): The number of warmup epochs or steps before applying cosine annealing.
        anneal_per_step (bool): Whether to anneal the learning rate per step. If False, anneal per epoch. Default is False.
    """

    def __init__(self, lr_start, blr, lr_end, warmup_duration, anneal_per_step=False):
        self.lr_start = lr_start
        self.blr = blr
        self.lr_end = lr_end
        self.warmup_duration = warmup_duration
        self.anneal_per_step = anneal_per_step

    def _calculate_lr(self, current, total):
        if current < self.warmup_duration:
            fac = current / self.warmup_duration
            lr_temp = self.blr * fac + self.lr_start * (1 - fac)
        else:
            lr_temp = self.lr_end + (self.blr - self.lr_end) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (current - self.warmup_duration)
                    / (total - self.warmup_duration)
                )
            )
        return lr_temp

    def _update_optimizer_lr(self, trainer, lr_temp):
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr_temp * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr_temp

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.anneal_per_step:
            epoch = trainer.current_epoch
            tot_epochs = trainer.max_epochs
            lr_temp = self._calculate_lr(epoch, tot_epochs)
            self._update_optimizer_lr(trainer, lr_temp)
            pl_module.lr = lr_temp

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.anneal_per_step:
            current_step = trainer.global_step // 2 # 2 separate optimizers tick the global_step twice
            total_steps = trainer.estimated_stepping_batches  # TODO: fix: * max_epochs
            lr_temp = self._calculate_lr(current_step, total_steps)
            self._update_optimizer_lr(trainer, lr_temp)
            pl_module.lr = lr_temp


class LinearWarmupLRCallback(pl.Callback):
    def __init__(self, starting_lr, ending_lr, warmup_steps):
        self.slr = starting_lr
        self.elr = ending_lr
        self.warmup_steps = warmup_steps

        self.incr = (ending_lr - starting_lr) / warmup_steps
        self.current_step = 0

    def on_train_start(self, trainer, pl_module):
        self.num_opts = len(trainer.optimizers)
        for optimizer in trainer.optimizers:
            optimizer.param_groups[0]["lr"] = self.slr

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.current_step < self.warmup_steps * self.num_opts:
            for optimizer in trainer.optimizers:
                optimizer.param_groups[0]["lr"] += self.incr
                self.current_step += 1


class ExponentialDecayLRCallback(pl.Callback):
    def __init__(self, starting_step, ending_step, decay):
        assert ending_step > starting_step
        self.ss = starting_step
        self.es = ending_step
        self.decay = decay

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        num_opts = len(trainer.optimizers)
        current_step = trainer.global_step // num_opts
        if current_step >= self.ss:
            factor = self.decay if current_step <= self.es else 1
            for optimizer in trainer.optimizers:
                optimizer.param_groups[0]["lr"] *= factor


class FLOPProfilerCallback(pl.Callback):
    """Measures the number of FLOPs during the first forward pass on the first batch"""

    def __init__(self):
        self.profiler = None
        self.measured = False  # To track if profiling has been triggered

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        self.profiler = FlopsProfiler(pl_module)

    @rank_zero_only
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.measured and trainer.current_epoch == 0 and batch_idx == 0:
            self.profiler.start_profile()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.measured and trainer.current_epoch == 0 and batch_idx == 0:
            flops = self.profiler.get_total_flops()
            pl_module.log("FLOPs_inference", float(flops), on_step=True, on_epoch=False)
            self.measured = True  # Mark as measured after profiling first batch
            self.profiler.end_profile()
            self.profiler = None  # Delete profiler when done
