
import pytorch_lightning as pl
import math
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler


class CosineAnnealLRCallback(pl.Callback):
    """Callback to adjust the learning rate using the cosine annealing schedule with warmup.

    Args:
        lr (float): The initial learning rate.
        min_lr (float): The minimum learning rate.
        warmup_epochs (int): The number of warmup epochs before applying cosine annealing.

    """

    def __init__(self, lr, min_lr, warmup_epochs):
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        tot_epochs = trainer.max_epochs

        if epoch < self.warmup_epochs:
            lr_temp = self.lr * epoch / self.warmup_epochs
        else:
            lr_temp = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - self.warmup_epochs)
                    / (tot_epochs - self.warmup_epochs)
                )
            )

        optimizer = trainer.optimizers[0]
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr_temp * param_group["lr_scale"]
            else:
                param_group["lr"] = lr_temp
        pl_module.lr = lr_temp


class FLOPProfilerCallback(pl.Callback):
    """ Measures the number of FLOPs during the first forward pass on the first batch
    """
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
            self.profiler = None # Delete profiler when done

