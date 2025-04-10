import os
import warnings
import numpy as np
import pytorch_lightning as pl
import math
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch.distributed as dist
from data.mzTab_writer import MztabWriter

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

    def __init__(
        self,
        lr_start,
        blr,
        lr_end,
        warmup_duration,
        decay_delay=0,
        decay_duration=100000,
        anneal_per_step=False,
    ):
        self.lr_start = lr_start
        self.blr = blr
        self.lr_end = lr_end
        self.warmup_duration = warmup_duration
        self.anneal_per_step = anneal_per_step
        self.decay_delay = decay_delay
        self.decay_duration = decay_duration
        self.delay_ticker = 0
        self.decay_ticker = 0

    def _calculate_lr(self, current, total):
        if current < self.warmup_duration:
            fac = current / self.warmup_duration
            lr_temp = self.blr * fac + self.lr_start * (1 - fac)
        elif (self.decay_duration > 0) & (self.delay_ticker >= self.decay_delay):

            if self.anneal_per_step:
                if self.decay_ticker >= self.decay_duration:
                    return self.lr_end
                else:
                    denominator = self.decay_duration
                    self.decay_ticker += 1
            else:
                denominator = total - self.decay_delay - self.warmup_duration

            lr_temp = self.lr_end + (self.blr - self.lr_end) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (current - self.decay_delay - self.warmup_duration)
                    / denominator
                )
            )
        else:
            lr_temp = self.blr
            self.delay_ticker += 1

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
            current_step = (
                trainer.global_step // 2
            )  # 2 separate optimizers tick the global_step twice
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


class MztabOutputCallback(pl.Callback):
    """
    Callback to write predictions to an mzTab file during testing,
    handling outputs batch by batch to be memory-efficient.

    Args:
        log_dir (str): Directory where the mzTab output file will be saved.
        global_args (argparse.Namespace): Parsed command-line arguments.
    """

    def __init__(self, log_dir, global_args):
        super().__init__()
        self.output_file = os.path.join(log_dir, "predictions_table.mzTab")
        self.global_args = global_args
        self.writer = None  # Will be initialized in setup

    @rank_zero_only
    def setup(self, trainer, pl_module, stage=None):
        self.writer = MztabWriter(self.output_file)
        self.writer.set_metadata(vars(self.global_args))
        self.writer.open_file()
        self.writer.write_headers()

    @rank_zero_only
    def teardown(self, trainer, pl_module, stage=None):
        if self.writer:
            self.writer.close_file()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        all_outputs = self._gather_outputs(outputs, pl_module)

        if trainer.is_global_zero:
            predictions = all_outputs["predictions"]
            peptides_true = all_outputs["peptides_true"]

            for idx, prediction in enumerate(predictions):
                peptide = prediction["peptide"]
                peak_file = prediction["peak_file"]
                scan_id = prediction["scan_id"]
                charge = prediction["precursor_charge"]
                precursor_mz = prediction["precursor_mz"]
                peptide_score = prediction["peptide_score"]
                aa_scores = prediction["aa_scores"]
                ground_truth_sequence = peptides_true[idx]
                calc_mz = pl_module.peptide_mass_calculator.mass(peptide, charge)
                title = prediction["title"]

                # Convert list fields to strings
                if isinstance(peptide, list):
                    peptide = ",".join(peptide)
                if isinstance(ground_truth_sequence, list):
                    ground_truth_sequence = ",".join(ground_truth_sequence)
                if isinstance(aa_scores, (list, np.ndarray)):
                    aa_scores = ",".join(f"{score:.5f}" for score in aa_scores)

                psm_entry = {
                    "sequence": peptide,
                    "PSM_ID": None,
                    "accession": "null",
                    "unique": "null",
                    "database": "null",
                    "database_version": "null",
                    "search_engine": "[MS, MS:1001456, CustomModel, 1.0]",
                    "search_engine_score[1]": peptide_score,
                    "modifications": "null",
                    "retention_time": "null",
                    "charge": charge,
                    "exp_mass_to_charge": precursor_mz,
                    "calc_mass_to_charge": calc_mz,
                    "spectra_ref": f"{peak_file}:{scan_id}",
                    "pre": "null",
                    "post": "null",
                    "start": "null",
                    "end": "null",
                    "opt_ms_run[1]_aa_scores": aa_scores,
                    "opt_ms_run[1]_ground_truth_sequence": ground_truth_sequence,
                    "title": title,
                }
                self.writer.write_psm(psm_entry)

            self.writer.flush()

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        all_outputs = self._gather_outputs(outputs, pl_module)

        if trainer.is_global_zero:
            predictions = all_outputs["predictions"]

            for idx, prediction in enumerate(predictions):
                peptide = prediction["peptide"]
                peak_file = prediction["peak_file"]
                scan_id = prediction["scan_id"]
                charge = prediction["precursor_charge"]
                precursor_mz = prediction["precursor_mz"]
                peptide_score = prediction["peptide_score"]
                aa_scores = prediction["aa_scores"]
                calc_mz = pl_module.peptide_mass_calculator.mass(peptide, charge)
                title = prediction["title"]

                # Convert list fields to strings
                if isinstance(peptide, list):
                    peptide = ",".join(peptide)
                if isinstance(aa_scores, (list, np.ndarray)):
                    aa_scores = ",".join(f"{score:.5f}" for score in aa_scores)

                psm_entry = {
                    "sequence": peptide,
                    "PSM_ID": None,
                    "accession": "null",
                    "unique": "null",
                    "database": "null",
                    "database_version": "null",
                    "search_engine": "[MS, MS:1001456, CustomModel, 1.0]",
                    "search_engine_score[1]": peptide_score,
                    "modifications": "null",
                    "retention_time": "null",
                    "charge": charge,
                    "exp_mass_to_charge": precursor_mz,
                    "calc_mass_to_charge": calc_mz,
                    "spectra_ref": f"{peak_file}:{scan_id}",
                    "pre": "null",
                    "post": "null",
                    "start": "null",
                    "end": "null",
                    "opt_ms_run[1]_aa_scores": aa_scores,
                    "opt_ms_run[1]_ground_truth_sequence": "null",
                    "title": title,
                }
                self.writer.write_psm(psm_entry)

            self.writer.flush()

    def _gather_outputs(self, outputs, pl_module):
        if pl_module.trainer.world_size > 1:
            gathered_predictions = pl_module.all_gather(outputs["predictions"])
            gathered_peptides_true = pl_module.all_gather(outputs["peptides_true"])
            predictions = [item for sublist in gathered_predictions for item in sublist]
            peptides_true = [
                item for sublist in gathered_peptides_true for item in sublist
            ]
            return {"predictions": predictions, "peptides_true": peptides_true}
        else:
            return outputs
