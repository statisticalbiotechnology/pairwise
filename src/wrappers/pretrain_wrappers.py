import torch
from wrappers.base_wrapper import BasePLWrapper
from models.heads import ClassifierHead
import torch.nn.functional as F
import torch.nn as nn
from depthcharge.encoders import PeakEncoder


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
                "mass": spectra["precursor_mz"] if self.use_mass else None,
                "charge": spectra["precursor_charge"] if self.use_charge else None,
                "length": spectra["peak_lengths"] if self.mask_zero_tokens else None,
            },
            target,
        ), batch_size

    def forward(self, parsed_batch, **kwargs):
        mzab, input_dict, target = parsed_batch
        outs = self.encoder(mzab, **input_dict, **kwargs)
        # Additional tokens added for charge/energy/mass
        num_cem_tokens = outs["num_cem_tokens"]
        embeds = outs["emb"][:, num_cem_tokens:, :]
        outs = self.head(embeds)
        return outs

    def _get_losses(self, returns, labels):
        loss = F.cross_entropy(
            returns.transpose(-1, -2),
            labels,
            reduction="mean",
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


class MaskedTrainingPLWrapper(BasePLWrapper):
    def __init__(self, encoder, datasets, args, collate_fn=None, fourier_level=True):
        self.fourier_level = fourier_level
        if not fourier_level:
            head = nn.Linear(encoder.running_units, 2)
        else:
            head = None

        super().__init__(encoder, datasets, args, collate_fn, head=head)

        self.mask_ratio = args.mask_ratio

        self.peak_encoder = PeakEncoder(encoder.running_units)
        self.mask_token = nn.Parameter(torch.randn(encoder.running_units))

    def _random_masking(self, input: torch.Tensor):
        N, L, D = input.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=input.device)  # noise in [0, 1]

        # TODO: make sure pad tokens are never included in the input
        # i e noise[arange>peak_lengths] = 0

        ids_shuffle = torch.argsort(noise, dim=1)
        inds_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        input_subset = torch.gather(
            input, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        return input_subset, inds_restore

    def _create_input(self, input: torch.Tensor):
        input_subset, ids_restore = self._random_masking(input)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            input_subset.shape[0], ids_restore.shape[1] + 1 - input_subset.shape[1], 1
        )

        x_ = torch.cat([input_subset, mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, input_subset.shape[2]),
        )  # unshuffle
        return x_

    def _parse_batch(self, batch):
        mz_arr = batch["mz_array"]
        int_arr = batch["intensity_array"]
        mzab = torch.stack([int_arr, int_arr], dim=-1)

        batch_size = input.shape[0]

        # input fourier features
        input = self.peak_encoder(mzab)
        target = input.clone()

        # randomly drop
        input, mask = self._random_masking(input)

        parsed_batch = {"fourier_features": input, "target_mask": mask}

        if self.fourier_level:
            pass
            # target should be fourier features

        parsed_batch = {
            **parsed_batch,
            "mass": batch["mass"],
            "charge": batch["charge"],
            "target": target,
            "peak_lengths": batch["peak_lengths"],
        }
        return parsed_batch, batch_size

    def forward(self, parsed_batch, **kwargs):
        mzab, input_dict, target = parsed_batch
        outs = self.encoder(mzab, **input_dict, **kwargs)
        # Additional tokens added for charge/energy/mass
        num_cem_tokens = outs["num_cem_tokens"]
        embeds = outs["emb"][:, num_cem_tokens:, :]
        outs = self.head(embeds)
        return outs
