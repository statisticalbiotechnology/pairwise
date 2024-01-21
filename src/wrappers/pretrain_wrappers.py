import torch
from wrappers.base_wrapper import BasePLWrapper
from models.heads import ClassifierHead
import torch.nn.functional as F
import torch.nn as nn
from models.peak_encoder import StaticPeakEncoder


class TrinaryMZPLWrapper(BasePLWrapper):
    def __init__(self, encoder, datasets, args, collate_fn=None, task_dict=None):
        self.penult_units = args.trinary_penult_units
        head = ClassifierHead(3, encoder.running_units, self.penult_units)
        super().__init__(encoder, datasets, args, head, collate_fn, task_dict=task_dict)
        self.corrupt_freq = args.trinary_freq
        self.corrupt_std = args.trinary_std

        self.TASK_NAME = "trinary_mz"

    def _parse_batch(self, batch, Eval=False):
        spectra = batch
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        corrupt_mz_arr, target = self.inptarg(mz_arr)

        key_padding_mask = self._get_padding_mask(mzab, spectra["peak_lengths"])

        batch_size = mz_arr.shape[0]
        mzab = torch.stack([corrupt_mz_arr, int_arr], dim=-1)
        return (
            mzab,
            {
                "mass": spectra["precursor_mz"] if self.use_mass else None,
                "charge": spectra["precursor_charge"] if self.use_charge else None,
                "key_padding_mask": key_padding_mask if self.mask_zero_tokens else None,
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
    def __init__(
        self,
        encoder,
        datasets,
        args,
        collate_fn=None,
        task_dict=None,
    ):
        self.predict_fourier = task_dict["predict_fourier"]
        self.mask_token_fourier = task_dict["mask_token_fourier"]
        self.learned_masked_token = task_dict["learned_masked_token"]
        self.mz_to_int_ratio = task_dict["mz_to_int_ratio"]

        if self.predict_fourier:
            head = torch.torch.nn.TransformerEncoderLayer(
                d_model=encoder.running_units,
                nhead=encoder.nhead,
                dim_feedforward=encoder.dim_feedforward,
                batch_first=True,
                dropout=encoder.dropout,
            )
        else:
            head = nn.Linear(encoder.running_units, 2)

        super().__init__(
            encoder,
            datasets,
            args,
            collate_fn=collate_fn,
            head=head,
            task_dict=task_dict,
        )

        self.target_peak_encoder = (
            StaticPeakEncoder(
                encoder.running_units, mz_to_int_ratio=self.mz_to_int_ratio
            )
            if self.predict_fourier
            else None
        )
        self.mask_ratio = args.mask_ratio

        d_mask_token = encoder.running_units if self.mask_token_fourier else 2

        if self.learned_masked_token:
            self.mask_token = nn.Parameter(torch.randn(d_mask_token))
        else:
            self.mask_token = nn.Parameter(
                torch.zeros(d_mask_token), requires_grad=False
            )

        self.TASK_NAME = "masked"

    def random_masking(self, input, seq_lengths):
        """
        Masks a random subset of deterministic size of the input tokens, excluding padding tokens.
        Padding tokens are always included in the kept tokens.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, seq_len, dim).
            seq_lengths (torch.Tensor): A 1D tensor of actual sequence lengths for each batch item.

        Returns:
            tuple: A tuple containing:
                - masked_input (torch.Tensor): The input tensor with masked tokens.
                - loss_mask (torch.Tensor): A boolean mask indicating the (predicted) positions to be considered in the loss.
                    I. e. True for masked positions, False for observed or padding positions.
        """
        N, L, D = input.shape  # batch, length, dim

        # Create a random mask for each valid position in the sequences
        max_length = torch.max(seq_lengths)
        random_values = torch.rand(N, max_length, device=input.device)

        # Determine the number of tokens to mask for each sequence, excluding padding
        num_mask = (seq_lengths * self.mask_ratio).long()

        # Create a mask for each sequence
        keep_mask = torch.ones(N, L, dtype=torch.bool, device=input.device)
        for i in range(N):
            # Masking only non-padding positions
            _, masked_indices = torch.topk(
                random_values[i, : seq_lengths[i]], k=num_mask[i], largest=True
            )
            keep_mask[i, masked_indices] = False

        # Apply the keep mask to the input
        masked_input = input.clone()
        masked_input[~keep_mask] = self.mask_token  # Add mask token (retain grad)

        loss_mask = ~keep_mask
        return masked_input, loss_mask

    def _parse_batch(self, batch, Eval=False):
        mz_arr = batch["mz_array"]
        int_arr = batch["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)

        peak_lengths = batch["peak_lengths"]
        # padding_mask = self._get_padding_mask(mzab, peak_lengths)

        batch_size = mzab.shape[0]

        if self.mask_token_fourier:
            # TODO: Not the cleanest but will have to do for now
            tokens = self.encoder.encode_peaks(mzab)
            input_tokens, loss_mask = self.random_masking(tokens, peak_lengths)
        else:
            input_mzab, loss_mask = self.random_masking(mzab, peak_lengths)
            input_tokens = self.encoder.encode_peaks(input_mzab)

        if self.predict_fourier:
            target = self.target_peak_encoder(mzab)
        else:
            target = mzab

        key_padding_mask = self._get_padding_mask(mzab, peak_lengths)

        if torch.any(loss_mask.sum(dim=1) == 0):
            bp = 0

        parsed_batch = {
            "fourier_features": input_tokens,
            "mass": batch["precursor_mz"],
            "charge": batch["precursor_charge"],
            "target": target,
            "peak_lengths": batch["peak_lengths"],
            "loss_mask": loss_mask,
            "key_padding_mask": key_padding_mask,
        }
        return parsed_batch, batch_size

    def forward(self, parsed_batch, **kwargs):
        outs = self.encoder(
            fourier_features=parsed_batch["fourier_features"],
            mass=parsed_batch["mass"] if self.encoder.use_mass else None,
            charge=parsed_batch["charge"] if self.encoder.use_charge else None,
            key_padding_mask=parsed_batch["key_padding_mask"]
            if self.mask_zero_tokens
            else None,
            return_mask=True,
            **kwargs
        )
        # Remove tokens added for charge/energy/mass
        num_cem_tokens = outs["num_cem_tokens"]
        preds = outs["emb"][:, num_cem_tokens:, :]

        if self.predict_fourier:
            mask = outs["mask"][:, num_cem_tokens:]
            preds = self.head(preds, src_key_padding_mask=mask)

        else:
            preds = self.head(preds)
        return preds

    def _get_losses(
        self, preds: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor
    ):
        loss_mask = loss_mask.int()
        loss = (preds - target) ** 2  # (N, L, D)
        loss = loss.mean(dim=-1)  # (N, L) Mean loss per token
        loss = loss * loss_mask

        # Avg loss for each batch member (N,)
        # Clamping incase of very short sequnces (which will simply have a loss signal of 0)
        loss = loss.sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
        loss = loss.mean()  # Scalar loss
        return loss

    def _get_train_stats(self, returns, parsed_batch):
        loss_mask = parsed_batch["loss_mask"]
        target = parsed_batch["target"]
        preds = returns
        stats = {}
        loss = self._get_losses(preds, target, loss_mask)
        stats["loss"] = loss
        return loss, stats

    def _get_eval_stats(self, returns, parsed_batch):
        loss_mask = parsed_batch["loss_mask"]
        target = parsed_batch["target"]
        preds = returns
        stats = {}
        loss = self._get_losses(preds, target, loss_mask)
        stats["loss"] = loss
        return stats
