from copy import deepcopy
import torch
from data_augmentation import RandomWindowAugmentation
from wrappers.base_wrapper import BasePLWrapper
from models.heads import ClassifierHead
import torch.nn.functional as F
import torch.nn as nn
from models.peak_encoder import StaticPeakEncoder, PosEncoder
from models.dino import DINOHead, DINOLoss, MultiCropWrapper


class TrinaryMZPLWrapper(BasePLWrapper):
    def __init__(self, encoder, global_args, collate_fn=None, task_dict=None):
        self.penult_units = global_args.trinary_penult_units
        head = ClassifierHead(3, encoder.running_units, self.penult_units)
        super().__init__(encoder, global_args, head, collate_fn, task_dict=task_dict)
        self.corrupt_freq = global_args.trinary_freq
        self.corrupt_std = global_args.trinary_std

        self.TASK_NAME = "trinary_mz"

    def _parse_batch(self, batch, Eval=False):
        spectra = batch
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        corrupt_mz_arr, target = self.inptarg(mz_arr)

        batch_size = mz_arr.shape[0]
        mzab = torch.stack([corrupt_mz_arr, int_arr], dim=-1)
        key_padding_mask = self._get_padding_mask(mzab, spectra["peak_lengths"])
        return (
            mzab,
            {
                "mass": spectra["precursor_mz"] if self.encoder.use_mass else None,
                "charge": (
                    spectra["precursor_charge"] if self.encoder.use_charge else None
                ),
                "key_padding_mask": key_padding_mask,  # if self.mask_zero_tokens else None,
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
    
    """
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
    """
    def configure_optimizers(self):
        opts = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        return opts


class MaskedTrainingPLWrapper(BasePLWrapper):
    def __init__(
        self,
        encoder,
        global_args,
        collate_fn=None,
        task_dict=None,
    ):
        self.predict_fourier = task_dict["predict_fourier"]
        self.mask_token_fourier = task_dict["mask_token_fourier"]
        self.learned_masked_token = task_dict["learned_masked_token"]
        self.mz_to_int_ratio = task_dict["mz_to_int_ratio"]
        self.positional_encoding = task_dict["positional_encoding"]

        self.mz_scale_mean = task_dict.get("mz_scale_mean", None)
        self.mz_scale_std = task_dict.get("mz_scale_std", None)

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
            global_args,
            collate_fn=collate_fn,
            head=head,
            task_dict=task_dict,
        )
        self.mask_ratio = task_dict["mask_ratio"]
        self.max_peaks = global_args.max_peaks
        self._setup_masking_parameters()
        self.TASK_NAME = "masked"

    def _setup_masking_parameters(self):
        self.target_peak_encoder = (
            StaticPeakEncoder(
                self.encoder.running_units, mz_to_int_ratio=self.mz_to_int_ratio
            )
            if self.predict_fourier
            else None
        )

        d_mask_token = self.encoder.running_units if self.mask_token_fourier else 2

        if self.learned_masked_token:
            self.mask_token = nn.Parameter(torch.randn(d_mask_token))
        else:
            self.mask_token = nn.Parameter(
                torch.zeros(d_mask_token), requires_grad=False
            )

        if self.positional_encoding:
            pos_encs = PosEncoder(d_mask_token)(
                torch.zeros((1, int(1.2 * self.max_peaks), d_mask_token))
            )
            self.register_buffer("pos_encodings", pos_encs)

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

        if self.positional_encoding:
            pos_encs_masked = self.pos_encodings[:, : input.shape[1], :]
            pos_encs_masked = pos_encs_masked.repeat((N, 1, 1))
            masked_input[~keep_mask] += pos_encs_masked[~keep_mask]

        loss_mask = ~keep_mask
        return masked_input, loss_mask

    def standard_scale_mz(self, mzab: torch.Tensor, peak_lengths: torch.Tensor):
        """
        Standard scale the mz_array_batch using precomputed mean and std for non-padding positions.
        """
        mz_array_batch = mzab[:, :, 0]
        B, L = mz_array_batch.shape

        mask = (
            torch.arange(L).unsqueeze(0).repeat((B, 1)).type_as(peak_lengths)
            < peak_lengths
        )

        valid_mz = mz_array_batch[mask]
        scaled_mz = (valid_mz - self.mz_scale_mean) / self.mz_scale_std

        scaled_mz_array_batch = mz_array_batch.clone()
        scaled_mz_array_batch[mask] = scaled_mz

        scaled_mzab = mzab.clone()
        scaled_mzab[:, :, 0] = scaled_mz_array_batch

        return scaled_mzab

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
            target = self.standard_scale_mz(mzab, peak_lengths)

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
            key_padding_mask=(
                parsed_batch["key_padding_mask"] if self.mask_zero_tokens else None
            ),
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


class MaskedAutoencoderWrapper(MaskedTrainingPLWrapper):
    def __init__(self, encoder, global_args, collate_fn=None, task_dict=None):
        self.decoder_running_units = task_dict["decoder_running_units"]
        self.decoder_nhead = task_dict["decoder_nhead"]
        self.decoder_dim_feedforward = task_dict["decoder_dim_feedforward"]
        self.decoder_dropout = task_dict["decoder_dropout"]
        self.padding_value = 0

        super().__init__(encoder, global_args, collate_fn, task_dict)
        del self.head
        del self.mask_token

        self.TASK_NAME = "masked_ae"

        output_dim = encoder.running_units if self.predict_fourier else 2

        if self.encoder.running_units != self.decoder_running_units:
            self.proj_before = nn.Linear(
                self.encoder.running_units, self.decoder_running_units
            )
        else:
            self.proj_before = None

        if self.decoder_running_units != output_dim:
            self.proj_after = nn.Linear(self.decoder_running_units, output_dim)
        else:
            self.proj_after = None

        self.decoder = torch.torch.nn.TransformerEncoderLayer(
            d_model=self.decoder_running_units,
            nhead=self.decoder_nhead,
            dim_feedforward=self.decoder_dim_feedforward,
            batch_first=True,
            dropout=self.decoder_dropout,
        )

        self._setup_masking_parameters()

    def _setup_masking_parameters(self):
        self.target_peak_encoder = (
            StaticPeakEncoder(
                self.encoder.running_units, mz_to_int_ratio=self.mz_to_int_ratio
            )
            if self.predict_fourier
            else None
        )

        d_mask_token = self.decoder_running_units

        if self.learned_masked_token:
            self.mask_token = nn.Parameter(torch.randn(d_mask_token))
        else:
            self.mask_token = nn.Parameter(
                torch.zeros(d_mask_token), requires_grad=False
            )

        if self.positional_encoding:
            pos_encs = PosEncoder(d_mask_token)(
                torch.zeros((1, int(1.2 * self.max_peaks), d_mask_token))
            )
            self.register_buffer("pos_encodings", pos_encs)

    def random_masking(self, input, seq_lengths, padding_value):
        """Constructs the masked encoder input along with masks indicating the positions of masked tokens and padding tokens.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, seq_len, dim).
                                It represents a batch of sequences with padding.
            seq_lengths (torch.Tensor): A tensor of shape (batch_size,) containing the actual lengths of sequences
                                        in the batch, excluding padding.
            padding_value (float): The value used for padding in the sequences.

        Returns:
            torch.Tensor: The encoder input tensor of shape (batch_size, max_num_keep, dim),
                        where max_num_keep is the maximum number of tokens kept across all sequences in the batch.
                        This tensor contains the observed (non-masked) tokens.
            torch.Tensor: The encoder padding mask of shape (batch_size, max_num_keep), indicating the positions of padding tokens.
            torch.Tensor: The combined mask of shape (batch_size, seq_len), indicating the positions of the masked tokens in the original input, excluding padding.
            torch.Tensor: The original padding mask of shape (batch_size, seq_len), indicating the positions of padding tokens in the original input.

        """
        N, L, D = input.shape  # batch, length, dim

        # Create a random mask for each valid position in the sequences
        max_length = torch.max(seq_lengths)
        random_values = torch.rand(N, max_length, device=input.device)

        # Determine the number of tokens to mask for each sequence, excluding padding
        num_mask = (seq_lengths * self.mask_ratio).long()
        num_keep = seq_lengths - num_mask

        encoder_input = torch.full(
            (N, num_keep.max(), D),
            padding_value,
            dtype=input.dtype,
            device=input.device,
        )

        orig_padding_mask = self._get_padding_mask(input, seq_lengths)

        masked_positions = torch.full((N, L), True, device=input.device)

        for i in range(N):
            # Masking only non-padding positions
            _, keep_indices = torch.topk(
                random_values[i, : seq_lengths[i]], k=num_keep[i], largest=True
            )
            # Sorting keep_indices to maintain original order
            # though not needed bc transformer permutation invariance
            sorted_keep_indices = keep_indices.sort()[0]
            encoder_input[i, : num_keep[i]] = input[i, sorted_keep_indices]
            masked_positions[i, sorted_keep_indices] = False

        keep_positions = ~masked_positions & ~orig_padding_mask
        masked_positions = masked_positions & ~orig_padding_mask
        encoder_padding_mask = self._get_padding_mask(encoder_input, num_keep)
        return (
            encoder_input,
            encoder_padding_mask,
            keep_positions,
            masked_positions,
            orig_padding_mask,
        )

    def _parse_batch(self, batch, Eval=False):
        mz_arr = batch["mz_array"]
        int_arr = batch["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)

        peak_lengths = batch["peak_lengths"]

        batch_size = mzab.shape[0]

        masked_items = self.random_masking(mzab, peak_lengths, self.padding_value)
        (
            encoder_input,
            encoder_padding_mask,
            keep_positions,
            masked_positions,
            orig_padding_mask,
        ) = masked_items

        if self.predict_fourier:
            target = self.target_peak_encoder(mzab)
        else:
            target = self.standard_scale_mz(mzab, peak_lengths)

        parsed_batch = {
            "encoder_input": {
                "masked_mzab": encoder_input,
                "mass": batch["precursor_mz"],
                "charge": batch["precursor_charge"],
                "key_padding_mask": encoder_padding_mask,
            },
            "misc_items": {
                "target": target,
                "keep_positions": keep_positions,
                "masked_positions": masked_positions,
                "decoder_padding_mask": orig_padding_mask,
            },
        }

        return parsed_batch, batch_size

    def _create_decoder_input(
        self,
        target: torch.Tensor,
        encoder_embeddings: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        keep_positions: torch.Tensor,
        masked_positions: torch.Tensor,
        padding_value: int,
        decoder_padding_mask: torch.Tensor,
    ):
        N, L, D_enc = target.shape  # batch, length, dim encoder
        _, _, D_dec = encoder_embeddings.shape  # _, _, dim decoder
        input = torch.zeros((N, L, D_dec), dtype=target.dtype, device=target.device)
        padding_token = torch.tensor(
            [padding_value] * D_dec, dtype=target.dtype, device=target.device
        )

        input[keep_positions] = encoder_embeddings[~encoder_padding_mask]
        input[masked_positions] = self.mask_token
        input[decoder_padding_mask] = padding_token

        if self.positional_encoding:
            pos_encs_masked = self.pos_encodings[:, :L, :]
            pos_encs_masked = pos_encs_masked.repeat((N, 1, 1))
            input[masked_positions] = pos_encs_masked[masked_positions]
        return input

    def forward(self, parsed_batch, **kwargs):
        encoder_input = parsed_batch["encoder_input"]
        outs = self.encoder(
            encoder_input["masked_mzab"],
            mass=encoder_input["mass"] if self.encoder.use_mass else None,
            charge=encoder_input["charge"] if self.encoder.use_charge else None,
            key_padding_mask=(
                encoder_input["key_padding_mask"] if self.mask_zero_tokens else None
            ),
            return_mask=True,
            **kwargs
        )
        # Remove tokens added for charge/energy/mass
        num_cem_tokens = outs["num_cem_tokens"]
        encoder_out = outs["emb"][:, num_cem_tokens:, :]

        # Project the embeddings to the decoder's running_units
        if self.proj_before:
            encoder_out = self.proj_before(encoder_out)

        misc_items = parsed_batch["misc_items"]

        decoder_input = self._create_decoder_input(
            target=misc_items["target"],
            encoder_embeddings=encoder_out,
            encoder_padding_mask=encoder_input["key_padding_mask"],
            keep_positions=misc_items["keep_positions"],
            masked_positions=misc_items["masked_positions"],
            padding_value=self.padding_value,
            decoder_padding_mask=misc_items["decoder_padding_mask"],
        )

        decoder_out = self.decoder(
            decoder_input, src_key_padding_mask=misc_items["decoder_padding_mask"]
        )

        if self.proj_after:
            preds = self.proj_after(decoder_out)
        else:
            preds = decoder_out

        return preds

    def _get_train_stats(self, returns, parsed_batch):
        loss_mask = parsed_batch["misc_items"]["masked_positions"]
        target = parsed_batch["misc_items"]["target"]
        preds = returns
        stats = {}
        loss = self._get_losses(preds, target, loss_mask)
        stats["loss"] = loss
        return loss, stats

    def _get_eval_stats(self, returns, parsed_batch):
        loss_mask = parsed_batch["misc_items"]["masked_positions"]
        target = parsed_batch["misc_items"]["target"]
        preds = returns
        stats = {}
        loss = self._get_losses(preds, target, loss_mask)
        stats["loss"] = loss
        return stats

    def configure_optimizers(self):
        opts = [
            torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                betas=(0.9, 0.9999),
                weight_decay=self.weight_decay,
            ),
        ]
        return opts


class DinoTrainingPLWrapper(BasePLWrapper):
    def __init__(
        self,
        encoder,
        global_args,
        collate_fn=None,
        task_dict=None,
    ):
        super().__init__(
            encoder, global_args, collate_fn=collate_fn, task_dict=task_dict
        )

        self.TASK_NAME = "dino"
        self.aug = RandomWindowAugmentation(
            global_crops_scale=task_dict["global_crops_scale"],
            local_crops_scale=task_dict["local_crops_scale"],
            num_global_crops=task_dict["num_global_crops"],
            num_local_crops=task_dict["num_local_crops"],
            padding_value=0,
        )

        _head_kwargs = dict(
            in_dim=encoder.running_units,
            out_dim=task_dict["mlp_out_dim"],
            use_bn=task_dict["mlp_use_bn"],
            norm_last_layer=task_dict["mlp_norm_last_layer"],
            nlayers=task_dict["mlp_nlayers"],
            hidden_dim=task_dict["mlp_hidden_dim"],
            bottleneck_dim=task_dict["mlp_bottleneck_dim"],
        )

        self.student = MultiCropWrapper(
            encoder,
            DINOHead(**_head_kwargs),
            pooling=task_dict["pooling"],
        )

        self.teacher = MultiCropWrapper(
            deepcopy(encoder),
            DINOHead(**_head_kwargs),
            pooling=task_dict["pooling"],
        )

        del self.encoder

        self.teacher.load_state_dict(self.student.state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.dino_loss = DINOLoss(
            out_dim=task_dict["mlp_out_dim"],
            num_crops_tot=task_dict["num_global_crops"] + task_dict["num_local_crops"],
            num_global_crops=task_dict["num_global_crops"],
            warmup_teacher_temp=task_dict["warmup_teacher_temp"],
            teacher_temp=task_dict["teacher_temp"],
            warmup_teacher_temp_epochs=task_dict["warmup_teacher_temp_epochs"],
            nepochs=task_dict["epochs"],
            student_temp=task_dict["student_temp"],
            center_momentum=task_dict["center_momentum"],
        )

        self.teacher_momentum = task_dict["teacher_momentum"]
        self.rand_window_size = task_dict["rand_window_size"]

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        self.update_teacher(self.teacher_momentum)
        return result

    @torch.no_grad()
    def update_teacher(self, momentum):
        for param_q, param_k in zip(
            self.student.parameters(), self.teacher.parameters()
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1 - momentum)

    def _parse_batch(self, batch, Eval=None):
        spectra = batch
        mz_arr = spectra["mz_array"]
        int_arr = spectra["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        lengths = spectra["peak_lengths"]
        crops = self.aug(mzab, lengths, self.rand_window_size)
        batch_size = mzab.shape[0]
        # TODO/FIXME: add support to include c/e/m tokens
        return crops, batch_size

    def forward(self, parsed_batch, **kwargs):
        student_out = self.student(parsed_batch)
        with torch.no_grad():
            teacher_out = self.teacher(parsed_batch[: self.aug.num_global_crops])
        return student_out, teacher_out

    def _get_losses(self, student_out, teacher_out):
        loss = self.dino_loss(
            student_out,
            teacher_out,
            epoch=self.trainer.current_epoch,  # TODO/FIXME: probably change anneal by step instead
        )
        if not torch.all(torch.isfinite(loss)):
            print("Loss is NaN")
            raise SystemExit(1)
        return loss

    def _get_train_stats(self, returns, parsed_batch):
        student_out, teacher_out = returns
        loss = self._get_losses(student_out, teacher_out)
        stats = {"loss": loss}
        return loss, stats

    def _get_eval_stats(self, returns, parsed_batch):
        student_out, teacher_out = returns
        loss = self._get_losses(student_out, teacher_out)
        stats = {"loss": loss}
        return stats

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.student.parameters(),
            betas=(0.9, 0.9999),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def get_encoder(
        self,
    ):
        """Return the encoder for downstream use"""
        return self.student.backbone
