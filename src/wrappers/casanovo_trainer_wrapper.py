import collections
import heapq
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import numpy as np
import pytorch_lightning as pl

# Assuming BeamSearchInterface is in wrappers.beam_search
from wrappers.beam_search import BeamSearchInterface

# Import necessary modules for evaluation and masses
import casanovo_eval as evaluate
from models.casanovo.masses import PeptideMass


class DeNovoSpec2Pep(pl.LightningModule, BeamSearchInterface):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a PyTorch Lightning Trainer.

    Parameters
    ----------
    encoder : torch.nn.Module
        Pre-instantiated encoder model.
    decoder : torch.nn.Module
        Pre-instantiated decoder model.
    global_args : argparse.Namespace
        Global arguments.
    token_dicts : dict
        Dictionary containing token mappings and other relevant data.
    task_dict : dict
        Dictionary containing task-specific parameters.
    **kwargs : dict
        Additional keyword arguments.
    """

    def __init__(
        self,
        encoder,
        decoder,
        global_args,
        token_dicts: Dict[str, Any],
        task_dict: Dict[str, Any],
        **kwargs: Dict,
    ):
        super().__init__()

        # Assign the pre-instantiated encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # Task and token dictionaries
        self.global_args = global_args

        self.predict_mode = getattr(global_args, "predict_only", False)

        self.input_dict = token_dicts["input_dict"]
        self.output_dict = token_dicts["output_dict"]
        self.int_to_aa = {b: a for a, b in self.output_dict.items()}
        self.residues = token_dicts["residues"]
        self.tokenizer = token_dicts["tokenizer"]
        self._aa2idx = self.tokenizer.index
        self.vocab_size = len(self._aa2idx) + 1
        self.stop_token = token_dicts["output_dict"]["<EOS>"]
        self.pad_token = token_dicts["output_dict"]["X"]

        self.task_dict = task_dict

        # Settings
        self.max_length = task_dict.get("max_length", 100)
        self.precursor_mass_tol = task_dict.get("precursor_mass_tol", 50)
        self.isotope_error_range = task_dict.get("isotope_error_range", (0, 1))
        self.min_peptide_len = task_dict.get("min_peptide_len", 6)
        self.n_beams = task_dict.get("n_beams", 1)
        self.top_match = 1
        self.calculate_precision = task_dict.get("calculate_precision", True)
        # Training hparams
        self.learning_rate = task_dict.get("learning_rate", 5e-4)
        self.weight_decay = task_dict.get("weight_decay", 1e-5)
        self.train_label_smoothing = task_dict.get("train_label_smoothing", 0.01)
        self.warmup_iters = task_dict.get("warmup_iters", 100_000)
        self.cosine_schedule_period_iters = task_dict.get(
            "cosine_schedule_period_iters", 600_000
        )

        # Build the peptide mass calculator
        self.peptide_mass_calculator = PeptideMass(self.residues)

        # Loss functions
        self.softmax = torch.nn.Softmax(dim=2)
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=self.pad_token, label_smoothing=self.train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token)

        # Other necessary attributes
        self.decoder.reverse = task_dict["reverse"]
        self.TASK_NAME = "casanovo_tf"

    def _get_padding_mask(self, sequences: torch.Tensor, seq_lengths: torch.Tensor):
        """
        sequences.shape = (batch_size, max_sequence_len, dim)
        seq_lengths.shape = (batch_size, 1)
        """
        assert len(sequences.shape) == 3
        assert len(seq_lengths.shape) == 2
        all_inds = (
            torch.arange(sequences.shape[1], device=sequences.device)
            .unsqueeze(0)
            .repeat((sequences.shape[0], 1))
        ).to(seq_lengths.device)
        return all_inds >= seq_lengths

    def _parse_batch(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parse the batch and extract the required data.

        Parameters
        ----------
        batch : Dict[str, Any]
            A batch of data containing 'mz_array', 'intensity_array', 'mass', 'charge', 'mz', 'intseq', and 'peptide_lengths'.

        Returns
        -------
        spectra : torch.Tensor
            The spectra tensor of shape [batch_size, n_peaks, 2].
        precursors : torch.Tensor
            The precursors tensor of shape [batch_size, 3].
        input_tokens : torch.Tensor
            The input tokens for the decoder of shape [batch_size, seq_len].
        target_tokens : torch.Tensor
            The target tokens for loss computation of shape [batch_size, seq_len + 1].
        """
        # Extract spectra
        mz_array = batch["mz_array"].to(self.device)  # Shape: [batch_size, n_peaks]
        intensity_array = batch["intensity_array"].to(
            self.device
        )  # Shape: [batch_size, n_peaks]
        spectra = torch.stack(
            [mz_array, intensity_array], dim=-1
        )  # Shape: [batch_size, n_peaks, 2]

        spectrum_padding_mask = self._get_padding_mask(spectra, batch["peak_lengths"])

        # Extract precursor information
        mass = batch["precursor_mass"].to(self.device)  # Shape: [batch_size]
        charge = batch["precursor_charge"].to(self.device)  # Shape: [batch_size]
        charge = torch.clamp(charge, 1, 10)
        mz = batch["precursor_mz"].to(self.device)  # Shape: [batch_size]
        precursors = torch.stack([mass, charge, mz], dim=1)  # Shape: [batch_size, 3]

        if self.predict_mode:
            return {
                "spectra": spectra,
                "precursors": precursors,
                "spectrum_padding_mask": spectrum_padding_mask,
            }, None

        # Extract tokens and peptide lengths
        tokens = batch["intseq"].to(self.device)  # Shape: [batch_size, seq_len]
        peptide_lengths = (
            batch["peptide_lengths"].squeeze(1).to(self.device)
        )  # Shape: [batch_size]

        batch_size, seq_len = tokens.shape

        # Initialize target tokens with padding tokens # TODO: Reverse, consider padding
        target_tokens = torch.full(
            (batch_size, seq_len + 1),
            fill_value=self.pad_token,
            dtype=torch.long,
            device=self.device,
        )

        # Copy the tokens into target_tokens
        target_tokens[:, :seq_len] = tokens  # Shape: [batch_size, seq_len + 1]

        # Insert the stop token at the position after the last amino acid in each sequence
        target_tokens[torch.arange(batch_size), peptide_lengths] = (
            self.stop_token
        )  # TODO: Reverse
        return {
            "spectra": spectra,
            "precursors": precursors,
            "input_tokens": tokens,
            "target_tokens": target_tokens,
            "spectrum_padding_mask": spectrum_padding_mask,
        }, batch_size

    def forward(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        spectrum_padding_mask: torch.Tensor,
        **kwargs,
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions. A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        return self.beam_search_decode(
            spectra.to(self.device),
            precursors.to(self.device),
            spectrum_padding_mask.to(self.device),
        )

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
        spectrum_padding_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        out = self.encoder(spectra, key_padding_mask=spectrum_padding_mask)
        emb = out["emb"]
        mem_masks = out["mask"]
        return self.decoder(sequences, precursors, emb, mem_masks)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        batch, _ = self._parse_batch(batch)
        spectra = batch["spectra"]
        precursors = batch["precursors"]
        input_tokens = batch["input_tokens"]
        target_tokens = batch["target_tokens"]
        spectrum_padding_mask = batch["spectrum_padding_mask"]
        pred, _ = self._forward_step(
            spectra, precursors, input_tokens, spectrum_padding_mask
        )
        pred = pred.reshape(-1, self.vocab_size + 1)
        if mode == "train":
            loss = self.celoss(pred, target_tokens.flatten())
        else:
            loss = self.val_celoss(pred, target_tokens.flatten())
        self.log(
            f"{self.TASK_NAME}_{mode}_loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=spectra.shape[0],
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], *args
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # Record the loss.
        loss = self.training_step(batch, mode="val")
        if not self.calculate_precision:
            return loss

        batch, _ = self._parse_batch(batch)
        spectra = batch["spectra"]
        precursors = batch["precursors"]
        target_tokens = batch["target_tokens"]
        spectrum_padding_mask = batch["spectrum_padding_mask"]
        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        peptides_true = self.tokenizer.detokenize(  # TODO: Reverse
            target_tokens, pad_token_idx=self.pad_token, EOS_token_idx=self.stop_token
        )
        peptides_pred = []
        for spectrum_preds in self.forward(spectra, precursors, spectrum_padding_mask):
            for _, _, pred in spectrum_preds:
                peptides_pred.append(pred)

        aa_precision, aa_recall, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true,
                peptides_pred,
                self.residues,
            )
        )
        # Log with specific naming conventions
        log_args = dict(
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=spectra.shape[0],
        )
        self.log(f"{self.TASK_NAME}_val_aa_prec", aa_precision, **log_args)
        self.log(f"{self.TASK_NAME}_val_aa_recall", aa_recall, **log_args)
        self.log(f"{self.TASK_NAME}_val_pep_prec", pep_precision, **log_args)

        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        A single test step for evaluating the model on a test batch.

        Parameters
        ----------
        batch : Dict[str, Any]
            A batch of data containing 'mz_array', 'intensity_array', 'mass', 'charge',
            'mz', 'intseq', 'peptide_lengths', and optionally 'peak_file' and 'scan_id'.
        batch_idx : int
            The index of the batch within the test set.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing 'predictions' and 'peptides_true'.
            'predictions' is a list of dictionaries with keys matching mzTab columns.
            'peptides_true' is a list of true peptide sequences.
        """
        # Parse the batch
        parsed_batch, batch_size = self._parse_batch(batch)
        spectra = parsed_batch["spectra"]
        precursors = parsed_batch["precursors"]
        spectrum_padding_mask = parsed_batch["spectrum_padding_mask"]

        predictions = []
        peptides_true = self.tokenizer.detokenize(
            parsed_batch["target_tokens"],
            pad_token_idx=self.pad_token,
            EOS_token_idx=self.stop_token,
            exclude_stop=True,
        )
        peptides_pred = []

        # Get peak_file and scan_id if present
        if "peak_file" in batch:
            peak_files = batch["peak_file"]  # List[str] of length batch_size
        else:
            peak_files = ["unknown"] * len(spectra)
        if "scan_id" in batch:
            scan_ids = batch["scan_id"]  # Tensor of shape [batch_size]
        else:
            scan_ids = [-1] * len(spectra)

        if "title" in batch:
            titles = batch["title"]
        else:
            titles = ["unknown"] * len(spectra)

        # Generate predictions
        model_outputs = self.forward(spectra, precursors, spectrum_padding_mask)

        for idx, spectrum_preds in enumerate(model_outputs):
            peak_file = peak_files[idx]
            scan_id = (
                int(scan_ids[idx].item())
                if isinstance(scan_ids[idx], torch.Tensor)
                else scan_ids[idx]
            )
            precursor_charge = float(precursors[idx, 1].cpu().numpy())
            precursor_mz = float(precursors[idx, 2].cpu().numpy())
            title = titles[idx]
            _pred_dict = {
                "peak_file": peak_file,  # Peak file name
                "scan_id": scan_id,  # Scan ID
                "precursor_charge": precursor_charge,  # Precursor Charge
                "precursor_mz": precursor_mz,  # Precursor m/z
                "title": title,
            }
            if spectrum_preds:
                # We're only considering the top match
                peptide_score, aa_scores, peptide = spectrum_preds[0]
                peptide = [aa for aa in peptide if aa != "$"]
                prediction = {
                    **_pred_dict,
                    "peptide": peptide,
                    "peptide_score": peptide_score,
                    "aa_scores": aa_scores,
                }
                predictions.append(prediction)
                peptides_pred.append(peptide)
            else:
                peptide = []
                prediction = {
                    **_pred_dict,
                    "peptide": peptide,
                    "peptide_score": "null",
                    "aa_scores": "null",
                }
                predictions.append(prediction)
                peptides_pred.append(peptide)

        aa_precision, aa_recall, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true,
                peptides_pred,
                self.residues,
            )
        )
        # Log with specific naming conventions
        log_args = dict(
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=spectra.shape[0],
        )
        self.log(f"{self.TASK_NAME}_test_aa_prec", aa_precision, **log_args)
        self.log(f"{self.TASK_NAME}_test_aa_recall", aa_recall, **log_args)
        self.log(f"{self.TASK_NAME}_test_pep_prec", pep_precision, **log_args)

        # Return predictions and peptides_true
        return {"predictions": predictions, "peptides_true": peptides_true}

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        # Parse the batch
        parsed_batch, _ = self._parse_batch(batch)
        spectra = parsed_batch["spectra"]
        precursors = parsed_batch["precursors"]
        spectrum_padding_mask = parsed_batch["spectrum_padding_mask"]

        predictions = []

        # Get peak_file and scan_id if present
        if "peak_file" in batch:
            peak_files = batch["peak_file"]  # List[str] of length batch_size
        else:
            peak_files = ["unknown"] * len(spectra)
        if "scan_id" in batch:
            scan_ids = batch["scan_id"]  # Tensor of shape [batch_size]
        else:
            scan_ids = [-1] * len(spectra)

        if "title" in batch:
            titles = batch["title"]
        else:
            titles = ["unknown"] * len(spectra)

        # Generate predictions
        model_outputs = self.forward(spectra, precursors, spectrum_padding_mask)

        for idx, spectrum_preds in enumerate(model_outputs):
            peak_file = peak_files[idx]
            scan_id = (
                int(scan_ids[idx].item())
                if isinstance(scan_ids[idx], torch.Tensor)
                else scan_ids[idx]
            )
            precursor_charge = float(precursors[idx, 1].cpu().numpy())
            precursor_mz = float(precursors[idx, 2].cpu().numpy())
            title = titles[idx]
            _pred_dict = {
                "peak_file": peak_file,  # Peak file name
                "scan_id": scan_id,  # Scan ID
                "precursor_charge": precursor_charge,  # Precursor Charge
                "precursor_mz": precursor_mz,  # Precursor m/z
                "title": title,
            }
            if spectrum_preds:
                # We're only considering the top match
                peptide_score, aa_scores, peptide = spectrum_preds[0]
                peptide = [aa for aa in peptide if aa != "$"]
                prediction = {
                    **_pred_dict,
                    "peptide": peptide,
                    "peptide_score": peptide_score,
                    "aa_scores": aa_scores,
                }
                predictions.append(prediction)
            else:
                peptide = []
                prediction = {
                    **_pred_dict,
                    "peptide": peptide,
                    "peptide_score": "null",
                    "aa_scores": "null",
                }
                predictions.append(prediction)

        # Return predictions and peptides_true
        return {"predictions": predictions, "peptides_true": None}

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_iters, self.cosine_schedule_period_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}

    def _replace_eos_with_null(self, tensor: torch.Tensor):
        tensor = tensor.clone()
        tensor[tensor == self.stop_token] = self.pad_token
        return tensor

    def to_aa_sequence(self, int_tensors: torch.Tensor | list):
        # Check if the input is a Tensor and convert it to a list
        if isinstance(int_tensors, torch.Tensor):
            int_tensors = self._replace_eos_with_null(int_tensors)
            int_tensors = int_tensors.tolist()

        def convert_sequence(seq):
            # Convert to amino acids and strip null tokens
            return [self.int_to_aa[i] for i in seq if self.int_to_aa[i] != "X"]

        # Check if the input is a list of lists or a single list
        if int_tensors and isinstance(int_tensors[0], list):
            # Convert each sequence in the list of lists into strings
            aa_sequences = [convert_sequence(seq) for seq in int_tensors]
        else:
            # Convert the single sequence into a string
            aa_sequences = convert_sequence(int_tensors)

        return aa_sequences


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (
            1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        return lr_factor
