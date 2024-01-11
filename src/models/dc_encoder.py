from collections.abc import Callable
from typing import Any
import warnings
import depthcharge
from depthcharge.encoders import PeakEncoder
import torch
import models.model_parts as mp


class SpectrumTransformerEncoder(depthcharge.transformers.SpectrumTransformerEncoder):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoder | Callable[..., Any] | bool = True,
        use_charge=False,
        use_energy=False,
        use_mass=False,
    ) -> None:
        super().__init__(
            d_model, nhead, dim_feedforward, n_layers, dropout, peak_encoder
        )
        self.running_units = self.d_model
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass

    def precursor_hook(
        self,
        mz_int,
        charge=None,
        energy=None,
        mass=None,
        **kwargs: dict,
    ) -> torch.Tensor:
        """Define how additional information in the batch may be used.

        Overwrite this method to define custom functionality dependent on
        information in the batch. Examples would be to incorporate any
        combination of the mass, charge, retention time, or
        ion mobility of a precursor ion.

        The representation returned by this method is preprended to the
        peak representations that are fed into the Transformer encoder and
        ultimately contribute to the spectrum representation that is the
        first element of the sequence in the model output.

        By default, this method returns a tensor of zeros.

        Parameters
        ----------
        mz_int : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The zero-padded (m/z, intensity) dimensions for a batch of mass spectra.
        **kwargs : dict
            The additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The precursor representations.
        """
        ce_emb = []

        # Create Fourier features for each available charge/energy/mass float
        if charge is not None:
            if not self.use_charge:
                warnings.warn(
                    "Inputting charge while this model has not been configured to use charge."
                )
            ce_emb.append(mp.FourierFeatures(charge, self.d_model, 10.0).unsqueeze(1))
        if energy is not None:
            if not self.use_energy:
                warnings.warn(
                    "Inputting energy while this model has not been configured to use energy."
                )
            ce_emb.append(mp.FourierFeatures(energy, self.d_model, 150.0).unsqueeze(1))
        if mass is not None:
            if not self.use_mass:
                warnings.warn(
                    "Inputting mass while this model has not been configured to use mass."
                )
            ce_emb.append(mp.FourierFeatures(mass, self.d_model, 20000.0).unsqueeze(1))

        # If no inputs were provided, return zero tensor
        if not ce_emb:
            return (
                torch.zeros((mz_int.shape[0], self.d_model))
                .type_as(mz_int)
                .unsqueeze(1)
            )

        # Concatenate embeddings and mz_int tensor
        ce_emb = torch.cat(ce_emb, dim=1)
        return ce_emb

    def forward(
        self,
        mz_int: torch.Tensor,
        charge: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        **kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a batch of mass spectra.

        Parameters
        ----------
        mz_int : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The zero-padded (m/z, intensity) dimensions for a batch of mass spectra.
        **kwargs : dict
            Additional fields provided by the data loader. These may be
            used by overwriting the `precursor_hook()` method in a subclass.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, d_model)
            The latent representations for the spectrum and each of its
            peaks.
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """

        n_batch = mz_int.shape[0]
        zeros = ~mz_int.sum(dim=2).bool()

        # Encode peaks into fourier features
        peaks = self.peak_encoder(mz_int)

        # Encode precursor information
        precursor_latents = self.precursor_hook(
            mz_int,
            charge,
            energy,
            mass,
            **kwargs,
        )

        peaks = torch.cat([precursor_latents, peaks], dim=1)

        # Additional mask entries (sequence dim) due to charge/energy/mass
        cem_mask_pos = torch.tensor(
            [[False] * precursor_latents.shape[1]] * n_batch
        ).type_as(zeros)
        mask = torch.cat([cem_mask_pos, zeros], dim=1)

        encoder_out = self.transformer_encoder(peaks, src_key_padding_mask=mask)
        return {"emb": encoder_out, "mask": mask}


def dc_encoder_base(use_charge=False, use_energy=False, use_mass=False, **kwargs):
    model = SpectrumTransformerEncoder(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        n_layers=9,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
    )
    return model
