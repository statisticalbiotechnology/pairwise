from math import ceil
import torch
import torch.nn as nn
from depthcharge.encoders.sinusoidal import FloatEncoder as DcFloatEncoder
from depthcharge.encoders.sinusoidal import PositionalEncoder as DcPosEncoder
from models.casanovo.encoders import PeakEncoder


class StaticPeakEncoder(torch.nn.Module):
    """Encode mass spectrum.

    Parameters
    ----------
    d_model : int
        The number of features to output.
    min_mz_wavelength : float, optional
        The minimum wavelength to use for m/z.
    max_mz_wavelength : float, optional
        The maximum wavelength to use for m/z.
    min_intensity_wavelength : float, optional
        The minimum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    max_intensity_wavelength : float, optional
        The maximum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    mz_to_int_ratio: int, optional
        1 means only consider mz, 0 only intensity
    """

    def __init__(
        self,
        d_model: int,
        min_mz_wavelength: float = 0.001,
        max_mz_wavelength: float = 10000,
        min_intensity_wavelength: float = 1e-6,
        max_intensity_wavelength: float = 1,
        learnable_wavelengths: bool = False,
        mz_to_int_ratio: float = 2 / 3,
    ) -> None:
        """Initialize the MzEncoder."""
        super().__init__()
        self.d_model = d_model
        self.learnable_wavelengths = learnable_wavelengths

        d_mz = ceil(mz_to_int_ratio * d_model)
        d_int = d_model - d_mz

        self.mz_encoder = DcFloatEncoder(
            d_model=d_mz,
            min_wavelength=min_mz_wavelength,
            max_wavelength=max_mz_wavelength,
            learnable_wavelengths=False,
        )

        self.int_encoder = DcFloatEncoder(
            d_model=d_int,
            min_wavelength=min_intensity_wavelength,
            max_wavelength=max_intensity_wavelength,
            learnable_wavelengths=False,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode m/z values and intensities.

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak. These
            should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectra, n_peaks, d_model)
            The encoded features for the mass spectra.
        """
        encoded = torch.cat(
            [
                self.mz_encoder(X[:, :, 0]),
                self.int_encoder(X[:, :, 1]),
            ],
            dim=2,
        )

        return encoded


class PeakEncoderPW(PeakEncoder):
    """Encode mass spectrum, m/z, intensity, and pairwise perceiver features.

    This class inherits from the PeakEncoder and concatenates the output from
    the PairwisePerceiverFeatures module if provided.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    pw_perceiver_config : dict
        Configuration for PairwisePerceiverFeatures to process pairwise mass differences.
    min_wavelength : float, optional
        The minimum wavelength to use for m/z.
    max_wavelength : float, optional
        The maximum wavelength to use for m/z.
    learned_intensity_encoding : bool, optional
        Use a learned intensity encoding for intensity values.
    """

    def __init__(
        self,
        dim_model: int,
        pw_perceiver_config: dict = None,  # Config for pairwise perceiver features
        min_wavelength: float = 0.001,
        max_wavelength: float = 10000,
        learned_intensity_encoding: bool = True,
    ) -> None:
        """Initialize the PeakEncoderPW."""
        super().__init__(
            dim_model=dim_model,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
            learned_intensity_encoding=learned_intensity_encoding,
        )

        # Initialize PairwisePerceiverFeatures if provided
        if pw_perceiver_config:
            self.pw_perceiver = PairwisePerceiverFeatures(**pw_perceiver_config)
            self.pw_perceiver_units = pw_perceiver_config["k"]
            combined_input_dim = dim_model + self.pw_perceiver_units
            self.combiner = nn.Linear(combined_input_dim, dim_model, bias=False)
        else:
            self.pw_perceiver = None
            self.combiner = (
                nn.Identity()
            )  # If no pairwise perceiver, use identity layer

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode m/z, intensity, and pairwise perceiver features.

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. Axis 0 represents a mass spectrum, axis 1
            contains the peaks in the mass spectrum, and axis 2 is essentially
            a 2-tuple specifying the m/z-intensity pair for each peak.
            These should be zero-padded, such that all spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectra, n_peaks, dim_model)
            The encoded features for the mass spectra.
        """

        # Encode m/z and intensity using Fourier or learned features from PeakEncoder
        encoded = super().forward(X)

        # Compute pairwise perceiver features and concatenate if provided
        if self.pw_perceiver:
            pw_features = self.pw_perceiver(
                X[:, :, 0].unsqueeze(-1)
            )  # (batch, seq, pw_units)
            encoded = torch.cat([encoded, pw_features], dim=2)

        # Apply combiner if using pw_perceiver, otherwise identity
        return self.combiner(encoded)
