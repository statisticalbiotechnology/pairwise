from collections.abc import Callable
from typing import Any
import warnings
import depthcharge
import torch
import models.model_parts as mp
from models.peak_encoder import StaticPeakEncoder, PeakEncoderPW


class SpectrumTransformerEncoder(depthcharge.transformers.SpectrumTransformerEncoder):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: PeakEncoderPW | Callable[..., Any] | bool = True,
        use_charge=False,
        use_energy=False,
        use_mass=False,
        cls_token=False,
        pw_perceiver_config=None,
    ) -> None:
        super().__init__(
            d_model, nhead, dim_feedforward, n_layers, dropout, peak_encoder
        )

        if callable(peak_encoder):
            self.peak_encoder = peak_encoder
        elif peak_encoder:
            self.peak_encoder = PeakEncoderPW(d_model, pw_perceiver_config)
        else:
            self.peak_encoder = torch.nn.Identity()

        self.running_units = self.d_model
        # Already has attributes self.nhead, self.dim_feedforward, self.dropout for some reason
        self.use_charge = use_charge
        self.use_energy = use_energy
        self.use_mass = use_mass

        if cls_token:
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

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

    def encode_peaks(self, mz_int):
        return self.peak_encoder(mz_int)

    def forward(
        self,
        mz_int: torch.Tensor | None = None,
        fourier_features: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
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

        assert (
            sum(x is not None for x in [mz_int, fourier_features]) == 1
        ), "Exactly one of mz_int and fourier_features must be specified"

        if mz_int is not None:
            n_batch = mz_int.shape[0]
        elif fourier_features is not None:
            n_batch = fourier_features.shape[0]

        # Encode peaks into fourier features
        if mz_int is not None:
            peaks = self.encode_peaks(mz_int)
        elif fourier_features is not None:
            peaks = fourier_features

        precursor_latents = self.precursor_hook(
            peaks,
            charge,
            energy,
            mass,
            **kwargs,
        )

        num_cem_tokens = precursor_latents.shape[1]
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(n_batch, -1, -1)
            peaks = torch.cat([cls_tokens, precursor_latents, peaks], dim=1)
            num_cem_tokens += 1
        else:
            peaks = torch.cat([precursor_latents, peaks], dim=1)

        if key_padding_mask is not None:
            # Additional mask entries (sequence dim) due to charge/energy/mass, and cls_token
            cem_mask_pos = torch.tensor([[False] * num_cem_tokens] * n_batch).type_as(
                key_padding_mask
            )
            mask = torch.cat([cem_mask_pos, key_padding_mask], dim=1)
        else:
            mask = None

        encoder_out = self.transformer_encoder(peaks, src_key_padding_mask=mask)

        return {
            "emb": encoder_out,
            "mask": mask,
            "num_cem_tokens": num_cem_tokens,
        }

    def get_layer_id(self, param_name):
        """
        Assign a parameter with its layer id
        Following MAE: https://github.com/facebookresearch/mae/blob/main/util/lr_decay.py
        """

        if param_name.startswith("peak_encoder") or param_name.startswith("cls_token"):
            return 0
        elif param_name.startswith("transformer_encoder.layers."):
            return int(param_name.split(".")[2])
        else:
            return self.n_layers


def dc_encoder_smaller(
    use_charge=False,
    use_energy=False,
    use_mass=False,
    static_peak_encoder=False,
    pw_perceiver_config=None,
    **kwargs,
):
    d_model = 256
    if static_peak_encoder:
        peak_encoder = StaticPeakEncoder(d_model)
    else:
        peak_encoder = True
    model = SpectrumTransformerEncoder(
        d_model=d_model,
        nhead=8,
        dim_feedforward=1024,
        n_layers=9,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        peak_encoder=peak_encoder,
        dropout=0.25,
        pw_perceiver_config=pw_perceiver_config,
    )
    return model


def dc_encoder_base(
    use_charge=False,
    use_energy=False,
    use_mass=False,
    static_peak_encoder=False,
    dropout=0,
    cls_token=False,
    pw_perceiver_config=None,
    **kwargs,
):
    d_model = 512
    if static_peak_encoder:
        peak_encoder = StaticPeakEncoder(d_model)
    else:
        peak_encoder = True
    model = SpectrumTransformerEncoder(
        d_model=d_model,
        nhead=8,
        dim_feedforward=2048,
        n_layers=9,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        peak_encoder=peak_encoder,
        dropout=dropout,
        cls_token=cls_token,
        pw_perceiver_config=pw_perceiver_config,
    )
    return model


def dc_encoder_tiny(
    use_charge=False,
    use_energy=False,
    use_mass=False,
    static_peak_encoder=False,
    dropout=0,
    cls_token=False,
    pw_perceiver_config=None,
    **kwargs,
):
    d_model = 64
    if static_peak_encoder:
        peak_encoder = StaticPeakEncoder(d_model)
    else:
        peak_encoder = True
    model = SpectrumTransformerEncoder(
        d_model=d_model,
        nhead=8,
        dim_feedforward=256,
        n_layers=2,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        peak_encoder=peak_encoder,
        dropout=dropout,
        cls_token=cls_token,
        pw_perceiver_config=pw_perceiver_config,
    )
    return model


def dc_encoder_larger(
    use_charge=False,
    use_energy=False,
    use_mass=False,
    static_peak_encoder=False,
    dropout=0,
    cls_token=False,
    pw_perceiver_config=None,
    **kwargs,
):
    d_model = 1024
    if static_peak_encoder:
        peak_encoder = StaticPeakEncoder(d_model)
    else:
        peak_encoder = True
    model = SpectrumTransformerEncoder(
        d_model=d_model,
        nhead=8,
        dim_feedforward=2048,
        n_layers=9,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        peak_encoder=peak_encoder,
        dropout=dropout,
        cls_token=cls_token,
        pw_perceiver_config=pw_perceiver_config,
    )
    return model


def dc_encoder_larger_deeper(
    use_charge=False,
    use_energy=False,
    use_mass=False,
    static_peak_encoder=False,
    dropout=0,
    cls_token=False,
    pw_perceiver_config=None,
    **kwargs,
):
    d_model = 1024
    if static_peak_encoder:
        peak_encoder = StaticPeakEncoder(d_model)
    else:
        peak_encoder = True
    model = SpectrumTransformerEncoder(
        d_model=d_model,
        nhead=8,
        dim_feedforward=2048,
        n_layers=15,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        peak_encoder=peak_encoder,
        dropout=dropout,
        cls_token=cls_token,
        pw_perceiver_config=pw_perceiver_config,
    )
    return model


def dc_encoder_huge(
    use_charge=False,
    use_energy=False,
    use_mass=False,
    static_peak_encoder=False,
    dropout=0,
    cls_token=False,
    pw_perceiver_config=None,
    **kwargs,
):
    d_model = 2048
    if static_peak_encoder:
        peak_encoder = StaticPeakEncoder(d_model)
    else:
        peak_encoder = True
    model = SpectrumTransformerEncoder(
        d_model=d_model,
        nhead=8,
        dim_feedforward=2048,
        n_layers=18,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        peak_encoder=peak_encoder,
        dropout=dropout,
        cls_token=cls_token,
        pw_perceiver_config=pw_perceiver_config,
    )
    return model


def dc_casanovo_encoder(
    use_charge=False,
    use_energy=False,
    use_mass=False,
    static_peak_encoder=False,
    dropout=0,
    cls_token=False,
    pw_perceiver_config=None,
    **kwargs,
):
    dropout = 0 if dropout is None else dropout
    d_model = 512
    if static_peak_encoder:
        peak_encoder = StaticPeakEncoder(d_model)
    else:
        peak_encoder = True
    model = SpectrumTransformerEncoder(
        d_model=d_model,
        nhead=8,
        dim_feedforward=1024,
        n_layers=9,
        use_charge=use_charge,
        use_mass=use_mass,
        use_energy=use_energy,
        peak_encoder=peak_encoder,
        dropout=dropout,
        cls_token=cls_token,
        pw_perceiver_config=pw_perceiver_config,
    )
    return model
