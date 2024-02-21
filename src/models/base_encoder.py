"""
x: mz and ab; bs, sl, 2
fourier_features: expanded mz features, bs, sl, running_units
charge: bs,
mass: bs,
energy: bs,
length: # of peaks; bs,

"""
import torch
from abc import ABC, abstractmethod

class base_encoder(ABC):
    @abstractmethod
    def forward(
 	self,
        mz_int: torch.Tensor | None = None,
        fourier_features: torch.Tensor | None = None,
        charge: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        length: torch.Tensor | None = None,
        **kwargs: dict,
    ):
        assert (mz_int != None) and (fourier_features != None), (
		"Either mz_int or fourier_features must be defined"
	)


