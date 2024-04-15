from math import floor
from typing import Callable, Union

from lance.torch.data import LanceDataset as _LanceDataset
from lance.torch.data import pa
import torch

Any = object()


class LanceDataset(_LanceDataset):
    def __len__(self):
        num_rows = floor(
            self.dataset.count_rows() / (self.batch_size * self.sampler._world_size)
        )
        return max(1, num_rows - 1)


def _tensorize(obj: Any, dtype: torch.dtype = torch.float32) -> Any:  # noqa: ANN401
    """Turn lists into tensors.

    Parameters
    ----------
    obj : any object
        If a list, attempt to make a tensor. If not or if it fails,
        return the obj unchanged.


    dtype : torch.dtype
        The datatype of the created tensors.

    Returns
    -------
    Any
        Whatever type the object is, unless its been transformed to
        a PyTorch tensor.
    """
    if not isinstance(obj, list):
        return obj

    try:
        return torch.tensor(obj, dtype=dtype)
    except ValueError:
        pass

    return obj


def _to_batch_dict(
    batch: pa.RecordBatch,
    collate_fn: Callable,
) -> Union[dict[str, torch.Tensor], torch.Tensor]:
    """Convert a pyarrow RecordBatch to torch Tensor."""
    batch
    batch_list = batch.to_pylist()
    out_list = []
    for row in batch_list:
        _dict = {}
        for key, val in row.items():
            _dict[key] = _tensorize(val)
        out_list.append(_dict)
    return collate_fn(out_list)
