from typing import Any, Dict, Iterable, List, Union
import torch


def pad_length_collate_fn(batch, padding_value=0):
    sequences, labels = zip(*batch)

    # Pad sequences with zeros to match the length of the longest sequence
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=padding_value
    )
    return padded_sequences, torch.cat(labels)


def depthcharge_collate_fn(
    batch: Iterable[Dict[Any, Union[List, torch.Tensor]]], precision: torch.dtype=torch.float32
) -> Dict[str, Union[torch.Tensor, list[Any]]]:
    """The collate function from depthcharge-ms
        https://github.com/wfondrie/depthcharge/blob/bd2861ffe61092f3d30d96d01d2ee53309812c0a/depthcharge/data/spectrum_datasets.py.
        refactored into a function.

    Transform compatible data types into PyTorch tensors and
    pad the m/z and intensities arrays of each mass spectrum with
    zeros to be stacked into tensor.

    Parameters
    ----------
    batch : iterable of dict
        A batch of data.
    precision : torch.dtype
        Floating point precision of the returned tensors.

    Returns
    -------
    dict of str, tensor or list
        A dictionary mapping the columns of the lance dataset
        to a PyTorch tensor or list of values.
    """
    mz_array = torch.nn.utils.rnn.pad_sequence(
        [s.pop("mz_array") for s in batch],
        batch_first=True,
    )

    intensity_array = torch.nn.utils.rnn.pad_sequence(
        [s.pop("intensity_array") for s in batch],
        batch_first=True,
    )

    batch = torch.utils.data.default_collate(batch)
    batch["mz_array"] = mz_array
    batch["intensity_array"] = intensity_array

    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
            batch[key] = val.type(precision)

    return batch
