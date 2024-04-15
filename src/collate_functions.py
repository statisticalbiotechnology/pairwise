from typing import Any, Dict, Iterable, List, Tuple, Union
import torch
from data.mskb_tokenizer import MSKBTokenizer


def subsample_max_peaks(mz_tensor, int_tensor, max_peaks=300):
    """Subsample mass spectra to retain the peaks with the highest intensities.

    Parameters
    ----------
    mz_tensor : torch.Tensor
        Tensor containing m/z values.
    int_tensor : torch.Tensor
        Tensor containing intensity values.
    max_peaks : int, optional
        Maximum number of peaks to retain, by default 300.

    Returns
    -------
    torch.Tensor, torch.Tensor
        Subsampled m/z tensor and intensity tensor.
    """
    combined_tensor = torch.stack([mz_tensor, int_tensor], dim=-1)

    # Sort by intensity
    sorted_indices = combined_tensor[:, 1].argsort(descending=True)

    # Take the max_peaks highest pairs
    highest_indices = sorted_indices[:max_peaks]
    # Keep original order
    highest_indices_original_order, _ = highest_indices.sort()

    subsampled_tensor = combined_tensor[highest_indices_original_order]

    # Split back into m/z and intensity tensors
    subsampled_mz = subsampled_tensor[:, 0]
    subsampled_intensity = subsampled_tensor[:, 1]
    return subsampled_mz, subsampled_intensity


def minmax_scale(
    tensor: torch.Tensor, scale_range: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    """Scale each sequence in the tensor to the specified range along the last dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to be scaled.
    scale_range : Tuple[float, float], optional
        Range to scale the tensor to, by default (0, 1).

    Returns
    -------
    torch.Tensor
        Scaled tensor.
    """
    # Calculate min and max along the last dimension
    min_vals, _ = torch.min(tensor, dim=-1, keepdim=True)
    max_vals, _ = torch.max(tensor, dim=-1, keepdim=True)

    # Scale each sequence to the specified range
    scaled_tensor = scale_range[0] + (tensor - min_vals) * (
        scale_range[1] - scale_range[0]
    ) / (max_vals - min_vals)

    return scaled_tensor


def pad_peaks(
    batch: Iterable[Dict[Any, Union[List, torch.Tensor]]],
    precision: torch.dtype = torch.float32,
    max_peaks: int = 300,
) -> Dict[str, Union[torch.Tensor, list[Any]]]:
    """
    Transform compatible data types into PyTorch tensors and
    pad the m/z and intensities arrays of each mass spectrum with
    zeros to be stacked into a tensor.

    Parameters
    ----------
    batch : iterable of dict
        A batch of data.
    precision : torch.dtype
        Floating point precision of the returned tensors.
    max_peaks : int
        Maximum length to limit each sequence.
        Subsamples peaks corresponding to the highest intensities.

    Returns
    -------
    dict of str, tensor or list
        A dictionary mapping the columns of the lance dataset
        to a PyTorch tensor or list of values.
    """
    mz_tensors = []
    int_tensors = []
    lengths = torch.zeros((len(batch), 1), dtype=torch.int32)

    for i, b in enumerate(batch):
        mz_tensor_b = b.pop("mz_array")
        intensity_tensor_b = b.pop("intensity_array")
        if len(mz_tensor_b) > max_peaks:
            mz_tensor_b, intensity_tensor_b = subsample_max_peaks(
                mz_tensor_b, intensity_tensor_b, max_peaks
            )
        lengths[i] = len(mz_tensor_b)
        mz_tensors.append(mz_tensor_b)
        int_tensors.append(intensity_tensor_b)

    mz_array = torch.nn.utils.rnn.pad_sequence(
        mz_tensors,
        batch_first=True,
    )

    intensity_array = torch.nn.utils.rnn.pad_sequence(
        int_tensors,
        batch_first=True,
    )

    intensity_array = minmax_scale(intensity_array)

    batch = torch.utils.data.default_collate(batch)
    batch["mz_array"] = mz_array
    batch["intensity_array"] = intensity_array
    batch["peak_lengths"] = lengths

    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
            batch[key] = val.type(precision)

    return batch


def pad_peptides(
    batch: Iterable[Dict[Any, Union[List, torch.Tensor]]],
    precision: torch.dtype = torch.float32,
    max_peaks: int = 300,
    null_token_idx=22,
    tokenizer: MSKBTokenizer = None,
    label_name="sequence",
) -> Dict[str, Union[torch.Tensor, list[Any]]]:
    """
    Transform compatible data types into PyTorch tensors and
    pad the m/z and intensities arrays of each mass spectrum with
    zeros to be stacked into a tensor.

    Parameters
    ----------
    batch : iterable of dict
        A batch of data.
    precision : torch.dtype
        Floating point precision of the returned tensors.
    max_peaks : int
        Maximum length to limit each sequence.
        Subsamples peaks corresponding to the highest intensities.

    Returns
    -------
    dict of str, tensor or list
        A dictionary mapping the columns of the lance dataset
        to a PyTorch tensor or list of values.
    """
    intseqs = []
    peptide_lengths = torch.zeros((len(batch), 1), dtype=torch.int32)

    for i, b in enumerate(batch):
        if tokenizer is not None:
            seq = b.pop(label_name)
            b["intseq"] = tokenizer.tokenize(seq)
        intseq_b = b.pop("intseq")
        peptide_lengths[i] = len(intseq_b)
        intseqs.append(intseq_b)

    intseq_array = torch.nn.utils.rnn.pad_sequence(
        intseqs, batch_first=True, padding_value=null_token_idx
    )
    batch = pad_peaks(batch, precision, max_peaks)
    batch["intseq"] = intseq_array
    batch["peptide_lengths"] = peptide_lengths

    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
            batch[key] = val.type(precision)

    return batch
