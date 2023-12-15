from typing import Any, Dict, Iterable, List, Union
import torch




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

def pad_peaks(
    batch: Iterable[Dict[Any, Union[List, torch.Tensor]]], precision: torch.dtype=torch.float32, max_peaks: int = 300
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

    for b in batch:
        mz_tensor_b = b.pop("mz_array")
        intensity_tensor_b = b.pop("intensity_array")
        if len(mz_tensor_b) > max_peaks:
            mz_tensor_b, intensity_tensor_b = subsample_max_peaks(mz_tensor_b, intensity_tensor_b, max_peaks)
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

    batch = torch.utils.data.default_collate(batch)
    batch["mz_array"] = mz_array
    batch["intensity_array"] = intensity_array

    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and torch.is_floating_point(val):
            batch[key] = val.type(precision)

    return batch
