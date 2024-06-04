from data.lance_helper_fns import LanceDataset, _to_batch_dict
from lance.sampler import ShardedBatchSampler
from functools import partial
from collate_functions import pad_peaks, pad_peptides
import torch
import numpy as np
from tqdm import tqdm

to_tensor_fn = partial(_to_batch_dict, collate_fn=partial(pad_peaks, max_peaks=1000))

train_dataset = LanceDataset(
    "/Users/alfred/Datasets/instanovo_splits_subset/train/indexed.lance",
    batch_size=100,
    to_tensor_fn=to_tensor_fn,
    with_row_id=None,
    sampler=ShardedBatchSampler(
        rank=0,
        world_size=1,
        randomize=False,
    ),
)

# Initialize variables for tracking statistics
global_min = float("inf")
global_max = float("-inf")
sum_mz = 0.0
sum_mz_squared = 0.0
total_count = 0

# Iterate through the dataset to update statistics
for batch in tqdm(train_dataset, total=len(train_dataset)):
    mz_array_batch = batch["mz_array"]
    peak_lengths_batch = batch["peak_lengths"]

    # Create a mask for valid peaks
    mask = (
        torch.arange(mz_array_batch.size(1))
        .unsqueeze(0)
        .repeat((mz_array_batch.shape[0], 1))
        < peak_lengths_batch
    )

    # Apply the mask to filter out padded values
    valid_peaks = mz_array_batch[mask]

    # Update global min and max
    batch_min = valid_peaks.min().item()
    batch_max = valid_peaks.max().item()
    global_min = min(global_min, batch_min)
    global_max = max(global_max, batch_max)

    # Update sums for mean and standard deviation calculation
    sum_mz += valid_peaks.sum().item()
    sum_mz_squared += (valid_peaks**2).sum().item()
    total_count += valid_peaks.numel()

# Calculate mean and standard deviation
mean_mz = sum_mz / total_count
std_mz = np.sqrt((sum_mz_squared / total_count) - (mean_mz**2))

print(f"Min: {global_min}")
print(f"Max: {global_max}")
print(f"Mean: {mean_mz}")
print(f"Standard Deviation: {std_mz}")
