import os
import numpy as np
import lance
from tqdm import tqdm

# Define your Lance dataset directory
ROOT_DIR = "/proj/bedrock/datasets/MassIVE_KB/"
LANCE_DIR = os.path.join(ROOT_DIR, "indexed.lance")
CHUNK_SIZE=1000
ds = lance.dataset(LANCE_DIR)

# Calculate the sizes for each dataset split
total_rows = ds.count_rows()
val_size = int(0.01 * total_rows)  # 1% for validation
test_size = int(0.0025 * total_rows)  # 0.25% for testing
train_size = total_rows - val_size - test_size  # Remaining for training

# Generate and shuffle indices
indices = np.arange(total_rows)
np.random.shuffle(indices)

# Split indices
train_indices = indices[:train_size]
val_indices = indices[train_size : train_size + val_size]
test_indices = indices[train_size + val_size :]


def generate_record_batches(dataset, indices, chunk_size=1000):
    """
    Yields RecordBatches for specified indices in chunks, with a progress bar.
    """
    # Prepare tqdm to show progress over the chunks
    total_chunks = (len(indices) + chunk_size - 1) // chunk_size

    for i in tqdm(
        range(0, len(indices), chunk_size),
        total=total_chunks,
        desc="Processing batches",
    ):
        chunk_indices = indices[i : i + chunk_size]
        # Ensure the last chunk is processed even if it's smaller than chunk_size
        batches = dataset.take(chunk_indices, batch_readahead=0).to_batches()
        for batch in batches:
            yield batch


# Paths for the split datasets using os.path.join for compatibility
train_dataset_path = os.path.join(ROOT_DIR, "train.lance")
val_dataset_path = os.path.join(ROOT_DIR, "val.lance")
test_dataset_path = os.path.join(ROOT_DIR, "test.lance")

# Writing the datasets with progress bars
lance.write_dataset(
    generate_record_batches(ds, train_indices, chunk_size=CHUNK_SIZE), train_dataset_path, schema=ds.schema
)
lance.write_dataset(
    generate_record_batches(ds, val_indices, chunk_size=CHUNK_SIZE), val_dataset_path, schema=ds.schema
)
lance.write_dataset(
    generate_record_batches(ds, test_indices, chunk_size=CHUNK_SIZE), test_dataset_path, schema=ds.schema
)

print("Dataset has been split into train, validation, and test sets.")
