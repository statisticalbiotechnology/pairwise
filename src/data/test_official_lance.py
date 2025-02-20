#  Copyright (c) 2024. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import shutil
import tempfile

import lance
import numpy as np
import pyarrow as pa
import torch

# import pytest

# torch = pytest.importorskip("torch")
# from lance.torch.data import LanceDataset

from data.lance_helper_fns import LanceDataset as MyLanceDataset

from tqdm import tqdm  # noqa: E402


def create_dataset_and_return_path(temp_dir):
    arr = pa.array(range(1000))
    tbl = pa.Table.from_arrays([arr], ["ids"])

    # Write 10 files
    ds = lance.write_dataset(tbl, temp_dir, max_rows_per_file=100)
    assert len(ds.get_fragments()) == 10

    return ds


def simulate_rank_behavior(rank, world_size, dataset_path):
    # All ranks access the same dataset path
    ds = MyLanceDataset(
        dataset_path,
        batch_size=10,
        columns=["ids"],
        rank=rank,
        world_size=world_size,
        with_row_id=False,
        collate_fn=lambda x: x,
        shard_granularity="batch",
    )

    # Create a list to store retrieved data points
    retrieved_data = []

    for batch in tqdm(ds):
        retrieved_data.extend(batch)

    return retrieved_data


if __name__ == "__main__":
    WORLD_SIZE = 4

    all_batches = []
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # temp_dir is the path to the temporary directory
        print("Temporary directory:", temp_dir)

        lance_dataset = create_dataset_and_return_path(temp_dir)

        for rank in range(WORLD_SIZE):
            # Simulate rank 0 behavior
            cur_batches = simulate_rank_behavior(
                rank=rank, world_size=WORLD_SIZE, dataset_path=lance_dataset.uri
            )
            all_batches.append(cur_batches)

            print(
                f"{len(cur_batches)} batches retrieved from rank {rank}: \n{cur_batches[:5]}\n"
            )

        # Compare the retrieved data points between ranks
        min_len = min(len(datapoints) for datapoints in all_batches)
        for k in range(min_len):
            for i in range(WORLD_SIZE):
                for j in range(i + 1, WORLD_SIZE):
                    assert (
                        all_batches[i][k] != all_batches[j][k]
                    ), f"Datapoints retrieved from Rank {i} and Rank {j} are not different"
