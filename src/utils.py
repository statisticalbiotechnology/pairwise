from depthcharge.data import SpectrumDataset
import numpy as np
from torch.utils.data.dataset import random_split
from torch.utils.data.dataset import Subset
import os
from pathlib import Path
import torch
from functools import partial

from collate_functions import pad_peaks, pad_peptides

from loader_parquet import PeptideDataset, PeptideParser


def get_spectrum_dataset_splits(
    data_root_dir, splits=[0.6, 0.2, 0.2], max_peaks=300, random_seed=42, subset=0
):
    assert abs(sum(splits) - 1) < 1e-6
    lance_dir = os.path.join(data_root_dir, "indexed.lance")
    if os.path.exists(lance_dir):
        spectrum_dataset = SpectrumDataset.from_lance(lance_dir)
    else:
        mgf_files = list(Path(data_root_dir).rglob("**/*.mgf"))
        spectrum_dataset = SpectrumDataset(mgf_files, path=lance_dir)

    # Calculate sizes based on the desired split ratios
    train_size, val_size, test_size = splits

    # Use random_split to create the train, validation, and test datasets
    dataset_train, dataset_val, dataset_test = random_split(
        spectrum_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed),
    )

    if subset:
        assert subset >= 0 and subset <= 1
        dataset_train, dataset_val, dataset_test = [
            Subset(dataset, np.arange(int(len(dataset) * subset)))
            for dataset in [dataset_train, dataset_val, dataset_test]
        ]

    return (dataset_train, dataset_val, dataset_test), partial(
        pad_peaks, max_peaks=max_peaks
    )


def prepend_relative_path(root_dir, rel_path):
    return os.path.join


def get_ninespecies_dataset_splits(data_root_dir, ds_config, max_peaks=300, subset=0):
    path_dict = ds_config["paths"]

    for split in ["train", "val", "test"]:
        path_list = path_dict[split]
        for i, path in enumerate(path_list):
            path_list[i] = os.path.join(data_root_dir, path)
        path_dict[split] = path_list

    parser = PeptideParser(ds_config)
    dfs, token_dicts = parser.get_data()
    amod_dict = token_dicts["amod_dict"]
    dataset_train = PeptideDataset(dfs["train"], amod_dict)
    dataset_val = PeptideDataset(dfs["val"], amod_dict)
    dataset_test = PeptideDataset(dfs["test"], amod_dict)

    if subset:
        assert subset >= 0 and subset <= 1
        dataset_train, dataset_val, dataset_test = [
            Subset(dataset, np.arange(int(len(dataset) * subset)))
            for dataset in [dataset_train, dataset_val, dataset_test]
        ]

    return (
        (dataset_train, dataset_val, dataset_test),
        partial(pad_peptides, max_peaks=max_peaks, null_token_idx=amod_dict["X"]),
        token_dicts,
    )


def get_rank() -> int:
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_num_parameters(model):
    sum = 0
    for param in list(model.parameters()):
        sum += param.numel()
    return sum


if __name__ == "__main__":
    ### Test code
    data_dir = "/Users/alfred/Documents/Datasets/instanovo_data_subset"
    lance_dir = "/Users/alfred/Documents/Datasets/instanovo_data_subset/indexed.lance"
    mdsaved_dir = "//Users/alfred/Documents/Datasets/instanovo_data_subset/mdsaved"

    datasets = get_spectrum_dataset_splits(data_dir)
    print(next(iter(datasets[0])))
    print(len(datasets[0]))
    print(len(datasets[1]))
    print(len(datasets[2]))
