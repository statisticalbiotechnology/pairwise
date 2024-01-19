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


def get_ninespecies_dataset_splits(
    data_root_dir, ds_config, max_peaks=300, subset=0, include_hidden=False
):
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


def partition_seq(seq, collect_mods=False):
    Seq = []
    if collect_mods:
        mods = []
    p = 0
    while p < len(seq):
        aa = seq[p]
        if aa == "(":
            let = seq[p - 1]
            end = seq[p:].find(")")
            mod = seq[p + 1 : p + end]
            if collect_mods:
                mods.append(mod)
            p += end
            aa = "%c_%s" % (let, mod)
            Seq[-1] = aa
        else:
            Seq.append(aa)
        p += 1
    output = {"seq": Seq}
    if collect_mods:
        output["mods"] = mods

    return output


class Scale:
    masses = {
        "A": 71.037113805,
        "R": 156.101111050,
        "N": 114.042927470,
        "D": 115.026943065,
        "C": 103.009184505,
        "Q": 128.058577540,
        "E": 129.042593135,
        "G": 57.021463735,
        "H": 137.058911875,
        "I": 113.084064015,
        "L": 113.084064015,
        "K": 128.094963050,
        "M": 131.040484645,
        "F": 147.068413945,
        "P": 97.052763875,
        "S": 87.032028435,
        "T": 101.047678505,
        "W": 186.079312980,
        "Y": 163.063328575,
        "V": 99.068413945,
    }

    def __init__(self, amod_dict):
        self.amod_dict = amod_dict
        int2mass = np.zeros((len(amod_dict)))
        for aa, integer in amod_dict.items():
            if len(aa.split("_")) == 2:
                aa, modwt = aa.split("_")
                int2mass[integer] = self.masses[aa] + eval(modwt)
            else:
                if aa in self.masses.keys():
                    int2mass[integer] = self.masses[aa]
                else:
                    int2mass[integer] = 0

        self.tok2mass = {key: int2mass[amod_dict[key]] for key in amod_dict.keys()}
        self.mp = torch.tensor(int2mass, dtype=torch.float32)
        """self.mp = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                list(int2mass.keys()), list(int2mass.values()),
                key_dtype=tf.int64, value_dtype=tf.float32
            ), num_oov_buckets=1
        )"""

    def intseq2mass(self, intseq):
        return torch.gather(self.mp, 0, intseq).sum(1)

    def modseq2mass(self, modified_sequence):
        return np.sum(
            self.tok2mass[tok] for tok in partition_seq(modified_sequence)["seq"]
        )


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
