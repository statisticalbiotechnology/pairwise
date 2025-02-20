from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th
import os
import re
from copy import deepcopy
import sys
from data.lance_helper_fns import _tensorize


def map_fn(example, precision):
    for key, val in example.items():
        example[key] = _tensorize(val, dtype=precision)
    return example


class LoaderHF:
    def __init__(
        self,
        dataset_directory: str,
        val_species: str,
        precision: th.dtype = th.float32,
        **kwargs
    ):
        val_species = re.sub("-", "_", val_species).lower()
        assert val_species in [
            "apis_mellifera",
            "homo_sapiens",
            "bacillus_subtilis",
            "candidatus_endoloripes",
            "methanosarcina_mazei",
            "mus_musculus",
            "saccharomyces_cerevisiae",
            "solanum_lycopersicum",
            "vigna_mungo",
        ], "val_species not in our list"

        # Dataset
        dataset_path = {
            "train": os.path.join(dataset_directory, "*"),
            "val": os.path.join(dataset_directory, "*%s*" % val_species),
        }
        dataset = load_dataset("parquet", data_files=dataset_path, streaming=True)
        # Filter out withheld validation data from training
        dataset["train"] = dataset["train"].filter(
            lambda example: re.sub("-", "_", example["experiment_name"]).lower()
            != "%s" % val_species
        )

        # Map to format outputs
        dataset = dataset.map(
            lambda example: map_fn(example, precision),
            remove_columns=[
                "experiment_name",
                "evidence_index",
                "scan_number",
                "sequence",
                "retention_time",
                "title",
                "scans",
            ],
        )

        # Filter for length
        if "pep_length" in kwargs.keys():
            dataset = dataset.filter(
                lambda example: (example["peptide_length"] >= kwargs["pep_length"][0])
                & (example["peptide_length"] <= kwargs["pep_length"][1])
            )
        # Filter for charge
        if "charge" in kwargs.keys():
            dataset = dataset.filter(
                lambda example: (example["precursor_charge"] >= kwargs["charge"][0])
                & (example["precursor_charge"] <= kwargs["charge"][1])
            )
        # Split val into val/test
        dataset["test"] = dataset["val"]

        # Shuffle the dataset
        if "buffer_size" in kwargs.keys():
            dataset["train"] = dataset["train"].shuffle(
                buffer_size=kwargs["buffer_size"]
            )

        self.dataset = dataset

    def build_dataloader(self, dataset, batch_size, num_workers, collate_fn):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )


"""
loader = LoaderHF(
    dataset_directory='parquet/processed/',
    val_species='homo-sapiens',
    dictionary_path="ns_dictionary.txt",
    top_peaks=100,
    num_workers=16,
    pep_length=[0, 30],
    buffer_size=10000,
)
from time import time
start = time()
print("Clock started")
for i, batch in enumerate(loader.dataloader['val']):
    if i==9:
        start = time()
    if i == 1000:
        break
    print("\r%d"%i, end="")
print()
full_time = time() - start
print("Spectra per second: ", (990*100) / full_time)
"""
