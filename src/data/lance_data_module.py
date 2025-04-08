from functools import partial
from typing import Callable
import pytorch_lightning as pl
from pathlib import Path

import os.path as path
from lance.sampler import ShardedBatchSampler
from data.lance_helper_fns import LanceDataset, _to_batch_dict


class LanceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        collate_fn: Callable,
        seed: int = 0,
        include_test=True,
    ):
        super().__init__()

        _splits = ["train", "val", "test"] if include_test else ["train", "val"]
        subdirs = [path.join(data_dir, split) for split in _splits]

        subdirs_exist = all(
            [path.exists(path.join(_path, "indexed.lance")) for _path in subdirs]
        )
        lance_subsets_exist = all([path.exists(_path + ".lance") for _path in subdirs])

        assert (
            subdirs_exist or lance_subsets_exist
        ), f'Expected subdirs "train", "val", "test" (or train.lance ... etc) in data_dir: {data_dir}'

        if subdirs_exist:
            lance_paths = [path.join(subdir, "indexed.lance") for subdir in subdirs]
        elif lance_subsets_exist:
            lance_paths = [subdir + ".lance" for subdir in subdirs]

        self.lance_paths = lance_paths
        self.batch_size = batch_size
        # self.collate_fn = collate_fn
        self.to_tensor_fn = partial(_to_batch_dict, collate_fn=collate_fn)
        self.seed = seed

    def setup(self, stage=None):
        train_sampler = ShardedBatchSampler(
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            randomize=True,
            seed=self.seed,
        )
        test_sampler = ShardedBatchSampler(
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            randomize=False,
            seed=self.seed,
        )
        if stage == "fit" or stage is None:

            self.train_dataset = LanceDataset(
                self.lance_paths[0],
                batch_size=self.batch_size,
                to_tensor_fn=self.to_tensor_fn,
                sampler=train_sampler,
                with_row_id=None,
            )

            self.val_dataset = LanceDataset(
                self.lance_paths[1],
                batch_size=self.batch_size,
                to_tensor_fn=self.to_tensor_fn,
                sampler=test_sampler,
                with_row_id=None,
            )

        if stage == "validate" or stage is None:
            self.val_dataset = LanceDataset(
                self.lance_paths[1],
                batch_size=self.batch_size,
                to_tensor_fn=self.to_tensor_fn,
                sampler=test_sampler,
                with_row_id=None,
            )

        if stage == "test" or stage is None:
            self.test_dataset = LanceDataset(
                self.lance_paths[2],
                batch_size=self.batch_size,
                to_tensor_fn=self.to_tensor_fn,
                sampler=test_sampler,
                with_row_id=None,
            )

    def train_dataloader(self):
        return self.train_dataset  # or wrap it in a DataLoader if necessary

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset


class BenchmarkDataModule(pl.LightningDataModule):
    def __init__(
        self,
        test_dataset: Path,
        batch_size: int,
        collate_fn: Callable,
        seed: int = 0,
    ):
        super().__init__()

        self.data_path = test_dataset
        self.batch_size = batch_size
        self.to_tensor_fn = partial(_to_batch_dict, collate_fn=collate_fn)
        self.seed = seed

    def setup(self, stage=None):
        test_sampler = ShardedBatchSampler(
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            randomize=False,
            seed=self.seed,
        )

        if stage == "test" or stage == "predict" or stage is None:
            self.test_dataset = LanceDataset(
                self.data_path,
                batch_size=self.batch_size,
                to_tensor_fn=self.to_tensor_fn,
                sampler=test_sampler,
                with_row_id=None,
            )

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return self.test_dataset

    def predict_dataloader(self):
        return self.test_dataset
