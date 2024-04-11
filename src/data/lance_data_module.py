from functools import partial
from typing import Callable
import pytorch_lightning as pl
from pathlib import Path

import os.path as path
from lance.torch.data import LanceDataset
from lance.sampler import ShardedBatchSampler
from data.lance_helper_fns import _to_batch_dict


class LanceDataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: Path, batch_size: int, collate_fn: Callable, seed: int = 0
    ):
        super().__init__()

        self.data_dirs = [
            path.join(data_dir, split) for split in ["train", "val", "test"]
        ]

        assert all(
            [path.exists(_path) for _path in self.data_dirs]
        ), f'Expected subdirs "train", "val", "test" in data_dir: {data_dir}'

        # Assert that each subdir has been inedexed by lance and contains a "indexed.lance" dir
        lance_dirs = [path.join(subdir, "indexed.lance") for subdir in self.data_dirs]

        assert all([path.exists(_path) for _path in lance_dirs]), (
            f'Expected subdirs "train", "val", "test" in data_dir: {data_dir} '
            "to have pre-indexed lance dirs (indexed.lance)"
        )
        self.lance_dirs = lance_dirs
        self.batch_size = batch_size
        # self.collate_fn = collate_fn
        self.to_tensor_fn = partial(_to_batch_dict, collate_fn=collate_fn)
        self.seed = seed

    def setup(self, stage=None):
        if stage == "fit" or stage is None:

            train_sampler = ShardedBatchSampler(
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                randomize=True,
                seed=self.seed,
            )
            self.train_dataset = LanceDataset(
                self.lance_dirs[0],
                batch_size=self.batch_size,
                to_tensor_fn=self.to_tensor_fn,
                sampler=train_sampler,
                with_row_id=None,
            )
            val_sampler = ShardedBatchSampler(
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                randomize=False,
                seed=self.seed,
            )
            self.val_dataset = LanceDataset(
                self.lance_dirs[1],
                batch_size=self.batch_size,
                to_tensor_fn=self.to_tensor_fn,
                sampler=val_sampler,
                with_row_id=None,
            )

        if stage == "validate" or stage is None:
            val_sampler = ShardedBatchSampler(
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                randomize=False,
                seed=self.seed,
            )
            self.val_dataset = LanceDataset(
                self.lance_dirs[1],
                batch_size=self.batch_size,
                to_tensor_fn=self.to_tensor_fn,
                sampler=val_sampler,
                with_row_id=None,
            )

        if stage == "test" or stage is None:
            test_sampler = ShardedBatchSampler(
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                randomize=False,
                seed=self.seed,
            )
            self.val_dataset = LanceDataset(
                self.lance_dirs[2],
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
