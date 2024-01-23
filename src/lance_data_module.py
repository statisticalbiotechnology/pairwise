from typing import Callable
import pytorch_lightning as pl
from pathlib import Path

import os.path as path
from lance_dataset import LanceDataset


class LanceDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int, collate_fn: Callable):
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
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = LanceDataset(
                self.lance_dirs[0],
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                with_row_id=True,
            )
            # Similarly set up validation and test datasets if needed
            self.val_dataset = LanceDataset(
                self.lance_dirs[1],
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                with_row_id=True,
            )

        if stage == "validate" or stage is None:
            # Similarly set up validation and test datasets if needed
            self.val_dataset = LanceDataset(
                self.lance_dirs[1],
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                with_row_id=True,
            )

        if stage == "test" or stage is None:
            self.test_dataset = LanceDataset(
                self.lance_dirs[2],
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                with_row_id=True,
            )

    def train_dataloader(self):
        return self.train_dataset  # or wrap it in a DataLoader if necessary

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset
