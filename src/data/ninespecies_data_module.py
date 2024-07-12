from typing import Callable
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class NinespeciesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: list[Dataset],
        batch_size: int,
        num_workers: int = 0,
        collate_fn: Callable | None = None,
        pin_mem: bool = False,
        seed: int = 0,
        shuffle=True,
    ):
        super().__init__()

        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_mem = pin_mem
        self.seed = seed
        self.shuffle = shuffle

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.datasets[0],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #pin_memory=self.pin_mem,
            #drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            #shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets[1],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #pin_memory=self.pin_mem,
            drop_last=True,
            collate_fn=self.collate_fn,
            #persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets[2],
            batch_size=self.batch_size,
            #num_workers=self.num_workers,
            #pin_memory=self.pin_mem,
            drop_last=False,
            collate_fn=self.collate_fn,
            #persistent_workers=self.num_workers > 0,
        )
