from functools import partial
import pytorch_lightning as pl
from pathlib import Path
from data.lance_data_module import LanceDataModule

# from data.lance_data_module import LanceDataModule
import torch
from collate_functions import pad_peaks


# Define a simple Lightning Module for testing purposes
class TestModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Identity()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        return 0

    def validation_step(self, batch, batch_idx):
        return

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


DATA_DIR = Path("/Users/alfred/Datasets/instanovo_splits_subset")

collate_fn = partial(pad_peaks, max_peaks=100)
data_module = LanceDataModule(data_dir=DATA_DIR, batch_size=100, collate_fn=collate_fn)


# Instantiate the test model
model = TestModel()

# Create a PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=1, accelerator="cpu", devices=3, num_nodes=1, logger=False
)  # Adjust the trainer parameters as needed

# Run training
trainer.validate(model, datamodule=data_module)
