"""
This script processes IPC files in batches to generate embeddings for mass spectrometry data
using a deep learning model. It reads data, processes it in mini-batches, performs a model 
forward pass, and appends the embeddings to the original data, saving the output in new IPC files.

User Specifications:
- Input Directory: Path to the IPC files.
- Output Directory: Path to save processed files.
- Batch Size: Number of samples processed per mini-batch.
- Max Peaks: Maximum number of peaks to retain per spectrum.
- Columns of Interest: Specify column names (e.g., mz, intensity, charge, mass) that must be present in the IPC files.
- Model Forward: Define the model's forward pass for generating embeddings.

Dependencies:
- PyTorch, PyArrow, TQDM
"""

import torch
import pytorch_lightning as pl
import numpy as np
import random
from data.lance_data_module import LanceDataModule
from ipc_to_embeddings import process_ipc_in_batches
from parse_args import parse_args_and_config
import os

from wrappers.pretrain_wrappers import (
    DinoTrainingPLWrapper,
    MaskedAutoencoderWrapper,
    MaskedTrainingPLWrapper,
    TrinaryMZPLWrapper,
)


import utils

import models.encoder as encoders
import models.dc_encoder as dc_encoders

ENCODER_DICT = {
    **encoders.__dict__,
    **dc_encoders.__dict__,
}


PRETRAIN_TASK_DICT = {
    "masked": MaskedTrainingPLWrapper,
    "masked_ae": MaskedAutoencoderWrapper,
    "trinary_mz": TrinaryMZPLWrapper,
    "dino": DinoTrainingPLWrapper,
}


class DINOEmbeddings(torch.nn.Module):
    def __init__(self, pl_encoder) -> None:
        super(DINOEmbeddings, self).__init__()
        self.pl_wrapper = pl_encoder
        self.encoder = pl_encoder.get_encoder()
        self.cross_attn_token = pl_encoder.student.cross_attend_token
        self.multihead_attn = pl_encoder.student.multihead_attn

    def _pool_crossattend(self, embeds, padding_masks):
        batch_size = embeds.shape[0]
        cross_attend_tokens = self.cross_attn_token.expand(batch_size, -1, -1)
        attn_output, _ = self.multihead_attn(
            query=cross_attend_tokens,
            key=embeds,
            value=embeds,
            key_padding_mask=padding_masks,
        )
        return attn_output.squeeze(1)

    @torch.no_grad()
    def forward(self, batch):
        """
        Args:
            batch (dict):
                        "mz_array": torch.Tensor of shape (batch_size, max_seq_len, embed_dim)
                        "intensity_array": torch.Tensor of shape (batch_size, max_seq_len, embed_dim)
                        "peak_lengths": torch.Tensor of shape (batch_size, 1)
        """
        mz_arr = batch["mz_array"]
        int_arr = batch["intensity_array"]
        mzab = torch.stack([mz_arr, int_arr], dim=-1)
        lengths = batch["peak_lengths"]
        padding_masks = self.pl_wrapper._get_padding_mask(mzab, lengths)

        #
        _out = self.encoder(mzab, key_padding_mask=padding_masks)
        embeds = _out["emb"]
        # These masks accomadate the new shape of embeds due to additional c/e/m or cls tokens
        out_masks = _out["mask"]
        embed = self._pool_crossattend(embeds, out_masks)  # (batch_size, embed_dim)
        return embed


def load_encoder(global_args, pretrain_config=None, ds_config=None):

    config = {
        **vars(global_args),
        "downstream_config": ds_config,
        "pretrain_config": pretrain_config,
    }

    # Define encoder model
    encoder = ENCODER_DICT[global_args.encoder_model](
        use_charge=global_args.use_charge,
        use_mass=global_args.use_mass,
        use_energy=global_args.use_energy,
        dropout=config["pretrain_config"][global_args.pretraining_task].get(
            "dropout", 0
        ),
        cls_token=global_args.cls_token,
    )

    if global_args.pretraining_task not in PRETRAIN_TASK_DICT:
        raise NotImplementedError(
            f"{global_args.pretraining_task} pretraining task not implemented"
        )

    print(f"Loaded encoder checkpoint: {global_args.encoder_weights}")
    pl_encoder = PRETRAIN_TASK_DICT[global_args.pretraining_task].load_from_checkpoint(
        global_args.encoder_weights,
        global_args=global_args,
        encoder=encoder,
        task_dict=config["pretrain_config"][global_args.pretraining_task],
    )
    print(f"Loaded.")
    embedder = DINOEmbeddings(pl_encoder)
    embedder.eval()
    return embedder


def main(
    embedder,
    data_root_dir,
    output_root_dir,
    column_names,
    batch_size=32,
    max_peaks=1000,
):
    # List all files in the specified directory
    data_paths = [f for f in os.listdir(data_root_dir) if f.endswith(".ipc")]

    if not data_paths:
        print(f"No IPC files found in {data_root_dir}")
        return

    # Process each file and write embeddings to new IPC files
    for filename in data_paths:
        input_path = os.path.join(data_root_dir, filename)
        process_ipc_in_batches(
            input_path=input_path,
            output_dir=output_root_dir,
            model=embedder,
            column_names=column_names,
            batch_size=batch_size,
            max_peaks=max_peaks,
        )
        print(
            f"Processed {filename} and saved to {os.path.join(output_root_dir, filename)}"
        )


if __name__ == "__main__":
    ### ------- STUFF FOR LOADING THE ENCODER ------
    # Parse args
    global_args, pretrain_config, ds_config = parse_args_and_config()

    # Set any specific settings like matmul precision
    if global_args.matmul_precision:
        torch.set_float32_matmul_precision(global_args.matmul_precision)

    # Set random seeds for reproducibility
    torch.manual_seed(global_args.seed)
    np.random.seed(global_args.seed)
    random.seed(global_args.seed)
    pl.seed_everything(global_args.seed)

    # Load the pre-trained encoder model wrapped for DINO embeddings
    embedder = load_encoder(global_args, pretrain_config, ds_config)
    ### --------------------------------------------

    ### ------- SET INPUT DIR CONTAINING THE IPC FILES HERE ------
    # Define directories and column names
    DATA_ROOT_DIR = "/Users/alfred/Datasets/9_species_IPC"
    OUTPUT_ROOT_DIR = os.path.join(DATA_ROOT_DIR, "embedded")

    # Try increasing and see if processing a fulle file is faster
    MODEL_BATCH_SIZE = 32
    # May result in CUDA out of memory errors if too large, in that case decrease again

    # These columns names
    column_names = {
        "mz": "mz_array",
        "intensity": "intensity_array",
        "charge": "precursor_charge",
        "mass": "precursor_mz",
    }

    # Run the main function to process IPC files and generate embeddings
    main(
        embedder=embedder,
        data_root_dir=DATA_ROOT_DIR,
        output_root_dir=OUTPUT_ROOT_DIR,
        column_names=column_names,
        batch_size=MODEL_BATCH_SIZE,
        max_peaks=global_args.max_peaks,
    )
