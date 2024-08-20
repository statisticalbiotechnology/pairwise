import torch
import os
import numpy as np
import random
from pyteomics import mgf
from tqdm import tqdm

from collate_functions import pad_peaks


# Function to perform model forward pass
def model_forward(model, collated_batch):
    embedding = model(collated_batch)
    return embedding


# Function to collate and preprocess batches
def collate_and_preprocess(spectra, column_names, max_peaks=300):
    batch_dict = []
    for spectrum in spectra:
        mz_array = torch.tensor(spectrum[column_names["mz"]], dtype=torch.float32)
        intensity_array = torch.tensor(
            spectrum[column_names["intensity"]], dtype=torch.float32
        )
        # Access charge from the nested 'params' dictionary
        charge = spectrum["params"].get(column_names["charge"], None)
        charge = torch.tensor(charge, dtype=torch.float32)
        mass = spectrum["params"].get(column_names["mass"], None)
        if isinstance(mass, tuple):
            mass = mass[0]  # Take the first element if it's a tuple
        mass = torch.tensor(mass, dtype=torch.float32)

        batch_dict.append(
            {
                "mz_array": mz_array,
                "intensity_array": intensity_array,
                "charge": charge,
                "mass": mass,
            }
        )

    collated_batch = pad_peaks(batch_dict, max_peaks=max_peaks)
    return collated_batch


# Function to process MGF files in batches and save them with additional column
def process_mgf_in_batches(
    input_path, output_dir, model, column_names, batch_size=32, max_peaks=300
):
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(input_path).replace(".mgf", "_embedded.mgf")
    output_path = os.path.join(output_dir, output_filename)

    spectra = []
    with mgf.read(input_path) as reader:
        for spectrum in reader:
            spectra.append(spectrum)

    num_spectra = len(spectra)
    with tqdm(
        total=num_spectra, desc=f"Processing {output_filename}", unit="spectrum"
    ) as pbar:
        for start_idx in range(0, num_spectra, batch_size):
            end_idx = min(start_idx + batch_size, num_spectra)
            mini_batch = spectra[start_idx:end_idx]

            # Collate and preprocess the batch
            collated_batch = collate_and_preprocess(mini_batch, column_names, max_peaks)

            # Perform model forward pass
            embedding = model_forward(model, collated_batch)

            # Attach embeddings to spectra and save them
            for i, spectrum in enumerate(mini_batch):
                spectrum["spectrum_embedding"] = embedding[i].detach().numpy().tolist()

            # Write the processed spectra to the output file
            with open(output_path, "a") as output_file:
                mgf.write(mini_batch, output_file)

            pbar.update(end_idx - start_idx)


if __name__ == "__main__":
    # Example Usage
    data_root_dir = "/Users/alfred/Datasets/9_species_MGF"
    output_root_dir = os.path.join(data_root_dir, "embedded")

    MODEL_BATCH_SIZE = 32
    MAX_PEAKS = 1000

    data_paths = [
        "validation-00000-of-00001-b84568f5bf3ba95d.mgf",
    ]

    column_names = {
        "mz": "m/z array",
        "intensity": "intensity array",
        "charge": "charge",
        "mass": "pepmass",
    }

    # Dummy model for generating embeddings
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super(DummyModel, self).__init__()
            self.fc = torch.nn.Linear(10, 128)

        def forward(self, batch):
            batch_size = len(batch["mz_array"])
            embedding = torch.rand((batch_size, 128))
            return embedding

    model_dummy = DummyModel()

    for filename in data_paths:
        input_path = os.path.join(data_root_dir, filename)
        process_mgf_in_batches(
            input_path,
            output_root_dir,
            model_dummy,
            column_names,
            batch_size=MODEL_BATCH_SIZE,
            max_peaks=MAX_PEAKS,
        )
        print(
            f"Processed {filename} and saved to {os.path.join(output_root_dir, filename.replace('.mgf', '_embedded.mgf'))}"
        )
