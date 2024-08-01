"""
IPC Batch Processor

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
import pyarrow as pa
import pyarrow.ipc as ipc
import os
from tqdm import tqdm
from collate_functions import (
    pad_peaks,
)


# Function to perform model forward pass
def model_forward(model, collated_batch):
    embedding = model(collated_batch)
    return embedding


# Function to collate and preprocess batches
def collate_and_preprocess(mini_batch, column_names, max_peaks=300):
    batch_dict = []
    for j in range(mini_batch.num_rows):
        mz_array = mini_batch.column(column_names["mz"]).to_pylist()[j]
        intensity_array = mini_batch.column(column_names["intensity"]).to_pylist()[j]
        charge = mini_batch.column(column_names["charge"]).to_pylist()[j]
        mass = mini_batch.column(column_names["mass"]).to_pylist()[j]

        batch_dict.append(
            {
                "mz_array": torch.tensor(mz_array, dtype=torch.float32),
                "intensity_array": torch.tensor(intensity_array, dtype=torch.float32),
                "charge": charge,
                "mass": mass,
            }
        )

    collated_batch = pad_peaks(batch_dict, max_peaks=max_peaks)
    return collated_batch


# Function to process IPC files in batches and save them with additional column
def process_ipc_in_batches(
    input_path, output_dir, model, column_names, batch_size=32, max_peaks=300
):
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, output_filename)

    with pa.memory_map(input_path, "r") as source:
        reader = ipc.RecordBatchFileReader(source)
        schema = reader.schema

        # Create a new schema with the additional "spectrum_embedding" column
        new_schema = schema.append(
            pa.field("spectrum_embedding", pa.list_(pa.float32()))
        )

        with pa.OSFile(output_path, "wb") as sink:
            writer = ipc.new_file(sink, new_schema)

            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                num_rows = batch.num_rows

                description = (
                    f"{output_filename}: Record Batch {i+1}/{reader.num_record_batches}"
                )

                # Initialize progress bar for mini-batches within the current record batch
                with tqdm(total=num_rows, desc=description, unit="batch") as pbar:
                    for start_idx in range(0, num_rows, batch_size):
                        end_idx = min(start_idx + batch_size, num_rows)
                        mini_batch = batch.slice(start_idx, end_idx - start_idx)

                        # Collate and preprocess the batch
                        collated_batch = collate_and_preprocess(
                            mini_batch, column_names, max_peaks
                        )

                        # Perform model forward pass
                        embedding = model_forward(model, collated_batch)

                        # Convert the embedding to a PyArrow array
                        embedding_array = pa.array(embedding.detach().numpy().tolist())

                        # Create a new RecordBatch with the original columns and the new embedding column
                        batch_with_results = pa.RecordBatch.from_arrays(
                            mini_batch.columns + [embedding_array], schema=new_schema
                        )

                        # Write the updated mini-batch to the output file
                        writer.write_batch(batch_with_results)

                        # Update progress bar
                        pbar.update(end_idx - start_idx)

            writer.close()


if __name__ == "__main__":
    # Example Usage
    data_root_dir = "/Users/alfred/Datasets/9_species_IPC"
    output_root_dir = os.path.join(data_root_dir, "embedded")

    MODEL_BATCH_SIZE = 32
    MAX_PEAKS = 1000

    data_paths = [
        "apis_mellifera.ipc",
        "candidatus_endoloripes.ipc",
        "h_sapiens.ipc",
        "mus_musculus.ipc",
    ]

    column_names = {
        "mz": "mz_array",
        "intensity": "intensity_array",
        "charge": "precursor_charge",
        "mass": "precursor_mz",
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
        process_ipc_in_batches(
            input_path,
            output_root_dir,
            model_dummy,
            column_names,
            batch_size=MODEL_BATCH_SIZE,
            max_peaks=MAX_PEAKS,
        )
        print(
            f"Processed {filename} and saved to {os.path.join(output_root_dir, filename)}"
        )
