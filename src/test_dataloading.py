import time
from pathlib import Path
from depthcharge.data import SpectrumDataset
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from collate_functions import pad_peaks
import os
from tqdm import tqdm
import lance

data_dir = "/Users/alfred/Documents/Datasets/instanovo_data_subset"
lance_dir = "/Users/alfred/Documents/Datasets/instanovo_data_subset/indexed.lance"
mdsaved_dir = "//Users/alfred/Documents/Datasets/instanovo_data_subset/mdsaved"
batch_size = 100
num_workers = 0
epochs = 1
subset = -1  # max number of batches that will be loaded each epoch
sleep = 0.001
batch_readahead = None
# batch_readahead = 1

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# mnist_dataset = datasets.MNIST(
#     root="./data", train=True, download=True, transform=transform
# )
# mnist_dataloader = DataLoader(
#     mnist_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=num_workers,
#     multiprocessing_context="fork" if num_workers > 0 else None,
# )


# SpectrumDataset initialization
mgf_files = list(Path(data_dir).glob("*.mgf"))

if os.path.exists(lance_dir):
    spectrum_dataset = SpectrumDataset.from_lance(lance_dir)
else:
    spectrum_dataset = SpectrumDataset(mgf_files, path=lance_dir)

lance_dataset = lance.dataset(lance_dir)
lance_generator = lance_dataset.to_batches(
    batch_size=batch_size, batch_readahead=batch_readahead
)

spectrum_dataloader = DataLoader(
    spectrum_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    multiprocessing_context="fork" if num_workers > 0 else None,
    collate_fn=pad_peaks,
)


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time} seconds")
        return result

    return wrapper


@timeit
def loop_SpectrumDataset(dataloader, subset=-1, epochs=1, sleep=0):
    for epoch in range(epochs):
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Process the batch as needed for your training
            # mz, intensity, label = batch['mz'], batch['intensity'], batch['label']
            # Your training logic here
            # print(f"batch of mzs shape = {batch['mz_array'].shape}")
            if subset > 0 and batch_idx > subset:
                break
            if sleep:
                time.sleep(sleep)


@timeit
def loop_MNIST_dataloader(dataloader, subset=-1, epochs=1, sleep=0):
    for epoch in range(epochs):
        for batch_idx, (data, target) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            # Process the batch as needed for your training
            # Your training logic here
            if subset > 0 and batch_idx > subset:
                break
            if sleep:
                time.sleep(sleep)


@timeit
def iterative_read_test(lance_generator, subset=-1, epochs=1, sleep=0):
    for epoch in range(epochs):
        for batch_idx, batch in tqdm(enumerate(lance_generator)):
            # Replace with your compute logic
            bp = 0
            mz_list = batch["mz_array"].to_pylist()
            int_list = batch["intensity_array"].to_pylist()
            mz_tensors = []
            int_tensors = []
            for i in range(len(mz_list)):
                mz_tensors.append(torch.tensor(mz_list[i], dtype=torch.float32))
                int_tensors.append(torch.tensor(int_list[i], dtype=torch.float32))
            mz_array = torch.nn.utils.rnn.pad_sequence(
                mz_tensors,
                batch_first=True,
            )
            int_array = torch.nn.utils.rnn.pad_sequence(
                int_tensors,
                batch_first=True,
            )
            if subset > 0 and batch_idx > subset:
                break
            if sleep:
                time.sleep(sleep)


if __name__ == "__main__":
    # Compare the two methods
    # print(
    #     f"Running {epochs} epoch(s) of dataloading with LoadObj. Batch size = {batch_size}"
    # )
    # loop_LoadObj_dataset(L, batch_size, subset, epochs)

    # print("Running MNIST dataloading")
    # loop_MNIST_dataloader(mnist_dataloader, subset=subset, epochs=epochs)

    print(
        f"Running {epochs} epoch(s) of dataloading with SpectrumDataset. Batch size = {batch_size}. Time.sleep({sleep})"
    )
    loop_SpectrumDataset(spectrum_dataloader, subset, epochs, sleep=sleep)
    print(
        f"Testing {epochs} epoch(s) of Iterative Read with Lance. Batch size = {batch_size}. batch_readahead={batch_readahead}. Time.sleep({sleep})"
    )
    iterative_read_test(lance_generator, subset, epochs, sleep=sleep)
