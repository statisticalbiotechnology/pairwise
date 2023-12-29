import time
from pathlib import Path
from depthcharge.data import SpectrumDataset
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from collate_functions import pad_peaks
import os
from tqdm import tqdm

data_dir = "/Users/alfred/Documents/Datasets/instanovo_data_subset"
lance_dir = "/Users/alfred/Documents/Datasets/instanovo_data_subset/indexed.lance"
mdsaved_dir = "//Users/alfred/Documents/Datasets/instanovo_data_subset/mdsaved"
batch_size = 100
num_workers = 4
epochs = 1
subset = -1  # max number of batches that will be loaded each epoch

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

mnist_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
mnist_dataloader = DataLoader(
    mnist_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    multiprocessing_context="fork" if num_workers > 0 else None,
)


# SpectrumDataset initialization
mgf_files = list(Path(data_dir).glob("*.mgf"))

if os.path.exists(lance_dir):
    spectrum_dataset = SpectrumDataset.from_lance(lance_dir)
else:
    spectrum_dataset = SpectrumDataset(mgf_files, path=lance_dir)


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
def loop_SpectrumDataset(dataloader, subset=-1, epochs=1):
    for epoch in range(epochs):
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Process the batch as needed for your training
            # mz, intensity, label = batch['mz'], batch['intensity'], batch['label']
            # Your training logic here
            # print(f"batch of mzs shape = {batch['mz_array'].shape}")
            if subset > 0 and batch_idx > subset:
                break


@timeit
def loop_MNIST_dataloader(dataloader, subset=-1, epochs=1):
    for epoch in range(epochs):
        for batch_idx, (data, target) in tqdm(
            enumerate(dataloader), total=len(dataloader)
        ):
            # Process the batch as needed for your training
            # Your training logic here
            if subset > 0 and batch_idx > subset:
                break


if __name__ == "__main__":
    # Compare the two methods
    # print(
    #     f"Running {epochs} epoch(s) of dataloading with LoadObj. Batch size = {batch_size}"
    # )
    # loop_LoadObj_dataset(L, batch_size, subset, epochs)

    print("Running MNIST dataloading")
    loop_MNIST_dataloader(mnist_dataloader, subset=subset, epochs=epochs)

    print(
        f"Running {epochs} epoch(s) of dataloading with SpectrumDataset. Batch size = {batch_size}"
    )
    loop_SpectrumDataset(spectrum_dataloader, subset, epochs)
