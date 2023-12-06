import time
from pathlib import Path
from loaders.loader import LoadObj
from depthcharge.data import SpectrumDataset
from torch.utils.data import DataLoader
import os

data_dir = "/proj/bedrock/datasets/instanovo_data_subset/"
lance_dir = "/proj/bedrock/datasets/instanovo_data_subset/indexed.lance"
mdsaved_dir = "/proj/bedrock/datasets/instanovo_data_subset/mdsaved"
batch_size = 100
num_workers = 16
epochs = 1
subset = -1 #max number of batches that will be loaded each epoch

# LoadObj initialization
L = LoadObj(train_dirs=[data_dir], mdsaved_path=mdsaved_dir, preopen_files=False)

# SpectrumDataset initialization
mgf_files = list(Path(data_dir).glob("*.mgf"))

if os.path.exists(lance_dir):
    spectrum_dataset = SpectrumDataset.from_lance(lance_dir)
else:
    spectrum_dataset = SpectrumDataset(mgf_files, path=lance_dir)


spectrum_dataloader = spectrum_dataset.loader(batch_size=batch_size, shuffle=True, num_workers=num_workers)

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"{func.__name__} executed in {elapsed_time} seconds")
        return result
    return wrapper

@timeit
def loop_LoadObj_dataset(load_obj, batch_size, subset=-1, epochs=1):
    total_datapoints = len(load_obj.labels)
    for epoch in range(epochs):
        for batch_idx, i in enumerate(range(0, total_datapoints, batch_size), start=1):
            labels_str_list = load_obj.labels[i:i + batch_size]
            datapoint = load_obj.load_batch(labels_str_list)
            if subset > 0 and i > subset:
                break

@timeit
def loop_SpectrumDataset(dataloader, subset=-1, epochs=1):
    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            # Process the batch as needed for your training
            # mz, intensity, label = batch['mz'], batch['intensity'], batch['label']
            # Your training logic here
            # print(batch)
            if subset > 0 and batch_idx > subset:
                break

# Compare the two methods
print(f"Running {epochs} epoch(s) of dataloading with LoadObj. Batch size = {batch_size}")
loop_LoadObj_dataset(L, batch_size, subset, epochs)
print(f"Running {epochs} epoch(s) of dataloading with SpectrumDataset. Batch size = {batch_size}")
# loop_SpectrumDataset(spectrum_dataset, subset, epochs)
loop_SpectrumDataset(spectrum_dataloader, subset, epochs)
