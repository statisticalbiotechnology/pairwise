from depthcharge.data import SpectrumDataset
import lance
import pyarrow as pa

# Usage
SOURCE_DATA_FILE = "/Users/alfred/Datasets/MassIVE_KB/0d.mgf"
DEST_DATA_DIR = "/Users/alfred/Datasets/MassIVE_KB/indexed.lance"
FILE_TYPE = ".mgf"

annotated_dataset = SpectrumDataset(
    spectra=SOURCE_DATA_FILE,
    path=DEST_DATA_DIR,
    custom_fields=[pa.field("sequence", pa.string())],
)

lance_dataset = lance.dataset(DEST_DATA_DIR)
# lance_generator = lance_dataset.to_batches(batch_size=1)
lance_generator = lance_dataset.take([i for i in range(20)]).to_batches()
for batch in lance_generator:
    print(batch)
    break
