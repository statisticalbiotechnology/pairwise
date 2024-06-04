from depthcharge.data import SpectrumDataset
import pyarrow as pa
from pathlib import Path

# Define the root directory and the destination directory
SOURCE_DATA_ROOT_DIR = "/proj/bedrock/datasets/foundational_dataset/combined/mgfs"
DEST_DATA_DIR = "/proj/bedrock/datasets/foundational_dataset/combined/full.lance"
FILE_TYPE_SUFFIX = ".mgf"

# Create a Path object for the source directory
source_dir = Path(SOURCE_DATA_ROOT_DIR)

# Find all .mgf files in the directory recursively
mgf_files = list(source_dir.rglob(f'*{FILE_TYPE_SUFFIX}'))

# Log the files found (optional but useful for verification)
print(f"Found {len(mgf_files)} files to process.")

annotated_dataset = SpectrumDataset(
    spectra=mgf_files,
    path=DEST_DATA_DIR, batch_size=1000, min_peaks=10
    # custom_fields=[pa.field("sequence", pa.string())],
)

# import lance
# lance_dataset = lance.dataset(DEST_DATA_DIR)
# # lance_generator = lance_dataset.to_batches(batch_size=1)
# lance_generator = lance_dataset.take([i for i in range(20)]).to_batches()
# for batch in lance_generator:
#     print(batch)
#     break
