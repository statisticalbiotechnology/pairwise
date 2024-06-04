from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from depthcharge.data import SpectrumDataset

def split_dataset(
    source_data_dir,
    file_type=".mgf",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1

    # Find all files of the given type
    all_data_files = [
        f
        for f in Path(source_data_dir).rglob(f"**/*{file_type}")
        if "indexed.lance" not in str(f.resolve())
    ]

    # Split files into train, val, and test
    train_files, temp_files = train_test_split(
        all_data_files, train_size=train_ratio, random_state=seed
    )
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files, train_size=val_ratio_adjusted, random_state=seed
    )

    return train_files, val_files, test_files

def index_data(files, index_dir: Path):
    print(f"Indexing into {index_dir} ...")
    # This call to SpectrumDataset, when the index_dir doesn't exist, will start the indexing of the files
    SpectrumDataset(files, path=index_dir)

# Usage
SOURCE_DATA_DIR = "/proj/bedrock/datasets/foundational_dataset/msconvert"
DEST_DATA_DIR = Path("/proj/bedrock/datasets/foundational_dataset/msconvert")
# SOURCE_DATA_DIR = "/proj/bedrock/datasets/foundational_dataset/combined/mgfs"
# DEST_DATA_DIR = Path("/proj/bedrock/datasets/foundational_dataset/combined/")

print("Counting files ...")
train_files, val_files, test_files = split_dataset(
    SOURCE_DATA_DIR,
    ".mgf",
    train_ratio=0.9875,
    val_ratio=0.01,
    test_ratio=0.0025,
)
print("#Train files: ", len(train_files))
print("#Val files: ", len(val_files))
print("#Test files: ", len(test_files))

print("Starting the indexing ...")
# # Index each split directly into named .lance directories in the destination
index_data(train_files, DEST_DATA_DIR / "train.lance")
index_data(val_files, DEST_DATA_DIR / "val.lance")
index_data(test_files, DEST_DATA_DIR / "test.lance")
