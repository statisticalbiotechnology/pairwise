from pathlib import Path
from sklearn.model_selection import train_test_split
import subprocess
from tqdm import tqdm
from depthcharge.data import SpectrumDataset


def split_dataset(
    source_data_dir,
    dest_data_dir,
    file_type=".mgf",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1

    # Create destination directory if it doesn't exist, else exit
    dest_path = Path(dest_data_dir)

    dest_path.mkdir(parents=True, exist_ok=False)

    # Find all files of the given type
    all_data_files = [
        f
        for f in Path(source_data_dir).rglob(f"**/*{file_type}")
        if "indexed.lance" not in str(f.resolve())
    ]

    # Split files into train and temp (val + test)
    train_files, temp_files = train_test_split(
        all_data_files, train_size=train_ratio, random_state=seed
    )

    # Split temp into val and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files, train_size=val_ratio_adjusted, random_state=seed
    )

    # Function to copy files using rsync
    def copy_files(files, dest_dir: Path):
        dest_dir.mkdir(parents=True, exist_ok=False)
        for f in tqdm(
            files, total=len(files), desc=f"Copying {dest_dir.name} files..."
        ):
            subprocess.run(["rsync", "-ah", str(f), str(dest_dir / f.name)])

    # Copy files to respective directories and return their paths
    train_dir = dest_path / "train"
    val_dir = dest_path / "val"
    test_dir = dest_path / "test"

    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

    return train_dir, val_dir, test_dir


def index_data_folder(split_dir: Path):
    lance_dir = split_dir / "indexed.lance"

    print("Indexing")
    mgf_files = list(split_dir.rglob("**/*.mgf"))

    # This call to SpectrumDataset, when the lance_dir doesn't exist, will start the indexing of the mgf_files
    SpectrumDataset(mgf_files, path=lance_dir)


# Usage
SOURCE_DATA_DIR = "/proj/bedrock/datasets/InstaNovo_dataset/foundational_model/"
DEST_DATA_DIR = "/proj/bedrock/datasets/InstaNovo_SPLITS_full/"
FILE_TYPE = ".mgf"
train_dir, val_dir, test_dir = split_dataset(
    SOURCE_DATA_DIR,
    DEST_DATA_DIR,
    FILE_TYPE,
    train_ratio=0.8,
    val_ratio=0.15,
    test_ratio=0.05,
)


for subdir in [train_dir, val_dir, test_dir]:
    print(f"Indexing {subdir.name} ...")
    index_data_folder(subdir)
