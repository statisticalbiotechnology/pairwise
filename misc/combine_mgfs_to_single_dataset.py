import os
import shutil
from tqdm import tqdm

def find_unique_files(source_dirs, target_dir):
    """Finds unique .mgf files from source directories that are not in the target directory."""
    source_files = {}
    target_files = set()

    # Collect all files in the target directory
    for root, _, filenames in os.walk(target_dir):
        for filename in filenames:
            if filename.endswith('.mgf'):
                target_files.add(filename)

    # Collect unique files from source directories
    for directory in source_dirs:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if filename.endswith('.mgf') and filename not in target_files:
                    # Store the first unique instance of the file
                    if filename not in source_files:
                        source_files[filename] = os.path.join(root, filename)

    return source_files

def copy_files(files, destination):
    """Copies files to the specified destination directory, checks if the destination exists."""
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    for file in tqdm(files.values(), desc="Copying files"):
        shutil.copy(file, destination)

def main():
    dir1 = '/proj/bedrock/datasets/foundational_dataset/PretrainV2'
    dir2 = '/proj/bedrock/datasets/foundational_dataset/msconvert'
    output_dir = '/proj/bedrock/datasets/foundational_dataset/combined'

    # Find unique files that do not exist in the output directory
    unique_files = find_unique_files([dir1, dir2], output_dir)

    print(f"Number of unique .mgf files to be copied: {len(unique_files)}")

    # Optionally copy the unique files to the output directory
    copy_files(unique_files, output_dir)

if __name__ == "__main__":
    main()
