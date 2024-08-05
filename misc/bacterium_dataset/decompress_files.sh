#!/bin/bash

# Define the local directory where compressed files are saved
# local_directory="/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010000/"
local_directory="/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010613/"
# temp_directory="/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010000_uncompressed/"
temp_directory="/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010613_uncompressed/"

# Ensure the temporary directory exists
mkdir -p "$temp_directory"

# Get the total number of files to process
total_files=$(find "$local_directory" -name "*.gz" | wc -l)

# Initialize counter
counter=0

# Decompress all .gz files in the local directory to the temporary directory
find "$local_directory" -name "*.gz" | while read -r file; do
    counter=$((counter + 1))
    base_name=$(basename "$file" .gz)
    pv "$file" | gunzip > "$temp_directory/$base_name"
    echo "[$counter/$total_files] Decompressed: $file to $temp_directory/$base_name"
done

echo "Files decompressed to $temp_directory"
