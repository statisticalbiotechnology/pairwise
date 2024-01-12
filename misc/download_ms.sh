#!/bin/bash

# Set variables for URL and data folder path
URL="https://ftp.pride.ebi.ac.uk/pride/data/archive/2021/04/PXD021013/"
OUTPUT_FOLDER="/home/alfredn/Documents/Code/download_ms_data/data_mgf"
DATA_RAW_FOLDER="/home/alfredn/Documents/Code/download_ms_data/downloaded_raw"
MAX_FILES=5  # Set the desired maximum number of files

# Avail my aliases
shopt -s expand_aliases
source ~/.bash_aliases

# List all files, without .raw extensions, from the online repository
all_files=$(
    curl -sS $URL |
    grep -o "href=\".*3x.*.raw\"" |
    cut -c 7- |
    rev |
    cut -c 6- |
    rev |
    head -n $MAX_FILES  # Limit to the top N files
)

# Initialize array to hold missing files
missing=()
counter=0

# Loop through the files and check if they are in the testmgf directory
for i in $all_files; do
    if [ -f "$DATA_RAW_FOLDER/$i.raw" ]; then
        continue
    else
        missing[counter]=$i
        counter=$((counter + 1))
    fi
done

echo !!!!!! TOTAL FILES $counter !!!!!!

# Loop through all filenames in the array
# Download the .raw file to the raw data folder with progress messages
for i in "${missing[@]}"; do
    echo "Downloading $i.raw..."
    wget --show-progress $URL/$i.raw -P $DATA_RAW_FOLDER
    echo "Downloaded $i.raw"
done

echo "Download completed!"

# Loop through all downloaded .raw files and extract them if not already extracted
for i in $all_files; do
    if [ ! -f "$OUTPUT_FOLDER/$i.mgf" ]; then
        echo "Extracting $i.raw..."
        mono /usr/bin/ThermoRawFileParser1.4.2/ThermoRawFileParser.exe -i=$DATA_RAW_FOLDER/$i.raw -o=$OUTPUT_FOLDER/ -f=0 -g -m=0
        echo "Extracted $i.raw"
    else
        echo "Skipping $i.raw, already extracted."
    fi
done

echo "Extraction completed!"