import os
import gzip
import time
from pyteomics import mzml, mzid, mgf
from tqdm import tqdm

# Define the local directory where files are saved
local_directory = "/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010000_uncompressed/"
# local_directory = "/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010613_uncompressed/"
output_directory = "/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010000_annotated/"
# output_directory = "/proj/bedrock/datasets/bacteria_PXD010000__PXD010613/bacteria_PXD010613_annotated/"

# List mzML and mzID files
mzml_files = [f for f in os.listdir(local_directory) if f.endswith(".mzML")]
mzid_files = [f for f in os.listdir(local_directory) if f.endswith(".mzid")]

def read_mzml(file_path):
    spectra = {}
    with open(file_path, "rb") as f:
        with mzml.read(f) as reader:
            for spectrum in reader:
                spectra[spectrum["id"]] = {
                    "mz_array": spectrum["m/z array"],
                    "intensity_array": spectrum["intensity array"],
                    "scan_start_time": spectrum["scanList"]["scan"][0].get(
                        "scan start time", "N/A"
                    ),
                }
    return spectra

def read_mzid(file_path):
    psm_data = {}
    with open(file_path, "rb") as f:
        with mzid.read(f) as reader:
            for psm in reader:
                spectrum_id = psm["spectrumID"]
                items = psm.get("SpectrumIdentificationItem", [])
                for item in items:
                    peptide_sequence = item.get("PeptideSequence", "N/A")
                    psm_data[spectrum_id] = {
                        "PeptideSequence": peptide_sequence,
                        "ChargeState": item.get("chargeState", "N/A"),
                        "CalculatedMZ": item.get("calculatedMassToCharge", "N/A"),
                        "ExperimentalMZ": item.get("experimentalMassToCharge", "N/A"),
                        "QValue": item.get("MS-GF:QValue", "N/A"),
                    }
    return psm_data

def match_spectra_to_peptides(spectra_data, psm_data):
    matched_data = []
    for spectrum_id, spectrum in spectra_data.items():
        if spectrum_id in psm_data:
            peptide_info = psm_data[spectrum_id]
            matched_data.append((spectrum_id, spectrum, peptide_info))
    return matched_data

def process_files(mzml_files, mzid_files, local_directory):
    file_pairs = []
    for mzml_file in mzml_files:
        mzml_base = os.path.splitext(mzml_file)[0]

        # Find corresponding mzID file
        corresponding_mzid_file = None
        for mzid_file in mzid_files:
            mzid_base = os.path.splitext(mzid_file)[0].replace("_msgfplus", "")
            if mzml_base == mzid_base:
                corresponding_mzid_file = mzid_file
                break

        if corresponding_mzid_file:
            mzml_path = os.path.join(local_directory, mzml_file)
            mzid_path = os.path.join(local_directory, corresponding_mzid_file)
            file_pairs.append((mzml_path, mzid_path))
        else:
            print(f"No matching mzID file found for {mzml_file}")

    return file_pairs

def create_mgf_spectrum(spectrum_id, spectrum, peptide_info):
    params = {
        "TITLE": spectrum_id,
        "PEPMASS": peptide_info["ExperimentalMZ"],
        "CHARGE": f"{peptide_info['ChargeState']}+",
        "SEQ": peptide_info["PeptideSequence"],
        "SCANS": spectrum["scan_start_time"],
        "RTINSECONDS": spectrum["scan_start_time"],
    }

    return {
        "params": params,
        "m/z array": spectrum["mz_array"],
        "intensity array": spectrum["intensity_array"],
    }

def save_combined_data_to_mgf(matched_data, output_file):
    spectra = []
    for spectrum_id, spectrum, peptide_info in matched_data:
        spectra.append(create_mgf_spectrum(spectrum_id, spectrum, peptide_info))

    mgf.write(spectra, output_file)

# Define the output directory
os.makedirs(output_directory, exist_ok=True)

# Process the files and combine the data
file_pairs = process_files(mzml_files, mzid_files, local_directory)

# Process each file pair and save to individual MGF files
for mzml_path, mzid_path in tqdm(file_pairs, desc="Processing files"):
    start_time = time.time()
    spectra_data = read_mzml(mzml_path)
    read_mzml_time = time.time() - start_time
    print(f"Reading mzML took {read_mzml_time:.2f} seconds")

    start_time = time.time()
    psm_data = read_mzid(mzid_path)
    read_mzid_time = time.time() - start_time
    print(f"Reading mzID took {read_mzid_time:.2f} seconds")

    start_time = time.time()
    matched_data = match_spectra_to_peptides(spectra_data, psm_data)
    matching_time = time.time() - start_time
    print(f"Matching spectra to peptides took {matching_time:.2f} seconds")

    # Define the output file path based on the input file name
    output_file_name = os.path.basename(mzml_path).replace(".mzML", ".mgf")
    output_file_path = os.path.join(output_directory, output_file_name)

    # Save the combined data to MGF file
    start_time = time.time()
    save_combined_data_to_mgf(matched_data, output_file_path)
    saving_time = time.time() - start_time
    print(f"Saving combined data took {saving_time:.2f} seconds")

    print(f"Combined data saved to {output_file_path}")
