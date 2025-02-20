import os
import numpy as np
import pandas as pd

# Configuration
BASE_ARCH_DIR = ""
PAIRWISE_DIR = "/"

CASANOVO_TABLE_PATH = ""  # Set to actual path
DECIMAL_PLACES = 3  # Number of decimal places to round to

CASANOVO_SHEET_TO_SPECIES_KEY = {
    "honeybee": "apis_mellifera",
    "bacillus": "bacillus_subtilis",
    "clambacteria": "candidatus_endoloripes",
    "human": "homo_sapiens",
    "mmazei": "methanosarcina_mazei",
    "mouse": "mus_musculus",
    "tomato": "solanum_lycopersicum",
    "yeast": "saccharomyces_cerevisiae",
    "ricebean": "vigna_mungo",
}
SPECIES_LIST = {
    "apis_mellifera": "Apis mellifera",
    "bacillus_subtilis": "Bacillus subtilis",
    "candidatus_endoloripes": "Candidatus Endoloripes",
    "homo_sapiens": "Homo sapiens",
    "methanosarcina_mazei": "Methanosarcina mazei",
    "mus_musculus": "Mus musculus",
    "solanum_lycopersicum": "Solanum lycopersicum",
    "saccharomyces_cerevisiae": "Saccharomyces cerevisiae",
    "vigna_mungo": "Vigna mungo",
}

# Initialize a dictionary to store results
results = {
    species: {"Base": None, "Pairwise": None, "Casanovo": None}
    for species in SPECIES_LIST.keys()
}


# Function to extract precision at 100% coverage from CSV
def extract_precision_from_csv(file_path):
    if os.path.exists(file_path):
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
        if len(data) > 0 and data[-1, 0] == 1.0:
            return data[-1, 1]
        else:
            print(f"Warning: Last row in {file_path} does not have coverage=1.0")
    else:
        print(f"File not found: {file_path}")
    return None


# Extract precision for Base and Pairwise models
for model_name, model_dir in [("Base", BASE_ARCH_DIR), ("Pairwise", PAIRWISE_DIR)]:
    for species_key in SPECIES_LIST.keys():
        file_path = os.path.join(model_dir, f"{species_key}_roc_curve.csv")
        precision = extract_precision_from_csv(file_path)
        if precision is not None:
            precision = round(precision, DECIMAL_PLACES)
        results[species_key][model_name] = precision

# Extract precision for Casanovo from Excel
if CASANOVO_TABLE_PATH and os.path.exists(CASANOVO_TABLE_PATH):
    xls = pd.ExcelFile(CASANOVO_TABLE_PATH)
    for sheet_name in xls.sheet_names:
        species_key = CASANOVO_SHEET_TO_SPECIES_KEY.get(sheet_name)
        if species_key:
            df = pd.read_excel(xls, sheet_name)
            if "Casanovo_precision" in df.columns and "Casanovo_coverage" in df.columns:
                # Find the row with coverage close to 1 within a tolerance of 1e-2
                coverage_1_rows = df[
                    np.isclose(df["Casanovo_coverage"], 1.0, atol=1e-2)
                ]
                if not coverage_1_rows.empty:
                    precision = coverage_1_rows["Casanovo_precision"].iloc[-1]
                    if precision is not None:
                        precision = round(precision, DECIMAL_PLACES)
                    results[species_key]["Casanovo"] = precision
                else:
                    print(
                        f"Warning: No rows with coverage≈1.0 for {species_key} in Casanovo data"
                    )
            else:
                print(f"Required columns missing in Casanovo sheet: {sheet_name}")
else:
    print("Casanovo Excel file not found or CASANOVO_TABLE_PATH is not set")

# Convert results to a DataFrame
results_df = pd.DataFrame.from_dict(results, orient="index")
results_df.index.name = "Species"

# Calculate averages for each model
averages = results_df.mean(axis=0, skipna=True).round(DECIMAL_PLACES)
averages.name = "Average"

# Add averages row to the DataFrame
results_df = results_df._append(averages)

# Print results
print(results_df)

# Save results to a CSV file
output_csv_path = "precision_results.csv"
results_df.to_csv(output_csv_path)
print(f"Results saved to {output_csv_path}")
