import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Added pandas to read Excel files

# Constants and configurations
OUTPUT_DIR_MODEL1 = ""
OUTPUT_DIR_MODEL2 = ""
PLOTTING_OUTPUT_DIR = ""
FILE_EXT = ".pdf"

MODELS = {
    "PA": {
        "dir": OUTPUT_DIR_MODEL1,
        "filename_pattern": "{species_key}_roc_curve.csv",
        "label": "PA",
        "color": "red",
    },
    "Base": {
        "dir": OUTPUT_DIR_MODEL2,
        "filename_pattern": "{species_key}_roc_curve.csv",
        "label": "Base",
        "color": "blue",
    },
}

# Path to Casanovo's data Excel file
CASANOVO_TABLE_PATH = "/proj/bedrock/users/x_alfni/bedrock/Figure_S2.xlsx"  # Set this to your actual file path

# Mapping from Casanovo sheet names to species keys
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

# List of species with their display names
species_list = {
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

# Initialize a data dictionary to store ROC data for each model and species
data = {}
for model_name in MODELS.keys():
    data[model_name] = {}

# Read the ROC curve data from the CSV files for each model
for model_name, model_info in MODELS.items():
    model_dir = model_info["dir"]
    filename_pattern = model_info["filename_pattern"]
    for species_key, species_name in species_list.items():
        file_name = filename_pattern.format(species_key=species_key)
        file_path = os.path.join(model_dir, file_name)
        if os.path.exists(file_path):
            try:
                # Load the data from the CSV file, skipping the header
                data_array = np.loadtxt(file_path, delimiter=",", skiprows=1)
                data[model_name][species_key] = data_array
                print(f"Loaded ROC data for {species_name} ({model_name})")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                data[model_name][species_key] = None
        else:
            print(
                f"ROC curve data file not found for species {species_name} ({model_name}): {file_path}"
            )
            data[model_name][species_key] = None

# Read Casanovo data if CASANOVO_TABLE_PATH is set
casanovo_data = {}
if CASANOVO_TABLE_PATH:
    if os.path.exists(CASANOVO_TABLE_PATH):
        # Read the Excel file with multiple sheets
        xls = pd.ExcelFile(CASANOVO_TABLE_PATH)
        for sheet_name in xls.sheet_names:
            # Read each sheet into a DataFrame
            df = pd.read_excel(xls, sheet_name)
            # Map sheet_name to species_key
            species_key = CASANOVO_SHEET_TO_SPECIES_KEY.get(sheet_name)
            if species_key:
                casanovo_data[species_key] = df
                print(
                    f"Loaded Casanovo data for species {species_key} from sheet '{sheet_name}'"
                )
            else:
                print(
                    f"Sheet name '{sheet_name}' does not match any species in CASANOVO_SHEET_TO_SPECIES_KEY"
                )
    else:
        print(f"CASANOVO_TABLE_PATH does not exist: {CASANOVO_TABLE_PATH}")
else:
    print("CASANOVO_TABLE_PATH is not set. Skipping Casanovo data.")

for species_key, species_name in species_list.items():
    # Create a new figure for each species
    fig, ax = plt.subplots(figsize=(5, 4))
    has_data = False  # Flag to check if any data is available for this species

    # Plot data for each model
    for model_name, model_info in MODELS.items():
        model_data = data[model_name].get(species_key)
        if model_data is not None:
            # Get the coverage and precision data
            coverage = model_data[:, 0]
            precision = model_data[:, 1]

            # Plot the ROC curve (keeping the same color, linewidth, etc.)
            ax.plot(
                coverage,
                precision,
                color=model_info["color"],
                linewidth=0.75,
                label=model_info["label"],
            )
            has_data = True

    # Plot Casanovo data if available
    casanovo_df = casanovo_data.get(species_key)
    if casanovo_df is not None:
        if (
            "Casanovo_precision" in casanovo_df.columns
            and "Casanovo_coverage" in casanovo_df.columns
        ):
            coverage = casanovo_df["Casanovo_coverage"]
            precision = casanovo_df["Casanovo_precision"]
            ax.plot(
                coverage,
                precision,
                color="black",
                linewidth=0.75,
                linestyle="--",
                label="Casanovo",
            )
            has_data = True
        else:
            print(
                f"Casanovo data for species {species_name} does not contain required columns."
            )
    else:
        print(f"No Casanovo data for species {species_name}")

    # Same axes limits, labels, titles, etc.
    if has_data:
        ax.set_title(species_name, fontsize=9)
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Precision")
        ax.set_xlim([0, 1])
        ax.set_ylim([0.3, 1])  # Adjust if needed based on your data range

        # Add legend
        ax.legend(loc="lower left", fontsize=7)

        # Adjust tick label font size
        ax.tick_params(axis="both", which="major", labelsize=7)
    else:
        # If data is missing for all models, display a message
        ax.set_title(f"{species_name}\n(No data available)", fontsize=9)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Missing",
            horizontalalignment="center",
            verticalalignment="center",
            color="red",
            fontsize=12,
            transform=ax.transAxes,
        )

    # Tight layout for each figure
    plt.tight_layout()

    # Save each figure with the species key in the name
    output_figure_path = os.path.join(
        PLOTTING_OUTPUT_DIR, f"{species_key}_roc_curve" + FILE_EXT
    )
    plt.savefig(output_figure_path)
    print(f"ROC curve figure for {species_name} saved to {output_figure_path}")

    # Close the figure to free memory
    plt.close(fig)


# ------------------------------------------------------------------------------------
# Create an "average ROC" curve across all species for each method
# ------------------------------------------------------------------------------------
# We bin the coverage (0 to 1) into 500 bins and compute the average precision in each bin.
# We'll also do this for Casanovo data if available in all species.


# Prepare a helper function for binning and averaging
def bin_and_average(coverage_vals, precision_vals, num_bins=500):
    """Bin the coverage (0..1) into `num_bins` bins and compute average precision in each bin."""
    if len(coverage_vals) == 0:
        return None, None  # no data to average

    coverage_vals = np.array(coverage_vals)
    precision_vals = np.array(precision_vals)

    # Bin edges and indices
    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(coverage_vals, bin_edges) - 1

    # Collect precision per bin
    binned_precision = [[] for _ in range(num_bins)]
    for i, idx in enumerate(bin_indices):
        if 0 <= idx < num_bins:
            binned_precision[idx].append(precision_vals[i])

    # Compute average precision per bin
    avg_precision = [np.mean(x) if len(x) > 0 else np.nan for x in binned_precision]
    avg_precision = np.array(avg_precision)

    # Use midpoints as representative coverage
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_centers, avg_precision


# Gather coverage and precision data across all species for each model
average_data = {}
for model_name, model_info in MODELS.items():
    all_coverages = []
    all_precisions = []
    for species_key in species_list.keys():
        model_data = data[model_name].get(species_key)
        if model_data is not None:
            all_coverages.extend(model_data[:, 0])
            all_precisions.extend(model_data[:, 1])

    # Bin and compute the average
    bin_centers, avg_precision = bin_and_average(
        all_coverages, all_precisions, num_bins=500
    )
    average_data[model_name] = {
        "coverage": bin_centers,
        "precision": avg_precision,
        "color": model_info["color"],
        "label": model_info["label"],
    }

# Also compute an average for Casanovo if desired
# (only for species that actually have Casanovo data)
casanovo_coverages = []
casanovo_precisions = []
for species_key in species_list.keys():
    df = casanovo_data.get(species_key)
    if (
        df is not None
        and "Casanovo_coverage" in df.columns
        and "Casanovo_precision" in df.columns
    ):
        casanovo_coverages.extend(df["Casanovo_coverage"].values)
        casanovo_precisions.extend(df["Casanovo_precision"].values)

if len(casanovo_coverages) > 0:
    bin_centers_c, avg_precision_c = bin_and_average(
        casanovo_coverages, casanovo_precisions, num_bins=500
    )
    if bin_centers_c is not None:
        average_data["Casanovo"] = {
            "coverage": bin_centers_c,
            "precision": avg_precision_c,
            "color": "black",
            "label": "Casanovo",
            "linestyle": "--",
        }

# Plot the averaged ROC curves in a separate figure
fig, ax = plt.subplots(figsize=(5, 4))

for method_name, method_info in average_data.items():
    coverage = method_info["coverage"]
    precision = method_info["precision"]

    if coverage is not None and precision is not None:
        # Plot with the same style we used above
        linestyle = method_info.get("linestyle", "-")  # Default to solid if not given
        ax.plot(
            coverage,
            precision,
            color=method_info["color"],
            linewidth=0.75,
            linestyle=linestyle,
            label=f"Average {method_info['label']}",
        )

# Same axes limits, labels, etc.
ax.set_title("Average ROC across all species", fontsize=9)
ax.set_xlabel("Coverage")
ax.set_ylabel("Precision")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])  # Adjust if needed
ax.legend(loc="lower left", fontsize=7)
ax.tick_params(axis="both", which="major", labelsize=7)
plt.tight_layout()

# Save the average figure
avg_output_path = os.path.join(PLOTTING_OUTPUT_DIR, "average_roc_curve" + FILE_EXT)
plt.savefig(avg_output_path)
print(f"Average ROC curves figure saved to {avg_output_path}")

plt.close(fig)
