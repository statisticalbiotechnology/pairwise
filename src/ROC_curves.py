import os
import numpy as np
from pyteomics import mztab
from tqdm import tqdm
from casanovo_eval import aa_match_batch
from utils import RESIDUES_MSKB


### PA longer training
MZTAB_FILES = {
    "vigna_mungo": "/proj/bedrock/logs/log_22_04_29_311575__12_01_25/predictions_table.mzTab",
    "solanum_lycopersicum": "/proj/bedrock/logs/log_22_04_04_524563__12_01_25/predictions_table.mzTab",
    "mus_musculus": "/proj/bedrock/logs/log_22_02_18_547329__12_01_25/predictions_table.mzTab",
    "candidatus_endoloripes": "/proj/bedrock/logs/log_21_57_45_539976__12_01_25/predictions_table.mzTab",
    "homo_sapiens": "/proj/bedrock/logs/log_21_55_18_659579__12_01_25/predictions_table.mzTab",
    "apis_mellifera": "/proj/bedrock/logs/log_21_54_01_590740__12_01_25/predictions_table.mzTab",
    "methanosarcina_mazei": "/proj/bedrock/logs/log_22_00_36_235300__12_01_25/predictions_table.mzTab",
    "bacillus_subtilis": "/proj/bedrock/logs/log_21_56_10_286598__12_01_25/predictions_table.mzTab",
    "saccharomyces_cerevisiae": "/proj/bedrock/logs/log_22_03_28_533184__12_01_25/predictions_table.mzTab",
}

### PA (probably)
# MZTAB_FILES = {
#     "vigna_mungo": "/proj/bedrock/logs/log_00_32_21_475342__15_11_24/predictions_table.mzTab",
#     "solanum_lycopersicum": "/proj/bedrock/logs/log_10_42_17_047925__15_11_24/predictions_table.mzTab",
#     "mus_musculus": "/proj/bedrock/logs/log_00_30_10_333368__15_11_24/predictions_table.mzTab",
#     "candidatus_endoloripes": "/proj/bedrock/logs/log_10_40_54_651539__15_11_24/predictions_table.mzTab",
#     "homo_sapiens": "/proj/bedrock/logs/log_00_27_17_805437__15_11_24/predictions_table.mzTab",
#     "apis_mellifera": "/proj/bedrock/logs/log_10_36_08_349127__15_11_24/predictions_table.mzTab",
#     "methanosarcina_mazei": "/proj/bedrock/logs/log_00_29_35_381442__15_11_24/predictions_table.mzTab",
#     "bacillus_subtilis":"/proj/bedrock/logs/log_22_04_29_663815__11_12_24/predictions_table.mzTab",
#     "saccharomyces_cerevisiae": "/proj/bedrock/logs/log_00_30_54_347585__15_11_24/predictions_table.mzTab",
# }

### BASE ARCH (probably)
# MZTAB_FILES = {
#     "vigna_mungo": "/proj/bedrock/logs/log_00_43_58_138784__28_11_24/predictions_table.mzTab",
#     "solanum_lycopersicum": "/proj/bedrock/logs/log_00_41_09_669643__28_11_24/predictions_table.mzTab",
#     "mus_musculus": "/proj/bedrock/logs/log_00_49_56_197103__28_11_24/predictions_table.mzTab",
#     "candidatus_endoloripes": "/proj/bedrock/logs/log_00_46_49_892110__28_11_24/predictions_table.mzTab",
#     "homo_sapiens": "/proj/bedrock/logs/log_00_48_24_395449__28_11_24/predictions_table.mzTab",
#     "apis_mellifera": "/proj/bedrock/logs/log_00_45_16_444796__28_11_24/predictions_table.mzTab",
#     "methanosarcina_mazei": "/proj/bedrock/logs/log_00_39_29_902246__28_11_24/predictions_table.mzTab",
#     "bacillus_subtilis":"/proj/bedrock/logs/log_09_10_06_929389__16_12_24/predictions_table.mzTab",
#     "saccharomyces_cerevisiae": "/proj/bedrock/logs/log_14_52_50_199180__28_11_24/predictions_table.mzTab",
# }
# OUTPUT_DIR = "/proj/bedrock/results/pw_paper/pairwise_ROC_MSKB"
# OUTPUT_DIR = "/proj/bedrock/results/pw_paper/base_arch_ROC_MSKB"
OUTPUT_DIR = "/proj/bedrock/results/pw_paper/pairwise_LONGER_ROC_MSKB"
PEPTIDE_COLUMN = "sequence"
PEPTIDE_SCORE_COLUMN = "search_engine_score[1]"
AA_SCORES_COLUMN = "opt_ms_run[1]_aa_scores"
GROUND_TRUTH_COLUMN = "opt_ms_run[1]_ground_truth_sequence"
RESIDUES_DICT = RESIDUES_MSKB
SAVE_RESULTS = True

# Choose confidence calculation method: "peptide_score" or "aa_score_product"
CONFIDENCE_METHOD = "peptide_score" 
# CONFIDENCE_METHOD = "aa_score_product"

# Helper function to parse comma-separated list columns
def parse_comma_separated_list(value):
    if isinstance(value, str) and value:
        return [item.strip() for item in value.split(",")]
    return []

os.makedirs(OUTPUT_DIR, exist_ok=True)

for species, mz_tab_file in MZTAB_FILES.items():
    print(f"Processing {species} from {mz_tab_file}")
    mztab_data = mztab.MzTab(mz_tab_file)
    psm_df = mztab_data.spectrum_match_table

    psm_df[PEPTIDE_COLUMN] = psm_df[PEPTIDE_COLUMN].apply(parse_comma_separated_list)
    psm_df[GROUND_TRUTH_COLUMN] = psm_df[GROUND_TRUTH_COLUMN].apply(parse_comma_separated_list)
    psm_df[AA_SCORES_COLUMN] = psm_df[AA_SCORES_COLUMN].apply(
        lambda x: [float(score) for score in x.split(",")] if isinstance(x, str) else []
    )

    valid_preds = psm_df.dropna(subset=[PEPTIDE_COLUMN, PEPTIDE_SCORE_COLUMN, GROUND_TRUTH_COLUMN])

    total_spectra = len(valid_preds)
    peptide_matches = 0
    all_results = []
    
    for _, row in tqdm(valid_preds.iterrows(), total=total_spectra):
        pred_sequence = row[PEPTIDE_COLUMN]
        ground_truth_sequence = row[GROUND_TRUTH_COLUMN]
        aa_scores = row[AA_SCORES_COLUMN]
        peptide_score = row[PEPTIDE_SCORE_COLUMN]

        aa_matches_batch, _, _ = aa_match_batch([ground_truth_sequence], [pred_sequence], aa_dict=RESIDUES_DICT)
        aa_matches, pep_match = aa_matches_batch[0]

        if CONFIDENCE_METHOD == "peptide_score":
            confidence = peptide_score
        elif CONFIDENCE_METHOD == "aa_score_product":
            confidence = np.prod(aa_scores)
        else:
            raise ValueError("Invalid CONFIDENCE_METHOD. Choose 'peptide_score' or 'aa_score_product'.")

        all_results.append([int(pep_match), float(confidence)])
        if pep_match:
            peptide_matches += 1

    peptide_precision = peptide_matches / total_spectra
    print(f"Peptide-Level Precision for {species}: {peptide_precision:.4f}")

    if SAVE_RESULTS:
        # ROC curve
        all_results_np = np.array(all_results)
        # Sort by confidence score in descending order
        asort = all_results_np[:, 1].argsort()
        all_results_np = all_results_np[asort][::-1]
        pm, conf = all_results_np[:, 0], all_results_np[:, 1]

        # Remove the negative confidence check
        # c_neg = conf <= 0
        # num_neg = np.sum(c_neg)
        # print("#confidences lower than 0:", num_neg)
        # if num_neg > 0:
        #     print("confidences lower than 0 min val:", np.min(conf[c_neg]))
        #     print("confidences lower than 0 max val:", np.max(conf[c_neg]))
        #     raise ValueError("Found confs < 0")
        
        cumsum_pm = np.cumsum(pm)
        n_predictions = len(pm)

        # Calculate precision and coverage at each point
        precision = cumsum_pm / np.arange(1, n_predictions + 1)
        coverage = np.arange(1, n_predictions + 1) / n_predictions

        # Combine coverage and precision into a curve
        curve = np.column_stack((coverage, precision))

        output_file = os.path.join(OUTPUT_DIR, f"{species}_roc_curve.csv")
        np.savetxt(output_file, curve, delimiter=',', header="coverage,precision", comments='')
        print(f"ROC Curve saved to {output_file}")

        final_precision = curve[-1][1] if curve.size > 0 else 0.0
        print(f"Final Precision @ coverage=1.0: {final_precision:.6f}")
