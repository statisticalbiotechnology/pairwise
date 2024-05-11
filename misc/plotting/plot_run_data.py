import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import to_rgb, to_hex

# Constants
DOWNLOADED_DATA = "/Users/alfred/Documents/Code/bedrock/misc/plotting/csv/Foundational-msConvert-grid-search_05_10__23_21.csv"
EPOCH_CUTOFF = 10  # Number of epochs to display
METRIC_NAME = "denovo_tf_val_aa_prec_epoch"
METRIC_DISPLAY_NAME = "AA-level Precision"
LINE_THICKNESS = 3  # Line thickness for plots
ENABLE_COLOR_SCALING = True  # Enable or disable color scaling based on blr
SCALING_PARAM = (
    "downstream_config.denovo_tf.blr"  # Parameter to scale color lightness w.r.t.
)
FIGSIZE = (24.12 / 2, 17.8 / 2)

# Set the aesthetic style of the plots
sns.set_theme()  # Use Seaborn to set aesthetic style

# Define colors for each group
base_colors = {
    "Scratch": "#FFA500",  # Nice orange
    "Scratch, LR scheduling": "#FFA500",  # Initially same as Scratch, adjusted later
    "Pretrained": "#003366",  # Marine blue
}

# Define line styles
line_styles = {
    "Scratch": "dashed",  # Dashed line
    "Scratch, LR scheduling": "dashdot",  # Dotted-dashed line
    "Pretrained": "solid",  # Regular line
}

# Load the data
df = pd.read_csv(DOWNLOADED_DATA, converters={METRIC_NAME: eval})


# Function to adjust color lightness based on a value
def adjust_lightness(color, amount):
    c = to_rgb(color)
    c = np.array(c) + (1.0 - np.array(c)) * amount
    c = np.clip(c, 0, 1)
    return to_hex(c)


# Precompute colors for each blr within each group
color_settings = {}
if ENABLE_COLOR_SCALING:
    scratch_blr_values = sorted(df[df["Group"] == "Scratch"][SCALING_PARAM].unique())
    scratch_lightness = np.linspace(
        0.5, 0.0, len(scratch_blr_values)
    )  # lightness settings from dark to lighter
    scratch_colors = {
        blr: adjust_lightness(base_colors["Scratch"], l)
        for blr, l in zip(scratch_blr_values, scratch_lightness)
    }

    for group, group_df in df.groupby("Group"):
        if group == "Scratch, LR scheduling":
            # Match BLR settings to nearest in Scratch and reuse those lightness values
            warmup_blr_values = group_df[SCALING_PARAM].unique()
            warmup_colors = {}
            for wblr in warmup_blr_values:
                nearest_scratch_blr = min(
                    scratch_blr_values, key=lambda sblr: abs(sblr - wblr)
                )
                warmup_colors[wblr] = scratch_colors[nearest_scratch_blr]
            color_settings[group] = warmup_colors
        else:
            blr_values = sorted(group_df[SCALING_PARAM].unique())
            lightness = np.linspace(
                0.5, 0, len(blr_values)
            )  # lightness settings from dark to lighter
            base_color = base_colors[group]
            color_settings[group] = {
                blr: adjust_lightness(base_color, l)
                for blr, l in zip(blr_values, lightness)
            }
else:
    for group in df["Group"].unique():
        color_settings[group] = {
            None: base_colors[group]
        }  # Use base color without scaling

# Plotting
fig, ax = plt.subplots(figsize=FIGSIZE)

for index, row in df.iterrows():
    group = row["Group"]
    blr = row[SCALING_PARAM]
    color = color_settings[group].get(
        blr, base_colors[group]
    )  # Fallback to base color if not found
    style = line_styles[group]
    metrics = row[METRIC_NAME][:EPOCH_CUTOFF]  # Use global variable for cutoff

    ax.plot(
        range(1, EPOCH_CUTOFF + 1),
        metrics,
        color=color,
        linestyle=style,
        linewidth=LINE_THICKNESS,  # Adjust line thickness
        label=f"{group} LR={blr}" if group != "Scratch, LR scheduling" else f"{group}",
    )

# Simplify the legend to show unique entries
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(
    by_label.values(),
    by_label.keys(),
    loc="lower right",
    fontsize="xx-large",
    fancybox=True,
)

# Adjust label and tick sizes
ax.set_xlabel("Epoch", fontsize=20)  # Set font size for x-axis label
ax.set_ylabel(METRIC_DISPLAY_NAME, fontsize=20)  # Set font size for y-axis label
ax.set_title("Peptide Prediction Precision", fontsize=22)  # Set font size for the title

# Set tick sizes
ax.tick_params(axis="both", which="major", labelsize=18)  # Adjust ticks for both axes

csv_directory = os.path.dirname(DOWNLOADED_DATA)
pdf_filename = os.path.join(
    csv_directory, os.path.splitext(os.path.basename(DOWNLOADED_DATA))[0] + "_plot.pdf"
)
fig.savefig(pdf_filename, format="pdf")
