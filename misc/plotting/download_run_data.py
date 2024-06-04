import os
import pandas as pd
import wandb
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Initialize the wandb API
api = wandb.Api()
entity = "kall"

# Runs to download
# Project, group, run_id
RUNS = {
    "Foundational-msConvert-grid-search": {
        "Scratch": [
            "c3ovcpml",
            "mpjvoita",
            "lxyk5ruj",
        ],
        "Scratch, LR scheduling": [
            "w83fqylu",
        ],
        "Pretrained": [
            "75xd1jbo",
            "1wj6jler",
            "lmaxfwe1",
        ],
    }
}

# Hyperparameter setting to retrieve for each run
hparams = (
    "downstream_config.denovo_tf.blr",
    "downstream_config.denovo_tf.warmup",
)

# Logged metrics to retrieve for each run
metrics = ("denovo_tf_val_aa_prec_epoch",)


def get_nested_config(config, path):
    keys = path.split(".")
    value = config
    for key in keys:
        value = value.get(key, None)
        if value is None:
            break
    return value


for project, groups in RUNS.items():
    run_data = []
    for group, run_ids in groups.items():
        for run_id in tqdm(run_ids, desc=f"Downloading run data for {group}"):
            name = f"{entity}/{project}/{run_id}"
            run = api.run(name)

            # Collect hyperparameters and metric values
            run_hparams = {
                hparam: get_nested_config(run.config, hparam) for hparam in hparams
            }
            run_metrics = {}
            history = run.scan_history()
            for metric in metrics:
                values = [row.get(metric) for row in history]
                values = [v for v in values if v is not None]
                run_metrics[metric] = (
                    values if values else [None]
                )  # handle empty metrics

            # Append run data with group
            run_data.append(
                {**run_hparams, **run_metrics, "Group": group, "Run ID": run_id}
            )

    # Create DataFrame
    df = pd.DataFrame(run_data)

    # Save to CSV
    save_dir = "misc/plotting/csv/"
    os.makedirs(save_dir, exist_ok=True)
    now = datetime.now()
    filename = os.path.join(save_dir, f"{project}_{now.strftime('%m_%d__%H_%M')}.csv")
    df.to_csv(filename)
    print(f"Saved '{filename}'")
