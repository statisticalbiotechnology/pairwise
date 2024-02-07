#!/usr/bin/env bash
#SBATCH -A Berzelius-2024-32
#SBATCH -t 0-06:00:00
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_outs/slurm-%j.out   # Redirect stdout to "slurm-outs" folder
#SBATCH --error=slurm_outs/slurm-%j.err    # Redirect stderr to "slurm-outs" folder

module load Anaconda/2021.05-nsc1
conda activate FoundationV2

srun python src/main.py --config=configs/berz_scratch.yaml --num_devices=1 --num_nodes=1