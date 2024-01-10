#!/usr/bin/env bash
#SBATCH -A Berzelius-2023-189
#SBATCH -t 0-02:00:00
#SBATCH --gpus=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=16


module load Anaconda/2021.05-nsc1
conda activate FoundationV2

srun python src/main.py --config=configs/berzelius.yaml --num_devices=8 --num_nodes=1