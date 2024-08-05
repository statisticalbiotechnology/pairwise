#!/usr/bin/env bash
#SBATCH -A Berzelius-2024-32
#SBATCH -t 3-00:00:00
#SBATCH --gpus=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_outs/slurm-%j.out   # Redirect stdout to "slurm-outs" folder
#SBATCH --error=slurm_outs/slurm-%j.err    # Redirect stderr to "slurm-outs" folder

module load Anaconda/2021.05-nsc1
conda activate FoundationV2

srun python src/main.py --config=configs/berz_pt_mae_ds_mskb.yaml --num_devices=1 --num_nodes=1\
 --resume=0 --pretrain=0 --encoder_weights=/proj/bedrock/checkpoints/checkpoint_19_23_04_552168__30_01_24/epoch=3-masked_ae_val_loss_epoch=0.39.ckpt