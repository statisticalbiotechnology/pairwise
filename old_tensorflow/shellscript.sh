#!/bin/bash

#SBATCH -J foundational
#SBATCH -o %x.%A_%a.%N.out
#SBATCH -e %x.%A_%a.%N.gerr
#SBATCH -D ./
#SBATCH --get-user-env
 
#SBATCH --partition=compms-cpu-small # compms-gpu-a40
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=13
#SBATCH --mem=50G
#SBATCH --tasks-per-node=1
 
 
#SBATCH --mail-user=ge27buk@mytum.de
#SBATCH --mail-type=end
 
#SBATCH --export=NONE
#SBATCH --time=24:00:00
##SBATCH --array=1-100%3

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tf


#CUDA_VISIBLE_DEVICES=1 python -u PretrainModel.py &> logger$1.log
CUDA_VISIBLE_DEVICES=2 python -u downstream.py &> logger$1.log
