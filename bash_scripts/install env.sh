module load Mambaforge/23.3.1-1-hpc1-bdist

mamba create -n FoundationV2 python=3.10
mamba activate FoundationV2
mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
mamba env update -f env_without_torch.yml