### Install

#### **Recursively clone repo**
We attach our fork of the Depthcharge package as a submodule. Therefore, this repository needs to be recursively cloned:

`git clone --recursive git@github.com:statisticalbiotechnology/pairwise.git`

`cd pairwise`

#### **Install environment (Conda)**
1. **Create environment:**

    `conda create --name Pairwise python=3.10 -y`

    `conda activate Pairwise`

2. **Install PyTorch**

    Follow the official PyTorch installation guide and install the appropriate version for your system (with or without CUDA), since PyTorch's Conda channel is no longer officially supported:

    - Find the latest install command at: https://pytorch.org/get-started/locally/
    - Example for CUDA 12.1 support

        ```bash
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3. **Install further dependecies:**
    ```bash
    conda env update --file environment.yml --prune


4. **Install local Depthcharge**

    `conda activate Pairwise`

    `cd depthcharge`

    `python -m pip install .`

### Framework/structure
We utilize Pytorch Lightning. We define the models in `src/models/`, and wrap them with Lightning modules that contain the training code. The training wrappers are found in `src/pl_wrappers.py`.

A number of callbacks such as for annealing the LR and temperature are found in `src/pl_callbacks.py`. They can be toggeled by providing the corresponding argument, i.e. `--anneal_lr=True`.

### Logging
We use WandB for detailed logging.

**Either** sign in to your account
`wandb login`
and set the arguments

`python src/main.py --config=<your-config>  --wandb_project=<your-project> --wandb_entity=<your-entity>`

**or** run the main script `src/main.py` directly and select:
* option "(1) Private W&B dashboard, no account required" (requires internet connection)
or
* option "(4) Don’t visualize my results" (offline)

`python src/main.py --config=<your-config>`

### Train 
Example command: 

`python src/main.py --config=configs/example.yaml`

Example sbatch script for multi-node/multi-gpu training

```
#!/bin/bash
#SBATCH --gpus 64
#SBATCH -t 3-00:00:00
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=16

module load Anaconda/2021.05-nsc1
conda activate MS2transformers
export OMP_NUM_THREADS=16

srun python src/main.py --config=configs/example.yaml \
--num_devices=8 --num_nodes=8 --matmul_precision="medium"
```
Submit job
```
sbatch job.sbatch
```
### Arguments, configs and priority
The full list of arguments and their descriptions can be found in `src/parse_args.py`. These arguments are the same as the arguments in the config files. You can adjust the configs files or provide the arguments from the command line.

**Priority**: Provided command line args > config values > argparse defaults

### Datasets
TODO
