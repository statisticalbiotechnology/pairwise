### Install

#### **Recursively clone repo**
We attach our fork of the Depthcharge package as a submodule. Therefore, this repository needs to be recursively cloned:

`git clone --recursive git@github.com:statisticalbiotechnology/pairwise.git`

`cd pairwise`

#### **Install environment (Conda)**
1. **Create environment:**

    `conda env create -f environment.yml`

    `conda activate pairwise_env`

2. **Install local Depthcharge**

    `conda activate pairwise_env`

    `cd depthcharge`

    `python -m pip install .`

### Running PA (container)

The reproducible way to run PA for de-novo inference (and the backend used by the
[denovo_benchmarks](https://github.com/bittremieuxlab/denovo_benchmarks) harness) is the
Apptainer/Singularity image defined in [`misc/container.def`](misc/container.def).
The harness scripts it invokes are mirrored under
[`misc/denovo_benchmarks_pairwise/`](misc/denovo_benchmarks_pairwise/).

**What the definition builds** ([`misc/container.def`](misc/container.def)):
- Base image `continuumio/miniconda3:latest` (`container.def:2`).
- Clones this repo over HTTPS and rewrites the SSH submodule URL to HTTPS before
  `git submodule update --init --recursive` (`container.def:17-21`).
- Creates the `pairwise_env` conda env from [`environment.yml`](environment.yml)
  (`container.def:24`) and installs the depthcharge fork with
  `pip install .` from the `depthcharge/` submodule (`container.def:33-34`).
- Replaces `src/data/create_lance.py` with the benchmark copy (`container.def:37`).
- Downloads the checkpoint from HuggingFace (`container.def:40`):
  `https://huggingface.co/alfred-n/PA-transformer/resolve/main/pairwise_mskb.ckpt`

> Note: the `%files` section copies `algorithms/pairwise` and `algorithms/base`
> (`container.def:8-12`), so the image is meant to be **built from inside a
> `denovo_benchmarks` checkout** that provides those directories — not from this
> repo's root.

**Build & run**

```bash
# from a denovo_benchmarks checkout (build context with algorithms/pairwise + algorithms/base)
singularity build pairwise.sif algorithms/pairwise/container.def

# mount the directory holding your spectra at /algo/data
singularity run --bind /path/to/spectra:/algo/data pairwise.sif
```

**I/O contract** — input spectra are bind-mounted at `/algo/data`; the runscript
runs `cd /algo && ./make_predictions.sh data` (`container.def:48-49`).
The authoritative result is `/algo/outputs.csv` (`container.def:50`,
written by `output_mapper.py:205`), produced from the model's raw mzTab.

**Internal pipeline** ([`make_predictions.sh`](misc/denovo_benchmarks_pairwise/make_predictions.sh)):
1. `python src/data/create_lance.py --input /algo/data/ --output /algo/data.lance`
   converts the mounted spectra to a Lance dataset (`make_predictions.sh:17`).
2. `python src/main.py --config=configs/master_bm.yaml --downstream_root_dir=/algo/data.lance --downstream_weights=/algo/pairwise_mskb.ckpt`
   runs prediction (`make_predictions.sh:20`), writing
   `outs/logs/log/predictions_table.mzTab` (`configs/master_bm.yaml:2`,
   `src/pl_callbacks.py:178`).
3. `python output_mapper.py ...` converts that mzTab into `/algo/outputs.csv`
   (`make_predictions.sh:25`, `output_mapper.py:205`).

### I/O differences from Casanovo

PA is API-compatible with Casanovo's *task* but not its I/O. The key differences:

**Input**
- **No direct mzML/MGF reading.** Casanovo reads peak files directly; PA first
  converts them to a Lance dataset with
  [`create_lance.py`](misc/denovo_benchmarks_pairwise/create_lance.py)
  (depthcharge `SpectrumDataset`, `create_lance.py:27-33`) and the model reads
  Lance via `BenchmarkDataModule`/`LanceDataset` (`src/data/lance_data_module.py:104-134`).
- **MGF by default.** `create_lance.py` defaults to `--suffix .mgf`
  (`create_lance.py:14`). mzML/mzXML parsers exist in depthcharge
  (`depthcharge/data/parsers.py`), so `--suffix .mzML` works *in principle*, but
  the `title` custom field is MGF-only and raises
  `'title' not found in spectrum` on mzML (`create_lance.py:32`;
  depthcharge `parsers.py:154,157`), so it must be dropped for mzML. mzML also
  goes through `pyteomics.mzml`, which is not exercised by the bundled
  `environment.yml` (see [docs/backend_notes.md](docs/backend_notes.md)).
  depthcharge's default Lance schema already carries `peak_file` and `scan_id`
  (`parsers.py:86-92`), so `title` is the only field PA adds.
- **Precursor mass convention differs.** PA's `precursor_mass`
  (`batch["precursor_mass"]`, packed as `precursors = [mass, charge, mz]` in
  `DeNovoSpec2Pep._parse_batch`, `src/wrappers/casanovo_trainer_wrapper.py:150-154`)
  is **computed**, not stored: the Lance schema holds only `precursor_mz`/
  `precursor_charge` (`parsers.py:89-90`) and the benchmark collate fn sets
  `precursor_mass = precursor_mz * precursor_charge`
  (`src/collate_functions.py:156`, configured at `src/utils.py:252-253`). That is
  the **protonated cluster mass** `M + z·1.00728 ≈ mz·z`, **not** the neutral
  monoisotopic mass `(mz − 1.00728)·z` that Casanovo uses. Anyone reusing PA's
  precursor mass must convert: `M_neutral = precursor_mass − z·1.00728`.

**Output**
- **PA's native output is an mzTab**, `predictions_table.mzTab`
  (`src/pl_callbacks.py:178`), with columns including `sequence`
  (comma-separated tokens), `search_engine_score[1]` (peptide score),
  `opt_ms_run[1]_aa_scores` (per-residue confidence), `charge`,
  `exp_mass_to_charge`, `calc_mass_to_charge`, `spectra_ref` (`peak_file:scan_id`)
  and `title` (`src/data/mzTab_writer.py:9-31`, `src/pl_callbacks.py:223-246`).
  Residues are labelled `C+57.021`, `M+15.995`, etc. (`src/utils.py:33,49`).
  The harness then post-processes this into `outputs.csv` via
  [`output_mapper.py`](misc/denovo_benchmarks_pairwise/output_mapper.py), which
  rewrites tokens to ProForma Unimod (`C+57.021`→`C[UNIMOD:4]`,
  `M+15.995`→`M[UNIMOD:35]`, …; `output_mapper.py:20-30`). Casanovo emits mzTab
  natively, with `C[Carbamidomethyl]`-style residue labels.
- **No per-step token probability distributions** are written by the container or
  `make_predictions.sh`. The mzTab carries only scalar per-residue confidences,
  and `outputs.csv` even *simulates* those by repeating the peptide score per
  token (`output_mapper.py:191`). A consumer that needs the full distributions
  (e.g. a borgonovo-style backend) must call the model directly — see
  [docs/backend_notes.md](docs/backend_notes.md).

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
conda activate pairwise_env
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

### Reproducing 9 species cross validation results
1. Download checkpoint for specific species and 9 species version (1 or 2)
2. Checkout to branch "9_species_cross_validation"
3. Run script roc_curve.py with following command:
   python roc_curve.py path_to_checkpoint species_name output_name_extension
   - Output is a roc curve in a two column csv file
