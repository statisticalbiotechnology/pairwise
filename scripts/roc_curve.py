import torch
import sys
import os
from tqdm import tqdm

### ADAPT CODE START ###
code_path = "/proj/bedrock/users/x_joela/pairwise"
sys.path.append(os.path.join(code_path, "src"))

# Path to directory containing 9 species dataset
# Note: that parquet files for species must be in this path + parquet/processed/
data_root_dir = "/proj/bedrock/datasets/9_species_V1"

### ADAPT CODE END ###

from parse_args import parse_args_and_config, create_output_dirs
dataframe = False

# Checkpoint for a pytorch lightning checkpoint
assert len(sys.argv) == 4, "python roc_curve.py {ckpt_path} {val_species} {seed}"
ckpt = sys.argv[1]
species_name = sys.argv[2]
seed = sys.argv[3]
outname = f"{species_name}{seed}_roc_curve"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Will save file {outname}.csv")

import yaml
with open(os.path.join(code_path, "configs/denovo/9_species_config.yaml")) as f:
    ds_config = yaml.safe_load(f)

# Dataloader

species_size = {line.split()[0] : int(line.split()[1]) for line in open(os.path.join(data_root_dir, "species_sizes.txt")).read().strip().split("\n")}[species_name]
dataset_directory = os.path.join(data_root_dir, "parquet/processed")
dictionary_path = os.path.join(data_root_dir, "ns_dictionary.txt")
masses_path = os.path.join(data_root_dir, "ns_masses.txt")
from data.loader_hf import LoaderHF
loader = LoaderHF(
    dataset_directory=dataset_directory,
    val_species=species_name,
    dictionary_path=dictionary_path,
    tokenizer_path=data_root_dir,
    top_peaks=150,
    num_workers=1,
    pep_length=ds_config['pep_length'],
    charge=ds_config['charge'],
    buffer_size=ds_config['buffer_size'],
)
testloader = loader.build_dataloader(loader.dataset['test'], 100, 1)

from copy import deepcopy
amod_dic = loader.amod_dic
input_dict = deepcopy(loader.amod_dic)
input_dict["<SOS>"] = len(loader.amod_dic)
output_dict = deepcopy(loader.amod_dic)
output_dict["<EOS>"] = len(loader.amod_dic)
token_dicts = {
    "amod_dict": amod_dic,
    "input_dict": input_dict,
    "output_dict": output_dict,
}

token_dicts["residues"] = {line.split()[0]:float(line.split()[1]) for line in open(masses_path)}
amod_dict = token_dicts["amod_dict"]

# Downstream PL object
import models.encoder as encoders
import models.dc_encoder as dc_encoders
ENCODER_DICT = {
    **encoders.__dict__,
    **dc_encoders.__dict__,
}
import models.dc_decoder as dc_decoders
import models.decoder as decoders
DECODER_DICT = {
    **decoders.__dict__,
    **dc_decoders.__dict__,
}

from wrappers.downstream_wrappers import DeNovoTeacherForcing
DOWNSTREAM_TASK_DICT = {
    "denovo_tf": DeNovoTeacherForcing,
}

class GlobalArgs:
    def __init__(self):
        self.batch_size = 100
        self.num_workers = 1
        self.pin_mem = False
        self.accum_iter = 1
        self.num_devices = 1
        self.num_nodes = 1
        self.scale_lr_by_batchsize = False
        self.mask_zero_tokens = False
        self.log_wandb = False
        
        self.encoder_model = "encoder_pairwise"
        self.decoder_model = "casanovo_decoder"
global_args = GlobalArgs()

import utils

encoder = ENCODER_DICT[global_args.encoder_model](
    use_charge=False,
    use_mass=False,
    use_energy=False,
    dropout=ds_config.get("dropout", 0),
    cls_token=False,
)

decoder = DECODER_DICT[global_args.decoder_model](
    token_dicts,
    d_model=encoder.running_units,
    dropout=ds_config['denovo_tf']["decoder_dropout"],
    cross_attend=True,
    max_seq_len=ds_config['pep_length'][-1]+1, # +1 because of added EOS token
)
pl_downstream = DOWNSTREAM_TASK_DICT["denovo_tf"](
    encoder,
    decoder,
    global_args=global_args,
    token_dicts=token_dicts,
    task_dict=ds_config['denovo_tf'],
)

print(
    f"Found checkpoint: {ckpt}"
)
downstream_ckpt = torch.load(ckpt, map_location=device)
pl_downstream.load_state_dict(downstream_ckpt["state_dict"])
pl_downstream.to(device)
pl_downstream.eval()

# Scale
import utils
ms = utils.RESIDUES_MSKB
# Special exception for alternate 9 species V2 dataset
if data_root_dir.split("_")[-1] == "MSV000090982":
    ms = {
        line.split()[0] : float(line.split()[1]) 
        for line in 
        open(os.path.join(data_root_dir, "ns_masses.txt")).read().strip().split("\n")
    }
scale = {pl_downstream.output_dict[aa]:mass for aa, mass in ms.items() if aa in pl_downstream.output_dict}
scale = dict(sorted(scale.items()))
scale[len(scale)] = 0
scale[len(scale)] = 0
scale_ = torch.tensor(list(scale.values()))#.to(device)

# Do the evaluation

from casanovo_eval import aa_match_batch
from wrappers.downstream_wrappers import fill_null_after_first_EOS

A = {
    'target_intseq': [],
    'pred_intseq': [],
    'correct': []
}
all_results = []
total_spectra = 0
total_correct = 0
for M, batch in (pbar := tqdm(enumerate(testloader), total=species_size//global_args.batch_size)):
    parsed_batch, batch_size = pl_downstream._parse_batch(batch)
    total_spectra += batch_size
    #print("\rTotal spectra=%d out of %d"%(total_spectra, species_size), end='')
    if dataframe: A['target_intseq'].append(parsed_batch['target_intseq'].tolist())

    real_aa_mass = (parsed_batch['mass'] - 1.00727646688) * parsed_batch['charge'] - 18.010565 
    parsed_batch = {m: n.to(device) for m,n in parsed_batch.items()}
    with torch.no_grad():
        out = pl_downstream.eval_forward(parsed_batch)
    logits = out[1] #['logits'] # [1]
    confidences, pred_seq = logits.cpu().softmax(-1).max(-1)
    if dataframe: A['pred_intseq'].append(pred_seq.tolist())
    pred_seq = fill_null_after_first_EOS(pred_seq, output_dict['X'], output_dict['<EOS>'])
    
    # Scale
    pred_aa_mass = torch.gather(scale_[None].repeat(pred_seq.shape[0], 1), 1, pred_seq).sum(1)
    abs_ppm_mass = 1e6*abs(real_aa_mass - pred_aa_mass) / pred_aa_mass

    target_str = pl_downstream.to_aa_sequence(parsed_batch['target_intseq'])
    pred_str = pl_downstream.to_aa_sequence(pred_seq)

    aa_matches_batch, n_aa_true, n_aa_pred = aa_match_batch(
	target_str, pred_str, aa_dict=pl_downstream.residues, mode="best"
    )

    for m, matches in enumerate(aa_matches_batch):
        aa_matches, pep_match = matches
        confidence = confidences[m, :len(aa_matches)].mean()#prod()
        if abs_ppm_mass[m] > 50:
            confidence -= 1
        all_results.append([int(pep_match), float(confidence)])
        total_correct += pep_match
        if dataframe: A['correct'].append(int(pep_match))

    pbar.set_description("Recall_100: %.3f"%(total_correct / total_spectra))

print()

if dataframe:
    A['target_intseq'] = [m for n in A['target_intseq'] for m in n]
    A['pred_intseq'] = [m for n in A['pred_intseq'] for m in n]
    import pandas as pd
    pd.DataFrame(A).to_parquet(f"{outname}.parquet")

# Calculate ROC curve in loop

import numpy as np
all_results = np.array(all_results)
asort = all_results[:,1].argsort()
all_results = all_results[asort][::-1]
[pm, conf] = [m.squeeze() for m in np.split(all_results, 2, 1)]
cumsum = np.cumsum(pm)

precision = cumsum / np.arange(1,cumsum.shape[0]+1,1)
recall = np.arange(1, cumsum.shape[0]+1, 1) / cumsum.shape[0]
curve = np.concatenate([recall[:,None], precision[:,None]], 1)

print()

np.savetxt(f"{outname}.csv", curve, delimiter=',', header="recall,precision")
