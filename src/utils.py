from copy import deepcopy

from depthcharge.tokenizers.peptides import MskbPeptideTokenizer, PeptideTokenizer
from data.mskb_tokenizer import MSKBTokenizer
from data.ninespecies_data_module import NinespeciesDataModule
from data.loader_hf import LoaderHF, collate_fn
import numpy as np
from torch.utils.data.dataset import Subset
import os
import torch
from functools import partial

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data.lance_data_module import LanceDataModule
from pl_callbacks import (
    FLOPProfilerCallback,
    CosineAnnealLRCallback,
    LinearWarmupLRCallback,
    ExponentialDecayLRCallback,
)

from collate_functions import pad_peaks, pad_peptides

from loader_parquet import PeptideDataset, PeptideParser

RESIDUES_MSKB = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C+57.021": 160.030649,  # 103.009185 + 57.021464
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    # Amino acid modifications.
    "M+15.995": 147.035400,  # Met oxidation:   131.040485 + 15.994915
    "N+0.984": 115.026943,  # Asn deamidation: 114.042927 +  0.984016
    "Q+0.984": 129.042594,  # Gln deamidation: 128.058578 +  0.984016
    # N-terminal modifications.
    "+42.011": 42.010565,  # Acetylation
    "+43.006": 43.005814,  # Carbamylation
    "-17.027": -17.026549,  # NH3 loss
    "+43.006-17.027": 25.980265,  # Carbamylation and NH3 loss
}

N_TERMINAL_MSKB = [
    "+42.011",  # Acetylation
    "+43.006",  # Carbamylation
    "-17.027",  # NH3 loss
    "+43.006-17.027",  # Carbamylation and NH3 loss
]

RESIDUES_NINESPECIES = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C_+57.02": 160.030649,  # 103.009185 + 57.021464
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    # Amino acid modifications.
    "M_+15.99": 147.035400,  # Met oxidation:   131.040485 + 15.994915
    "N_+.98": 115.026943,  # Asn deamidation: 114.042927 +  0.984016
    "Q_+.98": 129.042594,  # Gln deamidation: 128.058578 +  0.984016
    # N-terminal modifications.
    "_+42.011": 42.010565,  # Acetylation
    "_+43.006": 43.005814,  # Carbamylation
    "_-17.027": -17.026549,  # NH3 loss
    "_+43.006-17.027": 25.980265,  # Carbamylation and NH3 loss
}

N_TERMINAL_NINESPECIES = [
    "_+42.011",  # Acetylation
    "_+43.006",  # Carbamylation
    "_-17.027",  # NH3 loss
    "_+43.006-17.027",  # Carbamylation and NH3 loss
]

RESIDUES_NINESPECIES_IDAI = {
    "G": 57.021464,
    "A": 71.037114,
    "S": 87.032028,
    "P": 97.052764,
    "V": 99.068414,
    "T": 101.047670,
    "C(+57.02)": 160.030649,  # 103.009185 + 57.021464
    "L": 113.084064,
    "I": 113.084064,
    "N": 114.042927,
    "D": 115.026943,
    "Q": 128.058578,
    "K": 128.094963,
    "E": 129.042593,
    "M": 131.040485,
    "H": 137.058912,
    "F": 147.068414,
    "R": 156.101111,
    "Y": 163.063329,
    "W": 186.079313,
    "O": 261.1915, # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3070376/
    # Amino acid modifications.
    "M[15.9949]": 147.035400,  # Met oxidation:   131.040485 + 15.994915
    "N[0.9840]": 115.026943,  # Asn deamidation: 114.042927 +  0.984016
    "Q[0.9840]": 129.042594,  # Gln deamidation: 128.058578 +  0.984016
    # N-terminal modifications.
    "[0.9840]": 0.984016, # Deamidation
    "[15.9949]": 15.994915, # Oxidation
    "[42.0106]": 42.010565,  # Acetylation
    "[43.0058]": 43.005814,  # Carbamylation
    "[-17.0265]": -17.026549,  # NH3 loss
    "[25.9803]": 25.980265,  # Carbamylation and NH3 loss
}


def get_token_dicts_mskb(amod_dict, include_hidden=False):
    amod_dict = deepcopy(amod_dict)
    amod_dict["X"] = len(amod_dict)  # Pad token

    input_dict = deepcopy(amod_dict)
    input_dict["<SOS>"] = len(amod_dict)
    if include_hidden:
        input_dict["<H>"] = len(input_dict)

    output_dict = deepcopy(amod_dict)
    output_dict["<EOS>"] = len(amod_dict)
    return {
        "residues": RESIDUES_MSKB,
        "amod_dict": amod_dict,
        "input_dict": input_dict,
        "output_dict": output_dict,
    }


def get_lance_data_module(
    data_root_dir,
    batch_size,
    max_peaks=300,
    seed=0,
    include_test=False,
):
    collate_fn = partial(pad_peaks, max_peaks=max_peaks)
    return LanceDataModule(
        data_root_dir, batch_size, collate_fn, seed=seed, include_test=include_test
    )


def get_mskb_data_module(
    data_root_dir,
    batch_size,
    max_peaks=300,
    max_length=30,
    seed=0,
    include_hidden=False,
):

    tokenizer = MSKBTokenizer(RESIDUES_MSKB, n_terminal=N_TERMINAL_MSKB)
    token_dicts = get_token_dicts_mskb(tokenizer.index, include_hidden=include_hidden)

    collate_fn = partial(
        pad_peptides,
        max_peaks=max_peaks,
        max_length=max_length,
        null_token_idx=token_dicts["amod_dict"]["X"],
        tokenizer=tokenizer,
        label_name="sequence",
    )

    return (
        LanceDataModule(data_root_dir, batch_size, collate_fn, seed=seed),
        token_dicts,
    )


def get_ninespecies_data_module(
    data_root_dir,
    ds_config,
    global_args,
    max_peaks=300,
    max_length=30,
    subset=0,
    include_hidden=False,
):
    path_dict = ds_config["paths"]

    for split in ["train", "val", "test"]:
        path_list = path_dict[split]
        for i, path in enumerate(path_list):
            path_list[i] = os.path.join(data_root_dir, path)
        path_dict[split] = path_list

    parser = PeptideParser(ds_config)
    dfs, token_dicts = parser.get_data(include_hidden=include_hidden)
    token_dicts["residues"] = RESIDUES_NINESPECIES
    amod_dict = token_dicts["amod_dict"]
    dataset_train = PeptideDataset(dfs["train"], amod_dict)
    dataset_val = PeptideDataset(dfs["val"], amod_dict)
    dataset_test = PeptideDataset(dfs["test"], amod_dict)

    if subset:
        #assert subset >= 0 and subset <= 1
        #dataset_train, dataset_val, dataset_test = [
        #    Subset(dataset, np.arange(int(len(dataset) * subset)))
        #    for dataset in [dataset_train, dataset_val, dataset_test]
        #]
        dataset_train = Subset(dataset_train, np.arange(int(len(dataset_train) * subset)))

    data_module = NinespeciesDataModule(
        (dataset_train, dataset_val, dataset_test),
        batch_size=(
            ds_config[global_args.downstream_task]["batch_size"]
            if global_args.batch_size < 0
            else global_args.batch_size
        ),
        num_workers=(
            ds_config["num_workers"]
            if global_args.num_workers < 0
            else global_args.num_workers
        ),
        collate_fn=partial(
            pad_peptides,
            max_peaks=max_peaks,
            max_length=max_length,
            null_token_idx=amod_dict["X"],
        ),
        pin_mem=global_args.pin_mem,
    )

    return (data_module, token_dicts)

def get_ninespecies_HF_data_module(
    data_root_dir,
    ds_config,
    global_args,
    #max_peaks=300,
    #max_length=30,
    #max_charge=10,
    subset=0,
    include_hidden=False,
):
    
    never_changing_path = "/proj/bedrock/datasets/9_species_InstaDeepAI"
    
    dataset_directory = os.path.join(never_changing_path, "parquet/processed")
    dictionary_path = os.path.join(never_changing_path, "ns_dictionary.txt")
    loader = LoaderHF(
        dataset_directory=dataset_directory,
        val_species=ds_config['val_species'],
        dictionary_path=dictionary_path,
        top_peaks=ds_config['top_peaks'],
        num_workers=global_args.num_workers,#ds_config['num_workers'],
        pep_length=ds_config['pep_length'],
        charge=ds_config['charge'],
        buffer_size=ds_config['buffer_size'],
    )
    dfs = {
        'train': loader.dataset['train'],
        'val': loader.dataset['val'],
        'test': loader.dataset['test'],
    }

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

    token_dicts["residues"] = RESIDUES_NINESPECIES_IDAI
    amod_dict = token_dicts["amod_dict"]
    dataset_train = loader.dataset['train']
    dataset_val = loader.dataset['val']
    dataset_test = loader.dataset['test']

    if subset:
        #assert subset >= 0 and subset <= 1
        #dataset_train, dataset_val, dataset_test = [
        #    Subset(dataset, np.arange(int(len(dataset) * subset)))
        #    for dataset in [dataset_train, dataset_val, dataset_test]
        #]
        dataset_train = Subset(dataset_train, np.arange(int(len(dataset_train) * subset)))

    data_module = NinespeciesDataModule(
        (dataset_train, dataset_val, dataset_test),
        batch_size=(
            ds_config[global_args.downstream_task]["batch_size"]
            if global_args.batch_size < 0
            else global_args.batch_size
        ),
        num_workers=(
            ds_config["num_workers"]
            if global_args.num_workers < 0
            else global_args.num_workers
        ),
        collate_fn=collate_fn,
        pin_mem=global_args.pin_mem,
        shuffle=False,
    )

    return (data_module, token_dicts)

def configure_callbacks(
    args, task_args, val_metric_name: str = "val_loss", metric_mode="min"
):
    callbacks = []
    filename = f"{{epoch}}-{{{val_metric_name}:.2f}}"
    #filename = f"{{epoch}}-{{step}}"
    # Checkpoint callback
    if not args.barebones:
        callbacks += [
            ModelCheckpoint(
                dirpath=args.output_dir,
                filename=filename,
                monitor=val_metric_name,  # requires that we log something called val_metric_name
                mode=metric_mode,
                save_top_k=args.save_top_k,
                save_last=args.save_last,
                every_n_epochs=args.every_n_epochs,
                #every_n_train_steps=args.every_n_train_steps,
            )
        ]

    if task_args["anneal_lr"]:
        # Cosine annealing LR with warmup callback
        callbacks += [
            CosineAnnealLRCallback(
                lr_start=task_args["lr_start"],
                blr=task_args["blr"],
                lr_end=task_args["lr_end"],
                warmup_duration=task_args["warmup_duration"],
                anneal_per_step=task_args["anneal_per_step"],
            )
        ]

    # measure FLOPs on the first train batch
    if args.profile_flops:
        callbacks += [FLOPProfilerCallback()]

    if args.early_stop > 0:
        callbacks += [EarlyStopping(val_metric_name, patience=args.early_stop)]

    return callbacks


def get_rank() -> int:
    rank_keys = ("RANK", "SLURM_PROCID", "LOCAL_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0


def get_num_parameters(model):
    sum = 0
    for param in list(model.parameters()):
        sum += param.numel()
    return sum


def partition_seq(seq, collect_mods=False):
    Seq = []
    if collect_mods:
        mods = []
    p = 0
    while p < len(seq):
        aa = seq[p]
        if aa == "(":
            let = seq[p - 1]
            end = seq[p:].find(")")
            mod = seq[p + 1 : p + end]
            if collect_mods:
                mods.append(mod)
            p += end
            aa = "%c_%s" % (let, mod)
            Seq[-1] = aa
        else:
            Seq.append(aa)
        p += 1
    output = {"seq": Seq}
    if collect_mods:
        output["mods"] = mods

    return output

def partition_modified_sequence(modseq):
    
    NotAA = lambda x: (x < 65) | (x > 90)

    sequence = []
    p = 0

    #if modseq == '.N[0.9840]AINIEELFQGISR.':
    #    print()
    while p < len(modseq):
        character = modseq[p]
        hx = ord(character)

        # Pull out mod, in the form of a floating point number
        if NotAA(hx):
            mod_lst = []

            # N-terminal modifications precede the amino acid letter
            nterm = True if p == 2 else False

            # All numerals and mathematical symbols are below 65
            while NotAA(hx):
                mod_lst.append(character)
                p += 1

                # This will happen if we have a C-term modification
                if p == len(modseq):
                    break
                else:
                    character = modseq[p]
                    hx = ord(character)
            mod = "".join(mod_lst)

            # Get rid of absent terminal modifications, represented as period
            if mod == '.':
                continue
            elif mod[-1] == '.':
                mod = mod[:-1]

            # Add the amino acid to the end of the number if N-term
            if nterm:# & (mod != "(+57.02)"):
                # Leave the 57.02 with C
                if "(+57.02)" in mod:
                    sequence[0] = sequence[0] + "(+57.02)"
                    mod = mod[8:]

                # The modification stands as its own token at the beginning
                if len(mod) > 0:
                    sequence.insert(0, mod)

            # Grab the previously stored sequence AA and add modification to it
            else:
                sequence[-1] += mod

            p -= 1

        else:
            sequence.append(character)

        #if "" in sequence:
        #    print(sequence)

        p += 1

    return sequence

class Scale:
    masses = {
        "A": 71.037113805,
        "R": 156.101111050,
        "N": 114.042927470,
        "D": 115.026943065,
        "C": 103.009184505,
        "Q": 128.058577540,
        "E": 129.042593135,
        "G": 57.021463735,
        "H": 137.058911875,
        "I": 113.084064015,
        "L": 113.084064015,
        "K": 128.094963050,
        "M": 131.040484645,
        "F": 147.068413945,
        "P": 97.052763875,
        "S": 87.032028435,
        "T": 101.047678505,
        "W": 186.079312980,
        "Y": 163.063328575,
        "V": 99.068413945,
    }

    def __init__(self, amod_dict):
        self.amod_dict = amod_dict
        int2mass = np.zeros((len(amod_dict)))
        for aa, integer in amod_dict.items():
            if len(aa.split("_")) == 2:
                aa, modwt = aa.split("_")
                int2mass[integer] = self.masses[aa] + eval(modwt)
            else:
                if aa in self.masses.keys():
                    int2mass[integer] = self.masses[aa]
                else:
                    int2mass[integer] = 0

        self.tok2mass = {key: int2mass[amod_dict[key]] for key in amod_dict.keys()}
        self.mp = torch.tensor(int2mass, dtype=torch.float32)
        """self.mp = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                list(int2mass.keys()), list(int2mass.values()),
                key_dtype=tf.int64, value_dtype=tf.float32
            ), num_oov_buckets=1
        )"""

    def intseq2mass(self, intseq):
        return torch.gather(self.mp, 0, intseq).sum(1)

    def modseq2mass(self, modified_sequence):
        return np.sum(
            self.tok2mass[tok] for tok in partition_seq(modified_sequence)["seq"]
        )


if __name__ == "__main__":
    ### Test code
    data_dir = "/Users/alfred/Documents/Datasets/instanovo_data_subset"
    lance_dir = "/Users/alfred/Documents/Datasets/instanovo_data_subset/indexed.lance"
    mdsaved_dir = "//Users/alfred/Documents/Datasets/instanovo_data_subset/mdsaved"

    datasets = get_spectrum_dataset_splits(data_dir)
    print(next(iter(datasets[0])))
    print(len(datasets[0]))
    print(len(datasets[1]))
    print(len(datasets[2]))
