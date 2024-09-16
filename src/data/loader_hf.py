from datasets import load_dataset
from torch.utils.data import DataLoader
import torch as th
import os
import re
from copy import deepcopy
import sys

def map_fn(example, tokenizer, dic=None, top=100, max_seq=50):
    ab = th.tensor(example['intensity_array'])
    ab_sort = (-ab).argsort()[:top]
    ab = ab[ab_sort]
    ab /= ab.max()
    spectrum_length = len(ab)
    mz = th.tensor(example['mz_array'])[ab_sort]
    mz_sort = mz.argsort()
    length = len(mz)
    mz_ = th.zeros(top)
    mz_[:len(mz_sort)] = mz[mz_sort]
    ab_ = th.zeros(top)
    ab_[:len(ab_sort)] = ab[mz_sort]
    example['mz_array'] = mz_
    example['intensity_array'] = ab_
    example['precursor_charge'] = th.tensor(example['precursor_charge'], dtype=th.int32)
    example['precursor_mass'] = th.tensor(example['precursor_mass'], dtype=th.float32)
    example['spectrum_length'] = th.tensor(len(example['mz_array']), dtype=th.int32)
    tokenized_sequence = tokenizer(example['modified_sequence'])
    peptide_length = len(tokenized_sequence)
    example['tokenized_sequence'] = th.tensor([dic[m] for m in tokenized_sequence] + (max_seq-peptide_length)*[dic['X']], dtype=th.int32)
    example['peptide_length'] = th.tensor(peptide_length, dtype=th.int32)
    example['spectrum_length'] = th.tensor(spectrum_length, dtype=th.int32)
    
    return example

def collate_fn(batch_list):
    species = [m['experiment_name'] for m in batch_list]
    speclen = th.stack([m['spectrum_length'] for m in batch_list])
    mz = th.stack([m['mz_array'][:speclen.max()] for m in batch_list])
    ab = th.stack([m['intensity_array'][:speclen.max()] for m in batch_list])
    charge = th.stack([m['precursor_charge'] for m in batch_list])
    mass = th.stack([m['precursor_mass'] for m in batch_list])
    length = th.stack([m['spectrum_length'] for m in batch_list])
    peplen = th.stack([m['peptide_length'] for m in batch_list])
    intseq = th.stack([m['tokenized_sequence'][:peplen.max()] for m in batch_list])

    out = {
        'experiment_name': species,
        'mz_array': mz,
        'intensity_array': ab,
        'charge': charge,
        'mass': mass,
        'peak_lengths': length,
        'intseq': intseq,
        'peptide_lengths': peplen[:,None],
        'spectrum_lengths': speclen[:,None],
    }

    return out

class LoaderHF:
    def __init__(self, 
        dataset_directory: str,
        val_species: str,
        dictionary_path: str=None,
        add_start_token: bool=True,
        tokenizer_path: str=None,
        top_peaks: int=100,
        batch_size: int=100,
        num_workers: int=0,
        **kwargs
    ):
        val_species = re.sub('-', '_', val_species).lower()
        assert val_species in [
            'apis_mellifera',
            'homo_sapiens',
            'bacillus_subtilis',
            'candidatus_endoloripes',
            'methanosarcina_mazei',
            'mus_musculus',
            'saccharomyces_cerevisiae',
            'solanum_lycopersicum',
            'vigna_mungo',
        ], "val_species not in our list"

        # Dictionary
        if dictionary_path is not None:
            self.amod_dic = {
                line.split()[0]:m for m, line in enumerate(open(dictionary_path))
            }
            self.amod_dic['X'] = len(self.amod_dic)
            self.amod_dic_rev = {b:a for a,b in self.amod_dic.items()}
            self.input_dic = deepcopy(self.amod_dic)
            if add_start_token:
                self.input_dic["<SOS>"] = len(self.amod_dic)

        # Tokenizer
        sys.path.append(tokenizer_path)
        from enumerate_tokens import partition_modified_sequence
        self.tokenizer = partition_modified_sequence
        
        # Dataset
        dataset_path = {
            'train': os.path.join(dataset_directory, '*'),
            'val': os.path.join(dataset_directory, '*%s*'%val_species),
        }
        dataset = load_dataset(
            'parquet',
            data_files=dataset_path,
            streaming=True
        )
        # Filter out withheld validation data from training
        dataset['train'] = dataset['train'].filter(
            lambda example:
            re.sub("-", "_", example['experiment_name']).lower() != "%s"%val_species
        )
        
        # Map to format outputs
        dataset = dataset.map(
            lambda example:
            map_fn(
                example,
                self.tokenizer,
                self.amod_dic,
                top=top_peaks, 
                max_seq=kwargs['pep_length'][1]
            ), 
            remove_columns=[
                'evidence_index', 
                'scan_number',
                'sequence',
                'precursor_mz',
                'retention_time',
                'title',
                'scans'
            ]
        )

        # Filter for length
        if 'pep_length' in kwargs.keys():
            dataset = dataset.filter(
                lambda example: 
                (example['peptide_length'] >= kwargs['pep_length'][0]) &
                (example['peptide_length'] <= kwargs['pep_length'][1])
            )
        # Filter for charge
        if 'charge' in kwargs.keys():
            dataset = dataset.filter(
                lambda example:
                (example['precursor_charge'] >= kwargs['charge'][0]) &
                (example['precursor_charge'] <= kwargs['charge'][1])
            )
        # Split val into val/test
        dataset['test'] = dataset['train'].filter(lambda x, idx: idx % 10 == 0, with_indices=True)
        #dataset['test'] = dataset['val'].filter(lambda x, idx: idx % 2 == 0, with_indices=True)
        #dataset['val'] = dataset['val'].filter(lambda x, idx: idx % 2 == 1, with_indices=True)
        
        # Shuffle the dataset
        if 'buffer_size' in kwargs.keys():
            dataset['train'] = dataset['train'].shuffle(buffer_size=kwargs['buffer_size'])
        #else:
        #    dataset['train'] = dataset['train'].shuffle()
        
        self.dataset = dataset

        # Dataloaders
        #num_workers = min(self.dataset['train'].n_shards, num_workers)
        #self.dataloader = {
        #    'train': self.build_dataloader(dataset['train'], batch_size, num_workers),
        #    'val':   self.build_dataloader(dataset['val']  , batch_size, 0),
        #}

    def build_dataloader(self, dataset, batch_size, num_workers):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )

"""
loader = LoaderHF(
    dataset_directory='parquet/processed/',
    val_species='homo-sapiens',
    dictionary_path="ns_dictionary.txt",
    top_peaks=100,
    num_workers=16,
    pep_length=[0, 30],
    buffer_size=10000,
)
from time import time
start = time()
print("Clock started")
for i, batch in enumerate(loader.dataloader['val']):
    if i==9:
        start = time()
    if i == 1000:
        break
    print("\r%d"%i, end="")
print()
full_time = time() - start
print("Spectra per second: ", (990*100) / full_time)
"""
