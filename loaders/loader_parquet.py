

import pandas as pd
import numpy as np
import torch as th
from utils import Scale, partition_seq

class LoaderDS:
    def __init__(self, config):
        # configuration is loader subsection of downstream yaml file
        self.config = config
        
        self.dfs = {}
        for key in config['paths'].keys():
            self.dfs[key] = self.load_parquet(config['paths'][key])
        
        self.filter_length(config['pep_length'][1])
        self.maxsl = config['pep_length'][1]# + 1 # plus start token
        self.filter_charge(config['charge'][0], config['charge'][1])
        self.token_dict()
        
        self.scale = Scale(self.amod_dic)
        self.inds = {}
        for key in self.dfs.keys():
            self.inds[key] = np.arange(len(self.dfs[key]))
        
        if config['add_mass']:
            self.add_mass_to_dfs()

    def add_mass_to_dfs(self):
        for key in self.dfs.keys():
            #self.dfs[key]['mass'] = [
            #    self.scale.modseq2mass(seq) for seq in 
            #    self.dfs[key].modified_sequence
            #]
            self.dfs[key]['mass'] = (
                self.dfs[key]['precursor_mz']*self.dfs[key]['precursor_charge']
            )

    def load_parquet(self, fullpaths):
        df = pd.concat([pd.read_parquet(path) for path in fullpaths])

        return df

    def filter_length(self, maxlen):
        for key in self.dfs.keys():
            lengths = np.vectorize(len)(self.dfs[key].sequence)
            boolean = lengths < maxlen
            self.dfs[key] = self.dfs[key].iloc[boolean]

    def filter_charge(self, minch, maxch):
        for key in self.dfs.keys():
            boolean = (
                (self.dfs[key].precursor_charge >= minch) &
                (self.dfs[key].precursor_charge <= maxch)
            )
            boolean = np.array(boolean)
            self.dfs[key] = self.dfs[key].iloc[boolean]
    
    def tokenize_seq(self, seq):
        lst = partition_seq(seq)
        tok = [self.amod_dic[m] for m in lst['seq']]

        return tok

    def token_dict(self):
        collect = []
        collect_mods = []
        for dset in self.dfs.values():
            for seq in dset.modified_sequence:
                ps = partition_seq(seq, collect_mods=True)
                for aa in ps['seq']: collect.append(aa)
                for mod in ps['mods']: collect_mods.append(mod)
        uniq = np.unique(collect)
        # enumerate tokens
        self.amod_dic = {n:m for m,n in enumerate(uniq)}
        # add null token
        self.amod_dic['X'] = len(self.amod_dic)
        self.rdic = {n:m for m,n in self.amod_dic.items()}
        self.mod_types = np.unique(collect_mods)
        
        print("Found %d aa-mod combinations."%(len(self.amod_dic)-1))
    
    def read_spec(self, index=0, dset='train'):
        row = self.dfs[dset].iloc[index]
        intseq = self.tokenize_seq(row.modified_sequence)
        output = {
            'mz': row.mz_array,
            'ab': row.intensity_array,
            'charge': row.precursor_charge,
            'mass': row.mass if self.config['add_mass'] else None,

            'intseq': intseq,
        }

        return output
    
    def load_batch(self, indices, dset='train', top=None, SeqInts=False):
        bs = len(indices)
        top = self.config['top_pks'] if top==None else top
        maxpl = self.config['pep_length'][1]
        
        mz = np.zeros((bs, top))
        ab = np.zeros((bs, top))
        charge = np.zeros((bs,))
        lengths = np.zeros((bs,))
        pep_lengths = np.zeros((bs,))
        mass = np.zeros((bs,))
        seqints = np.zeros((bs, self.maxsl), np.int32)
        for i, index in enumerate(indices):
            spec_dic = self.read_spec(index, dset)
            
            marg = np.argsort(spec_dic['ab'])[-top:]
            mzsort = np.argsort(spec_dic['mz'][marg])

            mz[i, :len(marg)] = spec_dic['mz'][marg][mzsort]
            ab[i, :len(marg)] = spec_dic['ab'][marg][mzsort]
            charge[i] = spec_dic['charge']
            lengths[i] = len(mzsort)
            pep_lengths[i] = len(spec_dic['intseq'])
            if self.config['add_mass']:
                mass[i] = spec_dic['mass']
            if SeqInts:
                seqints[i] = np.array(
                    spec_dic['intseq'] +
                    (maxpl-len(spec_dic['intseq']))*[self.amod_dic['X']]
                )
            
        output = {
            'mz': th.tensor(mz, dtype=th.float32),
            'ab': th.tensor(ab/ab.max(-1, keepdims=True), dtype=th.float32),
            'charge': th.tensor(charge, dtype=th.int32),
            'mass': (
                th.tensor(mass, dtype=th.float32) 
                if self.config['add_mass'] else None
            ),
            'length': th.tensor(lengths, dtype=th.int32),
            'seqint': th.tensor(seqints, dtype=th.int32),
            'peplen': th.tensor(pep_lengths, dtype=th.int32)
        }

        return output
"""
config = {
    'paths': {
        'train': [
            '/cmnfs/data/proteomics/foundational_model/ninespecies_xy/train-00001-of-00002-fb1cb1b5c4a4ef4f.parquet',
            '/cmnfs/data/proteomics/foundational_model/ninespecies_xy/train-00000-of-00002-ca1fbc3de7c99259.parquet'
        ],
        'val': ['/cmnfs/data/proteomics/foundational_model/ninespecies_xy/validation-00000-of-00001-b84568f5bf3ba95d.parquet'],
        'test': ['/cmnfs/data/proteomics/foundational_model/ninespecies_xy/test-00000-of-00001-15be7bb3864037e7.parquet'],
    },
    'pep_length': [0, 30],
    'charge': [0, 10],
    'top_pks': 100,
    'add_mass': True
}
L = LoaderDS(config)
spec = L.load_batch(np.arange(100))
print(spec)
"""
