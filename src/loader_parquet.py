from copy import deepcopy
import pandas as pd
import numpy as np
import torch as th
from tqdm import tqdm
from torch.utils.data import Dataset

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


class Scale:
    def __init__(self, amod_dict):
        self.amod_dict = amod_dict
        int2mass = np.zeros((len(amod_dict)))
        for aa, integer in amod_dict.items():
            if len(aa.split("_")) == 2:
                aa, modwt = aa.split("_")
                int2mass[integer] = masses[aa] + eval(modwt)
            else:
                if aa in masses.keys():
                    int2mass[integer] = masses[aa]
                else:
                    int2mass[integer] = 0

        self.tok2mass = {key: int2mass[amod_dict[key]] for key in amod_dict.keys()}
        self.mp = th.tensor(int2mass, dtype=th.float32)
        """self.mp = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                list(int2mass.keys()), list(int2mass.values()),
                key_dtype=tf.int64, value_dtype=tf.float32
            ), num_oov_buckets=1
        )"""

    def intseq2mass(self, intseq):
        return th.gather(self.mp, 0, intseq).sum(1)

    def modseq2mass(self, modified_sequence):
        return np.sum(
            self.tok2mass[tok] for tok in partition_seq(modified_sequence)["seq"]
        )


class PeptideParser:
    def __init__(self, config):
        # configuration is loader subsection of downstream yaml file
        self.config = config

        self.dfs = {}
        for key in config["paths"].keys():
            self.dfs[key] = self.load_parquet(config["paths"][key])

        self.filter_length(config["pep_length"][1])
        self.maxsl = config["pep_length"][1]  # + 1 # plus start token
        self.filter_charge(config["charge"][0], config["charge"][1])
        self.token_dict()
        self.input_dict = deepcopy(self.amod_dic)
        self.input_dict["<SOS>"] = len(self.amod_dic)
        self.output_dict = deepcopy(self.amod_dic)
        self.output_dict["<EOS>"] = len(self.amod_dic)

        self.scale = Scale(self.amod_dic)
        self.inds = {}
        for key in self.dfs.keys():
            self.inds[key] = np.arange(len(self.dfs[key]))

        if config["add_mass"]:
            self.add_mass_to_dfs()

    def add_mass_to_dfs(self):
        for key in self.dfs.keys():
            # self.dfs[key]['mass'] = [
            #    self.scale.modseq2mass(seq) for seq in
            #    self.dfs[key].modified_sequence
            # ]
            self.dfs[key]["mass"] = (
                self.dfs[key]["precursor_mz"] * self.dfs[key]["precursor_charge"]
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
            boolean = (self.dfs[key].precursor_charge >= minch) & (
                self.dfs[key].precursor_charge <= maxch
            )
            boolean = np.array(boolean)
            self.dfs[key] = self.dfs[key].iloc[boolean]

    def token_dict(self):
        collect = []
        collect_mods = []
        for split, dset in self.dfs.items():
            for seq in tqdm(
                dset.modified_sequence, desc=f"Finding all aa tokens ('{split}' split)"
            ):
                ps = partition_seq(seq, collect_mods=True)
                for aa in ps["seq"]:
                    collect.append(aa)
                for mod in ps["mods"]:
                    collect_mods.append(mod)
        uniq = np.unique(collect)
        # enumerate tokens
        self.amod_dic = {n: m for m, n in enumerate(uniq)}
        # add null token
        self.amod_dic["X"] = len(self.amod_dic)
        self.rdic = {n: m for m, n in self.amod_dic.items()}
        self.mod_types = np.unique(collect_mods)

        print("Found %d aa-mod combinations." % (len(self.amod_dic) - 1))

    def get_data(self):
        token_dicts = {
            "amod_dict": self.amod_dic,
            "input_dict": self.input_dict,
            "output_dict": self.output_dict,
        }
        return self.dfs, token_dicts


class PeptideDataset(Dataset):
    def __init__(self, df, amod_dict, add_mass=True, dtype=th.float32) -> None:
        super().__init__()
        self.df = df
        self.amod_dict = amod_dict
        self.add_mass = add_mass
        self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def tokenize_seq(self, seq):
        lst = partition_seq(seq)
        tok = [self.amod_dict[m] for m in lst["seq"]]
        return tok

    def __getitem__(self, index):
        row = self.df.iloc[index]
        intseq = self.tokenize_seq(row.modified_sequence)
        output = {
            "mz_array": th.tensor(row.mz_array, dtype=self.dtype),
            "intensity_array": th.tensor(row.intensity_array, dtype=self.dtype),
            "charge": row.precursor_charge,
            "mass": row.mass if self.add_mass else None,
            "intseq": th.tensor(intseq, dtype=th.long),
        }

        return output

    # def load_batch(self, indices, dset="train", top=None, SeqInts=False):
    #     bs = len(indices)
    #     top = self.config["top_pks"] if top == None else top
    #     maxpl = self.config["pep_length"][1]

    #     mz = np.zeros((bs, top))
    #     ab = np.zeros((bs, top))
    #     charge = np.zeros((bs,))
    #     lengths = np.zeros((bs,))
    #     pep_lengths = np.zeros((bs,))
    #     mass = np.zeros((bs,))
    #     seqints = np.zeros((bs, self.maxsl), np.int32)
    #     for i, index in enumerate(indices):
    #         spec_dic = self.read_spec(index, dset)

    #         marg = np.argsort(spec_dic["ab"])[-top:]
    #         mzsort = np.argsort(spec_dic["mz"][marg])

    #         mz[i, : len(marg)] = spec_dic["mz"][marg][mzsort]
    #         ab[i, : len(marg)] = spec_dic["ab"][marg][mzsort]
    #         charge[i] = spec_dic["charge"]
    #         lengths[i] = len(mzsort)
    #         pep_lengths[i] = len(spec_dic["intseq"])
    #         if self.config["add_mass"]:
    #             mass[i] = spec_dic["mass"]
    #         if SeqInts:
    #             seqints[i] = np.array(
    #                 spec_dic["intseq"]
    #                 + (maxpl - len(spec_dic["intseq"])) * [self.amod_dic["X"]]
    #             )

    #     output = {
    #         "mz": th.tensor(mz, dtype=th.float32),
    #         "ab": th.tensor(ab / ab.max(-1, keepdims=True), dtype=th.float32),
    #         "charge": th.tensor(charge, dtype=th.int32),
    #         "mass": (
    #             th.tensor(mass, dtype=th.float32) if self.config["add_mass"] else None
    #         ),
    #         "length": th.tensor(lengths, dtype=th.int32),
    #         "seqint": th.tensor(seqints, dtype=th.int32),
    #         "peplen": th.tensor(pep_lengths, dtype=th.int32),
    #     }

    #     return output


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
