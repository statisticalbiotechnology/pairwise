import numpy as np
import tensorflow as tf
import os
import pandas as pd
import re
import path

def gather_file_md(filepath, typ=None):
    if typ==None:
        typ = filepath.split('.')[-1].strip().lower()

    with open(filepath) as f:
        _ = f.read()
        end = f.tell()
        f.seek(0)
        
        pos = f.tell()
        pos_prev = f.tell()
        spectra = {}
        spec_ticker = 0
        while pos!=end:

            line = f.readline().strip()
            pos = f.tell()
            
            if typ=='mgf':
                if line == 'BEGIN IONS':
                    spectra[spec_ticker] = {}
                elif line.split('=')[0] == 'SCANS':
                    scan = int(line.split('=')[-1])
                    spectra[spec_ticker]['scan'] = scan
                elif line.split('=')[0] == 'RTINSECONDS':
                    rt = float(line.split('=')[-1])
                    spectra[spec_ticker]['rt'] = rt
                elif line.split('=')[0] == 'CHARGE':
                    charge = int(line.split('=')[-1][:-1])
                    spectra[spec_ticker]['charge'] = charge
                elif line.split('=')[0] == 'PEPMASS':
                    mass = float(line.split('=')[-1])
                    spectra[spec_ticker]['mass'] = mass
                elif len(line.split('.')) == 3:
                    peak_ticker = 0
                    spectra[spec_ticker]['pos'] = pos_prev
                    while line != 'END IONS':
                        peak_ticker += 1
                        line = f.readline().strip()
                    spectra[spec_ticker]['nmpks'] = peak_ticker

                    assert len(spectra[spec_ticker].keys()) == 6
                    spec_ticker += 1
                
                pos_prev = pos
            
            elif typ=='msp':

                # Start of a spectrum entry: label
                # - assume labels are {seq}/{charge}_{mods}_{ev}eV_NCE{nce}
                if line[:5]=='Name:':
                    spectra[spec_ticker] = {}
                    spectra[spec_ticker]['label'] = line.split()[-1]
                    seq, other = line.split()[-1].split('/')
                    spectra[spec_ticker]['seq'] = seq
                    charge, mods, ev, nce = other.split('_')
                    spectra[spec_ticker]['charge'] = int(charge)
                    spectra[spec_ticker]['ev'] = float(ev[:-2])
                    spectra[spec_ticker]['nce'] = float(nce[3:])
                    
                    # parsing mod
                    spectra[spec_ticker]['mod_label'] = mods
                    spectra[spec_ticker]['mod_pos'] = []
                    spectra[spec_ticker]['mod_name'] = []
                    spectra[spec_ticker]['mod_aa'] = []
                    if mods != '0':
                        m0 = mods.find('(')
                        mod_amt = int(mods[:m0])
                        for mod in mods[m0+1:-1].split(')('):
                            pos, aa, name = mod.split(',')
                            spectra[spec_ticker]['mod_pos'].append(int(pos))
                            spectra[spec_ticker]['mod_name'].append(name)
                            spectra[spec_ticker]['mod_aa'].append(aa)
                    # Done with label
                    # Search no more than 10 lines for MW
                    for i in range(10):
                        line = f.readline()
                        if line[:3]=='MW:':
                            spectra[spec_ticker]['mw'] = float(line.split()[-1])
                            break
                    # Search no more than 10 lines for Num peaks
                    for i in range(10):
                        line = f.readline()
                        if line[:10] == 'Num peaks:':
                            nmpks = int(line.split()[-1])
                            spectra[spec_ticker]['nmpks'] = nmpks
                            spectra[spec_ticker]['pos'] = f.tell()
                            for _ in range(nmpks): f.readline()
                            break

                    assert len(spectra[spec_ticker].keys()) == 12
                    spec_ticker += 1
            else:
                NotImplementedError("File type not implemented yet.")

    return spectra

def filter_length(df, len_rng):
    assert hasattr(df, 'seq')
    boolean = (
        (np.vectorize(len)(df['seq'])>=len_rng[0]) &
        (np.vectorize(len)(df['seq'])<=len_rng[1])
    )
    df = df.iloc[boolean]

    return df

def filter_charge(df, ch_rng):
    assert hasattr(df, 'charge')
    boolean = np.array(
        (df['charge']>=ch_rng[0]) &
        (df['charge']<=ch_rng[1])
    )
    df = df.iloc[boolean]
    
    return df

def filter_mod(df, mod_list):
    assert hasattr(df, 'mod_name')
    boolean = []
    for i in range(len(df)):
        pepmods = df.iloc[i]['mod_name']
        tick = True
        for j in pepmods:
            if j not in mod_list:
                boolean.append(False)
                tick=False
                break
        if tick:
            boolean.append(True)
    boolean = np.array(boolean)
    df = df.iloc[boolean]

    return df

class LoadObj:
    def __init__(self, 
                 paths=None, 
                 preopen=True, 
                 mdsaved_path='./mdsaved', 
                 top_pks=100, 
                 save_md=True,
                 filter_psms=False,
                 ):
        # cut out possible mdsaved directory from path search
        self.paths = [path for path in paths if 'evidence' not in path]
        self.preopen = preopen
        self.mdsaved_path = mdsaved_path
        self.top_pks = top_pks
        self.save_md = save_md
        
        if save_md and not os.path.exists(mdsaved_path):
            os.mkdir(mdsaved_path)
        
        self.gather_md()
        if filter_psms:
            self.filter_psms()

        self.gather_labels()
        if self.preopen:
            self.open_files()
    
    def gather_md(self):
        self.md = {}
        self.fn2full = {}
        for path in self.paths:
            # Snip off the filename
            filename = path.split('/')[-1].split('.')[0]
            self.fn2full[filename] = path
            # Create the (potential) full path to the saved md data
            fullpath = '%s/%s_md.pkl'%(self.mdsaved_path, filename)
            if os.path.exists(fullpath):
                df = pd.read_pickle(fullpath)
            else:
                # Read the file
                spec_dic = gather_file_md(path)
                df = pd.DataFrame(spec_dic).transpose() # transpose alters dtypes
                # Save these (2) entries as integers
                df['scan'] = df['scan'].astype("int32")
                df['pos'] = df['pos'].astype("int32")
                df['nmpks'] = df['nmpks'].astype('int32')
                if self.save_md:
                    df.to_pickle('%s/%s_md.pkl'%(self.mdsaved_path, filename))

            self.md[filename] = df
        self.filenames = list(self.md.keys())

    def filter_psms(self):
        for filename in self.md.keys():
            filename

    def gather_labels(self):
        # index.values needs df.loc, enumerate needs df.iloc
        listoflists = [
            ['%s|%d'%(filename,iloc) for iloc, loc in enumerate(self.md[filename].index.values)] 
            for filename in self.md.keys()
        ]
        self.labels = [m for n in listoflists for m in n]

    def open_files(self):
        self.fps = {
            name: open(path) 
            for name, path in zip(self.filenames, self.paths)
        }

    def close_files(self):
        for key in self.fps.keys():
            self.fps[key].close()
    
    def read_spec(self, filename, index, ann=False):
        fp = self.fps[filename] if self.preopen else open(self.fn2full[filename])
        md = self.md[filename] # calling the sample automatically casts dtypes
        fp.seek(md['pos'].iloc[index])

        pks = np.array([
                [float(m) for m in fp.readline().strip().split()] 
                for _ in range(md['nmpks'].iloc[index])
        ])
        #mz, ab = np.split(pks, 2, -1)
        mz = pks[:,0];ab = pks[:,1]
        output = {
                'mz': mz,
                'ab': ab,
                'charge': md['charge'].iloc[index],
                'mass': md['mass'].iloc[index]
        }
        if not self.preopen: fp.close()
        return output

    def load_batch(self, labels, top=None):
        if top==None: top=self.top_pks
        
        mz = np.zeros((len(labels), top))
        ab = np.zeros((len(labels), top))
        charge = np.zeros((len(labels),))
        mass = np.zeros((len(labels),))
        lengths = np.zeros((len(labels),))
        for i, label in enumerate(labels):
            fnm, ind = label.split('|')
            spec_dic = self.read_spec(fnm, int(ind))
                  
            marg = np.argsort(spec_dic['ab'])[-top:]
            mzsort = np.argsort(spec_dic['mz'][marg])
            
            mz[i, :len(marg)] = spec_dic['mz'][marg][mzsort]
            ab[i, :len(marg)] = spec_dic['ab'][marg][mzsort]
            charge[i] = spec_dic['charge']
            mass[i] = spec_dic['mass']
            lengths[i] = len(mzsort)

        output = {
                'mz': tf.constant(mz, tf.float32), 
                'ab': tf.constant(ab/ab.max(-1, keepdims=True), tf.float32),
                'charge': tf.constant(charge, tf.int32),
                'mass': tf.constant(mass, tf.float32),
                'length': tf.constant(lengths, tf.int32)
        }

        return output

###############################################################################
#                           Downstream loaders                                #
# Must change the information that is read from each spectrum and the targets #
# that are created in each batch.                                             #
###############################################################################

class LoadObjDS(LoadObj):
    def __init__(self, config, save_md=True):
        unixspec = list(
            config['datasets']['unixspec'] 
            if 'unixspec' in config['datasets'].keys() else 
            '*'
        )
        paths = [m for n in [
            path.glob.glob(config['datasets']['data_path'] + '/%s'%u)
            for u in unixspec
        ] for m in n]
        
        self.paths = [path for path in paths if 'mdsaved' not in path]
        self.config = config
        self.save_md = save_md
        self.top_pks = config['datasets']['top_pks']
        self.mdsaved_path = config['datasets']['mdsaved_path']
        self.gather_md()

        self.open_files()

class LoadObjDNV(LoadObjDS):
    def __init__(self, config, save_md=True):
        super().__init__(config=config, save_md=save_md)
        
        # Create the aa-mod to integer dictionary
        self.create_aamod_dict()
        # Turn all sequences (with mods) into their corresponding integer seqs
        self.create_intseq()
        # Filter spectra based on sequence length, charge, and modifications
        for key in self.md.keys():
            if 'seq_len' in config.keys():
                self.md[key] = filter_length(self.md[key], config['seq_len'])
            if 'charge' in config.keys():
                self.md[key] = filter_charge(self.md[key], config['charge'])
            if 'mods' in config.keys():
                self.md[key] = filter_mod(self.md[key], config['mods'])

        self.gather_labels()

    def split_labels_str(self, incl_str):
        return [label for label in self.labels if incl_str in label]

    def create_aamod_dict(self):
        amod_dic = []
        # iterate through all dataframes in loader
        for mdf in self.md.values():
            # iterate through all peptides in dataframe
            seqmods = []
            for i in range(len(mdf)):
                # iterate through all positions in peptide
                seqmod = []
                for I, aa in enumerate(mdf.iloc[i]['seq']):
                    # if there is a modification
                    if I in mdf.iloc[i]['mod_pos']:
                        ind = np.where(
                            I==np.array(mdf.iloc[i]['mod_pos'])
                        )[0][0]
                        aa = mdf.iloc[i]['mod_aa'][ind]
                        name = mdf.iloc[i]['mod_name'][ind]
                        amod = "_".join([aa, name])
                    else:
                        amod = aa+'_0'
                    seqmod.append(amod)
                    if amod not in amod_dic:
                        amod_dic.append(amod)
                seqmods.append(seqmod)
            mdf['modseq'] = seqmods
        self.amod_dic = {j:i for i,j in enumerate(np.unique(amod_dic))}
        self.amod_dic['X'] = len(self.amod_dic)
        self.diclen = len(self.amod_dic)
        print("Found %d aa/mod combinations in dataset"%(self.diclen-1))

    def create_intseq(self):
        assert hasattr(self, 'amod_dic')

        # iterate through all dataframes in loader
        for mdf in self.md.values():
            # iterate through all peptides in dataframe
            intseqs = []
            for i in range(len(mdf)):
                intseq = [self.amod_dic[a] for a in mdf.iloc[i]['modseq']]
                intseqs.append(intseq)
            mdf['intseq'] = intseqs

    def read_spec(self, filename, index):
        fp = self.fps[filename]
        md = self.md[filename] # calling the sample automatically casts dtypes
        fp.seek(md['pos'].iloc[index])
        
        Mz = np.zeros((md['nmpks'].iloc[index]))
        Ab = np.zeros((md['nmpks'].iloc[index]))
        Ann = np.empty((md['nmpks'].iloc[index],), dtype='str')
        for m in range(md['nmpks'].iloc[index]):
            mz, ab, ann = fp.readline().strip().split()
            Mz[m] = float(mz)
            Ab[m] = float(ab)
            Ann[m] = ann[1:-1].split('/')[0]
        
        output = {
            'mz': Mz.squeeze(),
            'ab': Ab.squeeze(),
            'ann': Ann.squeeze(),
            'charge': md['charge'].iloc[index],
            'mass': md['mw'].iloc[index],
            
            'intseq': md['intseq'].iloc[index],
            'mod_aa': md['mod_aa'].iloc[index],    
        }

        return output

    def load_batch(self, labels, top=None):
        if top==None: top=self.top_pks
        maxsl = self.config['seq_len'][1]
        
        mz = np.zeros((len(labels), top))
        ab = np.zeros((len(labels), top))
        #ann = np.zeros((len(labels), top))
        charge = np.zeros((len(labels),))
        mass = np.zeros((len(labels),))
        lengths = np.zeros((len(labels),))
        
        seqints = np.zeros((len(labels), maxsl))

        #int_array = np.zeros((len(labels)), 
        for i, label in enumerate(labels):
            fnm, ind = label.split('|')
            spec_dic = self.read_spec(fnm, int(ind))
            
            marg = np.argsort(spec_dic['ab'])[-top:]
            mzsort = np.argsort(spec_dic['mz'][marg])
            
            mz[i, :len(marg)] = spec_dic['mz'][marg][mzsort]
            ab[i, :len(marg)] = spec_dic['ab'][marg][mzsort]
            #ann[i, :len(marg)] = spec_dic['ann'][marg][mzsort]
            charge[i] = spec_dic['charge']
            mass[i] = spec_dic['mass']
            lengths[i] = len(mzsort)

            seqints[i] = np.array(
                spec_dic['intseq'] + 
                (maxsl-len(spec_dic['intseq']))*[self.diclen-1]
            )
        
        #target = tf.one_hot(seq_ints, self.diclen)

        output = {
            'mz': tf.constant(mz, tf.float32), 
            'ab': tf.constant(ab/ab.max(-1, keepdims=True), tf.float32),
            'charge': tf.constant(charge, tf.int32),
            'mass': tf.constant(mass, tf.float32),
            'length': tf.constant(lengths, tf.int32),

            'target': tf.constant(seqints, tf.int32)
        }

        return output

# Spectrum classifier
class LoadObjSC(LoadObjDS):
    def __init__(self, config, save_md=True):
        super().__init__(config=config, save_md=save_md)
        self.config = config
        
        # Add class labels to spectra in self.md
        self.add_labels_to_md()
        
        self.gather_labels()

    def create_label_dict(self):
        self.labeldic = {
            n:m for m,n in enumerate(self.config['datasets']['labels'])
        }

    def add_labels_to_md(self):
        self.create_label_dict()
        
        for filename in self.md.keys():
            for label in self.labeldic.keys():
                if label in filename:
                    self.md[filename]['class'] = (
                        self.labeldic[label] * 
                        np.ones((len(self.md[filename]),), dtype='int')
                    )
                    break

    def read_spec(self, filename, index, ann=False):
        fp = self.fps[filename]
        md = self.md[filename] # calling the sample automatically casts dtypes
        fp.seek(md['pos'].iloc[index])

        pks = np.array([
                [float(m) for m in fp.readline().strip().split()] 
                for _ in range(md['nmpks'].iloc[index])
        ])
        mz, ab = np.split(pks, 2, -1)
        output = {
                'mz': mz.squeeze(),
                'ab': ab.squeeze(),
                'charge': md['charge'].iloc[index],
                'mass': md['mass'].iloc[index],
                
                'class': md['class'].iloc[index]
        }

        return output

    def load_batch(self, labels, top=None):
        if top==None: top=self.top_pks
        
        mz = np.zeros((len(labels), top))
        ab = np.zeros((len(labels), top))
        #ann = np.zeros((len(labels), top))
        charge = np.zeros((len(labels),))
        mass = np.zeros((len(labels),))
        lengths = np.zeros((len(labels),))
        
        classes = np.zeros((len(labels),))

        #int_array = np.zeros((len(labels)), 
        for i, label in enumerate(labels):
            fnm, ind = label.split('|')
            spec_dic = self.read_spec(fnm, int(ind))
            
            marg = np.argsort(spec_dic['ab'])[-top:]
            mzsort = np.argsort(spec_dic['mz'][marg])
            
            mz[i, :len(marg)] = spec_dic['mz'][marg][mzsort]
            ab[i, :len(marg)] = spec_dic['ab'][marg][mzsort]
            #ann[i, :len(marg)] = spec_dic['ann'][marg][mzsort]
            charge[i] = spec_dic['charge']
            mass[i] = spec_dic['mass']
            lengths[i] = len(mzsort)

            classes[i] = spec_dic['class']

        #target = tf.one_hot(classes, self.predcats)

        output = {
            'mz': tf.constant(mz, tf.float32), 
            'ab': tf.constant(ab/ab.max(-1, keepdims=True), tf.float32),
            'charge': tf.constant(charge, tf.int32),
            'mass': tf.constant(mass, tf.float32),
            'length': tf.constant(lengths, tf.int32),

            'target': tf.constant(classes, tf.int32)
        }

        return output
"""
import yaml

with open("yaml/downstream.yaml") as stream:
    config = yaml.safe_load(stream)
config['specclass']['datasets']['top_pks'] = 100

L = LoadObjSC(config['specclass'])
"""
