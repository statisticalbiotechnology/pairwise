###############################################################################
#                                  Todo                                       #
###############################################################################
"""
Add mask charge/mass tasks
Add task where you add m/z to each peak and model must guess systematic error
Add a intensity rank task --> one_hot(x.shape[1]-argsort(argsort(x,-1),-1))
"""
# dependencies used throughout the program
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import yaml
import utils as U
import re
Adam = tf.keras.optimizers.Adam
# slurm doesn't always manage GPUs well -> cublas error
# You may need to set CUDA_VISIBLE_DEVICES={#} before python in shellscript.sh
print(tf.config.list_physical_devices())

# Immediately read in yaml files to prevent unintended changes to the
# experiment while I am changing the files outside of the shellscript
with open("./yaml/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)
with open("./yaml/datasets.yaml", 'r') as stream:
    dc = yaml.safe_load(stream)
with open("./yaml/models.yaml", 'r') as stream:
    mconf = yaml.safe_load(stream)
with open("./yaml/tasks.yaml", 'r') as stream:
    tc = yaml.safe_load(stream)
with open("./yaml/downstream.yaml") as stream:
    dsconfig = yaml.safe_load(stream)
config['lr'] = float(config['lr'])

###############################################################################
#                                  Loader                                     #
###############################################################################

import pathlib
import path
from loaders.loader import LoadObj
from copy import deepcopy

#HOME = str(pathlib.Path.home())
paths = [
    path.glob.glob(fp+'/*') for fp in dc['pretrain']['train_dirs']
]
paths = [m for n in paths for m in n]

L = LoadObj(
    paths, 
    preopen=dc['pretrain']['preopen_files'], 
    mdsaved_path=dc['pretrain']['mdsaved_path'], 
    top_pks=config['max_peaks']
)

labels = deepcopy(L.labels)

###############################################################################
#                                   Model                                     #
###############################################################################

from models import Encoder, Header
from utils import *

# NOTE about trading information between yaml files:
# I don't want to have to specity inputs in 2 different yaml files that are con-
# sistent with each other. Instead, 1 yaml file can have all relevant input spe-
# cifications, and share its information with the configuration of the other.
# For example, specify bin size in yaml/tasks.yaml and use that value to set
# number of bins in hidden mz head model

# Encoder model
encoder_dic = mconf['encoder_dict']
encoder = Encoder(**encoder_dic)

# Header model
header_dict = mconf['header_dict']
mzh = tc['hidden_mz']
header_dict['hidden_mz']['bins'] = int( 
    (mzh['mzlims'][1] - mzh['mzlims'][0]) / mzh['binsz']
) # hidden mz needs binsz and mz range apriori
abh = tc['hidden_ab']
header_dict['hidden_ab']['bins'] = int(1 / abh['binsz']) # so does hidden ab
header_dict['hidden_spectrum']['bins'] = config['max_peaks']
# hidden charge needs max charge to set the number of classes
header_dict['hidden_charge']['num_classes'] = tc['hidden_charge']['max_charge']
header_dict = {task: header_dict[task] for task in config['tasks']}
header = Header(header_dict)

# Call model to create weights
batch = L.load_batch(labels[:config['batch_size']])
mzab = tf.concat([batch['mz'][...,None], batch['ab'][...,None]], -1)
model_inp = {
    'x': mzab,
    'charge': batch['charge'],
    'mass': batch['mass'],
    'length': batch['length'],
    'training': False
}
out = encoder(**model_inp)
head_out = header(out['emb'], 'all')
print("Total encoder parameters: %d"%encoder.total_params())

# Optimizers
optencoder = Adam(learning_rate=config['lr'],name='encopt')
header.initialize_optimizers()

def save_all_weights(svdir):
    U.save_full_model(encoder, optencoder, svdir)
    # Save all header weights in one file
    header.save_weights("save/%s/weights/model_%s.wts"%(svdir, header.name))
    # Save header optimizer weights individually
    for task_name in config['tasks']:
        # optimizer.name should have opt_ already in it (see Header in models)
        fn = 'save/%s/weights/%s.wts'%(svdir, header.opts[task_name].name)
        U.save_optimizer_state(header.opts[task_name], fn)

if config['load']:
    ldpth = config['loadpath']
    encoder.load_weights(ldpth + 'model_encoder.wts')
    U.load_optimizer_state(
        optencoder, encoder.trainable_variables, ldpth + 'opt_encopt.wts.npy'
    )
    header.load_weights(ldpth + 'model_header.wts')
    for task_name in config['tasks']:
        # ASSUMPTION: header optimizers follow name convention 
        # opt_{task}.wts.npy
        U.load_optimizer_state(
            header.opts[task_name], 
            header.heads[task_name].trainable_variables, 
            ldpth + 'opt_%s.wts.npy'%task_name
        )

###############################################################################
#                                    Loss                                     #
###############################################################################

import tasks

# All tasks have loss variables for tracking
T = tasks.all_tasks(tc)
T = {task: T[task] for task in config['tasks']}
loss_spec = " ".join(['%s: %%7.5f'%task_name for task_name in T.keys()])

###############################################################################
#                           Downstream evaluation                             #
###############################################################################
if not config['debug']:
    import downstream as ds

    # Downstream object
    allds = {
        'charge': ds.ChargeDSObj,
        'peplen': ds.PeplenDSObj,
        'denovo': ds.DenovoDSObj,
    }

###############################################################################
#                                  Training                                   #
###############################################################################

from collections import deque
from time import time
import sys
import datetime

def train_step(batch, task, enc_opt, head_opt):
    inp = T[task].inptarg(batch)
    
    inp['length'] = batch['length']
    head_outputs = [task]
    with tf.GradientTape(persistent=True) as tape:
        enc_output = encoder(training=True, **inp)
        prediction = header(enc_output['emb'], head_outputs, training=True)
        loss = T[task].loss(prediction[task])
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, encoder.trainable_variables)
    enc_opt.apply_gradients(zip(grads, encoder.trainable_variables))
    grads = tape.gradient(loss, header.heads[task].trainable_variables)
    head_opt.apply_gradients(zip(grads, header.heads[task].trainable_variables))
    
    encoder.global_step.assign_add(1)

    return loss

def test():
    print()

graph = train_step if config['debug'] else tf.function(train_step)
def train(epochs=1, runlen=50, svfreq=3600):
    
    # Shorthand
    bs = config['batch_size']
    msg = config['log'] & (config['debug']!=True)
    swt = config['svwts'] & (config['debug']!=True)
    # Create experiment directory in save/
    if (msg or swt):
        dt = str(datetime.datetime.now()).split()
        dt[-1] = re.sub(':','-',dt[-1]) # linux has issue with : symbol
        svdir = "_".join(dt)
        os.mkdir('save/%s'%svdir);
        os.mkdir('save/%s/yaml'%svdir)
        os.system("cp ./yaml/*.yaml save/%s/yaml/"%svdir)
        if config['svwts']: save_all_weights(svdir)
    
    # Log starting messages and start collection all lines
    if msg:
        line = "%s\nTotal parameters: %d\n"%(
            config['header'], encoder.total_params()
        )
        U.message_board(line, "save/%s/epochout.txt"%svdir)
        #U.message_board(line, "save/%s/valout.txt"%svdir)
        line = "%s\n%s\n"%(svdir, config['header'])
        allepochlines = [line]
        #allvallines = [line]
    
    # Train
    running_time = deque(maxlen=runlen) # Full time
    load_time = deque(maxlen=runlen) # load_batch time
    graph_time = deque(maxlen=runlen) # train_step time
    svtime = time()
    sys.stdout.write("Starting training for %d epochs\n"%epochs)
    for epoch in range(epochs):
        start_epoch = time()
        perm = np.random.permutation(L.labels)
        for task_name, task in T.items(): task.reset_total_loss()
        
        for step in range(config['steps_per_epoch']):
            start_step = time()
            
            # Train model for a step
            random_labels = perm[step*bs : (step+1)*bs]
            random_task = np.random.choice(list(header_dict.keys()), 1)[0]
            TT = time();batch = L.load_batch(
                random_labels
            );load_time.append(time()-TT) # load time sandwich
            TT=time();loss = graph(
                batch, random_task, optencoder, header.opts[random_task]
            );graph_time.append(time()-TT) # graph time sandwich
            
            # Save running stats
            T[random_task].log_loss(loss.numpy())
            running_time.append(time()-start_step)
            
            # Stdout
            if step%50==0:
                means = tuple([
                    task.calc_avg_running_loss()['main']
                    for task_name, task in T.items()
                ])
                loss_string = loss_spec%means
                sys.stdout.write(
                    "\r\033[KStep %6d/%6d, loss=%s (%.2f,%.2f,%.2f s)"%(
                        step, config['steps_per_epoch'], loss_string, 
                        np.mean(running_time), np.mean(load_time), 
                        np.mean(graph_time)
                    )
                )
            
            # Saving weights and testing
            if time()-svtime > svfreq:
                if swt:
                    save_all_weights(svdir)
                """
                sys.stdout.write("\r\033[KDownstream evlauation: %s\n"%(dstask))
                line = DS.TrainEval()
                Line = "Downstream evlauation: %s; "%(dstask) + line
                sys.stdout.write("\r\033[K%s\n"%Line)
                if msg:
                    with open("save/%s/valout.txt"%outnm, 'a') as F:
                        F.write("Global step %d: %s\n"%(encoder.global_step, line))
                    allvallines.append(Line)
                """
                svtime = time()

        # End of epoch
        tot_losses = tuple([
            task.calc_avg_total_loss()['main'] for task_name, task in T.items()
        ])
        loss_string = loss_spec%tot_losses
        Line = 'Epoch %d, Global step %d: mean_loss=%s (%d s)'%(
            epoch, encoder.global_step.numpy(), loss_string, time()-start_epoch
        )
        sys.stdout.write("\r\033[K%s\n"%Line)
        # Run quick(ish) few shot evaluation
        #line2 = DS.TrainEval()
        #sys.std.write('%s\n'%line2)
        if msg:
            U.message_board(Line+'\n', "save/%s/epochout.txt"%svdir)
            allepochlines.append(Line+"\n")
    
    # End of pre-training
    # Save weights, perahps
    if swt:
        save_all_weights(svdir)
    # Run quick(ish) few shot downstream evaluation
    for dstask in config['downstream']:
        DS = allds[dstask](dsconfig, base_model=encoder)
        sys.stdout.write("\r\033[KDownstream evlauation: %s\n"%(dstask))
        line = DS.TrainEval()
        Line = "Downstream evlauation: %s; "%(dstask) + line[-1]
        sys.stdout.write("\r\033[K%s\n"%Line)
        if msg:
            U.message_board(Line+'\n', "save/%s/epochout.txt"%svdir)
            allepochlines.append(Line+"\n")
    if msg:
        # Append results to the .all files
        U.message_board("".join(allepochlines), "save/epochout.all")
    
    print()

np.random.seed(config['seed'])
tf.random.set_seed(config['seed'])
train(epochs=config['epochs'], runlen=100, svfreq=config['svfreq'])

