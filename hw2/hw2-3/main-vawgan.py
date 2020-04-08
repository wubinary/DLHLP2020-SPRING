import json
import os

from importlib import import_module

import numpy as np
##import tensorflow as tf
import torch 
from analyzer import Tanhize, read
from util.wrapper import validate_log_dirs
import argparse
import torch.utils.data as Data
from util.function import GaussianSampleLayer
from trainer import * 


parser = argparse.ArgumentParser(description='main-vawgan.py')
parser.add_argument('--corpus_name', default='vcc2016', help='Corpus name')
parser.add_argument('--logdir_root', default=None, help='root of log dir')
parser.add_argument('--logdir', default=None, help='log dir')
parser.add_argument('--restore_from', default=None, help='restore from dir (not from *.ckpt)')
parser.add_argument('--gpu_cfg', default=None, help= 'GPU configuration')
parser.add_argument('--summary_freq', type=int, default=1000, help='Update summary')
parser.add_argument('--ckpt', default=None, help= 'specify the ckpt in restore_from (if there are multiple ckpts)')
parser.add_argument('--architecture', default='architecture-vawgan-vcc2016.json', help= 'network architecture')
parser.add_argument('--model_module', default='model.vawgan', help='Model module')
parser.add_argument('--model', default='VAWGAN', help='Model: ConvVAE, VAWGAN')
parser.add_argument('--trainer_module', default='trainer.vawgan', help='Trainer module')
parser.add_argument('--trainer', default='VAWGANTrainer', help='Trainer: VAETrainer, VAWGANTrainer')
args = parser.parse_args()







def main(unused_args=None):
    ''' NOTE: The input is rescaled to [-1, 1] '''
    #module = import_module(args.model_module, package=None)
    #MODEL = getattr(module, args.model)
    #print("=== ",MODEL,"")
    #module = import_module(args.trainer_module, package=None)
    #TRAINER = getattr(module, args.trainer)


    dirs = validate_log_dirs(args)
    
    try:
        os.makedirs(dirs['logdir'])
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    

    with open(args.architecture) as f:
        arch = json.load(f)

    with open(os.path.join(dirs['logdir'], args.architecture), 'w') as f:
        json.dump(arch, f, indent=4)

    normalizer = Tanhize(
        xmax=np.fromfile('./etc/{}_xmax.npf'.format(args.corpus_name)),
        xmin=np.fromfile('./etc/{}_xmin.npf'.format(args.corpus_name)),
    )

    s_x_y = read(
        file_pattern=arch['training']['src_dir'],
        batch_size=arch['training']['batch_size'],
        capacity=2048,
        min_after_dequeue=1024,
        normalizer=normalizer,
        data_format='NHWC',
    )

    t_x_y = read(
        file_pattern=arch['training']['trg_dir'],
        batch_size=arch['training']['batch_size'],
        capacity=2048,
        min_after_dequeue=1024,
        normalizer=normalizer,
        data_format='NHWC',
    )
    
    
    machine = Trainer()
   
    machine.load_data(s_x_y,t_x_y)
    machine.train()
  
    
  
    

    
    

if __name__ == '__main__':
    main()
