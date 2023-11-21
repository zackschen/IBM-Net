# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy # needed (don't change it)
import importlib
import os, time
import sys
import socket
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/dataset')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')

from dataset import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from dataset import ContinualDataset
from utils.continual_training import train as ctrain
from dataset import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import setproctitle
import torch
import uuid
import datetime

def parse_args():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k + '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'        
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    return args

def main(args=None):
    os.environ['JOBLIB_TEMP_FOLDER'] = '/home/chencheng/Code/mammoth/Tempdir'

    if args is None:
        args = parse_args()    

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    dataset = get_dataset(args)

    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and args.minibatch_size is None:
        args.minibatch_size = dataset.get_minibatch_size()

    backbone = dataset.get_backbone(args)
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    model.dataset = dataset

    time_stamp = time.strftime('%Y-%m-%d-%H-%M',time.localtime(int(round(time.time()*1000))/1000))
    if not args.debug:
        path_file = f'./logs/{dataset.NAME}/{args.model}/{time_stamp}_seed={args.seed}.txt'
        if args.backbone == 'alexnet':
            path_file = f'./logs/{dataset.NAME}-alexnet/{args.model}/{time_stamp}_seed={args.seed}.txt'
        if args.notes is not None:
            path_file = f'./logs/{dataset.NAME}/{args.model}/{time_stamp}_{args.notes}_seed={args.seed}.txt'
            if args.backbone == 'alexnet':
                path_file = f'./logs/{dataset.NAME}-alexnet/{args.model}/{time_stamp}_{args.notes}_seed={args.seed}.txt'
        os.makedirs(f'./logs/{dataset.NAME}/{args.model}', exist_ok=True)
        file = open(path_file, 'a')
    else:
        file = None
    model.file = file


    print_args(args,file)
    
    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format(args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))     

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args, file)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)

def print_args(args,file):
    print("***************",file=file, flush=True)
    print("** Arguments **",file=file, flush=True)
    print("***************",file=file, flush=True)
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]),file=file, flush=True)
    print("************",file=file, flush=True)

if __name__ == '__main__':
    main()
