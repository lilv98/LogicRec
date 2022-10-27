import os
import torch
import argparse
import random
import pandas as pd
import numpy as np 
import pdb
import tqdm

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_data(path):
    ui_data = np.load(path + 'ratings_final.npy')
    kg_data = np.load(path + 'kg_final.npy')
    pdb.set_trace()

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/book/', type=str)
    # parser.add_argument('--save_path', default='./', type=str)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    read_data(cfg.data_root)
