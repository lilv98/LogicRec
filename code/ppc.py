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

def read_data(cfg):
    path = cfg.data_root + '/' + cfg.dataset

    i_count = 0
    with open(path + '/item_list.txt') as f:
        for line in f:
            i_count += 1
    print(f'Item ids range 0 - {i_count - 2}')

    kg = []
    with open(path + '/kg_final.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            kg.append([int(line[0]), int(line[1]), int(line[2])])
    kg = pd.DataFrame(kg, columns=['h', 'r', 't'])
    e_count = len(set(kg.t)|set(kg.h))
    print(f'Entity ids range 0 - {e_count - 1}')
    r_count = len(set(kg.r))
    print(f'Relation ids range 0 - {r_count - 1}')
    
    train_dict = {}
    with open(path + '/train.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            train_dict[int(line[0])] = [int(x) for x in line[1:]]

    test_dict = {}
    with open(path + '/train.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            test_dict[int(line[0])] = [int(x) for x in line[1:]]
    
    return kg, train_dict, test_dict, i_count, e_count, r_count


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--dataset', default='amazon-book', type=str)
    parser.add_argument('--seed', default=42, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    kg, train_dict, test_dict, i_count, e_count, r_count = read_data(cfg)
