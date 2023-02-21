import os
import torch
import argparse
import random
import pandas as pd
import numpy as np 
import pdb
import tqdm
from sklearn.model_selection import train_test_split
import pickle

def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--cached', default=0, type=int)
    parser.add_argument('--dataset', default='yelp2018', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--num_ng', default=8, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--alpha', default=1, type=int)
    parser.add_argument('--gamma', default=12, type=int)
    parser.add_argument('--cen', default=0.02, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--bs', default=1024, type=int)
    parser.add_argument('--add_loss', default=1, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--tolerance', default=5, type=int)
    parser.add_argument('--valid_interval', default=1000, type=int)
    parser.add_argument('--which', default='logicrec', type=str)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    input_path = cfg.data_root + cfg.dataset
    test_1p = load_obj(input_path + '/input/1p_test.pkl')
    test_2p = load_obj(input_path + '/input/2p_test.pkl')
    test_3p = load_obj(input_path + '/input/3p_test.pkl')
    test_2i = load_obj(input_path + '/input/2i_test.pkl')
    test_3i = load_obj(input_path + '/input/3i_test.pkl')
    test_pi = load_obj(input_path + '/input/pi_test.pkl')
    test_ip = load_obj(input_path + '/input/ip_test.pkl')
    test_2u = load_obj(input_path + '/input/2u_test.pkl')
    test_up = load_obj(input_path + '/input/up_test.pkl')
    valid = []
    test = []
    for data in [test_1p, test_2p, test_3p, test_2i, test_3i, test_pi, test_ip, test_2u, test_up]:
        for sample in data[:1000]:
            user = sample[-2]
            item = sample[-1]
            valid.append([user, item])
        for sample in data[1000:]:
            user = sample[-2]
            item = sample[-1]
            test.append([user, item])
    save_obj(valid, input_path + '/input/rec_valid.pkl')
    save_obj(test, input_path + '/input/rec_test.pkl')