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

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_data(path):
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
    with open(path + '/test.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            try:
                test_dict[int(line[0])] = [int(x) for x in line[1:]]
            except:
                pass
    
    kg_train, kg_test = train_test_split(kg, test_size=0.05)
    
    return kg_train, kg_test, train_dict, test_dict, i_count, e_count, r_count

def get_mapper(kg):
    t2hr_dict = {}
    for triple in tqdm.tqdm(kg.values):
        if triple[-1] in t2hr_dict:
            t2hr_dict[triple[-1]].append((triple[0], triple[1]))
        else:
            t2hr_dict[triple[-1]] = [(triple[0], triple[1])]
    return t2hr_dict

def construct_1p(t2hr_dict, data):
    ret = {}
    for user in tqdm.tqdm(data):
        items = data[user]
        for item in items:
            try:
                hrs = t2hr_dict[item]
                for hr in hrs:
                    if (hr[0], hr[1]) not in ret:
                        ret[(hr[0], hr[1], user)] = set([item])
                    else:
                        ret[(hr[0], hr[1], user)].add(item)
            except:
                pass
    return ret

def construct_2p_train(t2hr_dict, data):
    ret = {}
    for user in tqdm.tqdm(data):
        items = data[user]
        for item in items:
            hrs = t2hr_dict[item]
            for hr in hrs:
                hrs2 = t2hr_dict[hr[0]]
                for hr2 in hrs2:
                    # TODO: random choice
                    if (hr2[0], hr2[1], hr[1], user) not in ret:
                        ret[(hr2[0], hr2[1], hr[1], user)] = set([item])
                    else:
                        ret[(hr2[0], hr2[1], hr[1], user)].add(item)
    return ret

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
    path = cfg.data_root + '/' + cfg.dataset
    kg_train, kg_test, train_dict, test_dict, i_count, e_count, r_count = read_data(path)
    kg_train_t2hr_dict = get_mapper(kg_train)
    kg_test_t2hr_dict = get_mapper(kg_test)
    # data_1p_train = construct_1p(kg_train_t2hr_dict, train_dict)
    # data_1p_test = construct_1p(kg_test_t2hr_dict, test_dict)
    data_2p_train = construct_2p_train(kg_train_t2hr_dict, train_dict)
    pdb.set_trace()
    # save_obj(data_1p_train, path + '/1p_train.pkl')
    # save_obj(data_1p_test, path + '/1p_test.pkl')
    
    