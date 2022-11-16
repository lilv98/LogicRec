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
            try:
                hrs = t2hr_dict[item]
                hr = random.choice(hrs)
                hrs2 = t2hr_dict[hr[0]]
                hr2 = random.choice(hrs2)
                if (hr2[0], hr2[1], hr[1], user) not in ret:
                    ret[(hr2[0], hr2[1], hr[1], user)] = set([item])
                else:
                    ret[(hr2[0], hr2[1], hr[1], user)].add(item)
            except:
                pass
    return ret

def construct_2p_test(t2hr_dict_train, t2hr_dict_test, data):
    ret = {}
    for user in tqdm.tqdm(data):
        items = data[user]
        for item in items:
            try:
                hrs = t2hr_dict_test[item]
                hr = random.choice(hrs)
                hrs2 = t2hr_dict_train[hr[0]]
                hr2 = random.choice(hrs2)
                if (hr2[0], hr2[1], hr[1], user) not in ret:
                    ret[(hr2[0], hr2[1], hr[1], user)] = set([item])
                else:
                    ret[(hr2[0], hr2[1], hr[1], user)].add(item)
            except:
                pass
    return ret

def construct_3p_train(t2hr_dict, data):
    ret = {}
    for user in tqdm.tqdm(data):
        items = data[user]
        for item in items:
            try:
                hrs = t2hr_dict[item]
                hr = random.choice(hrs)
                hrs2 = t2hr_dict[hr[0]]
                hr2 = random.choice(hrs2)
                hrs3 = t2hr_dict[hr2[0]]
                hr3 = random.choice(hrs3)
                if (hr3[0], hr3[1], hr2[1], hr[1], user) not in ret:
                    ret[(hr3[0], hr3[1], hr2[1], hr[1], user)] = set([item])
                else:
                    ret[(hr3[0], hr3[1], hr2[1], hr[1], user)].add(item)
            except:
                pass
    return ret

def construct_3p_test(t2hr_dict_train, t2hr_dict_test, data):
    ret = {}
    for user in tqdm.tqdm(data):
        items = data[user]
        for item in items:
            try:
                hrs = t2hr_dict_test[item]
                hr = random.choice(hrs)
                hrs2 = t2hr_dict_train[hr[0]]
                hr2 = random.choice(hrs2)
                hrs3 = t2hr_dict_train[hr2[0]]
                hr3 = random.choice(hrs3)
                if (hr3[0], hr3[1], hr2[1], hr[1], user) not in ret:
                    ret[(hr3[0], hr3[1], hr2[1], hr[1], user)] = set([item])
                else:
                    ret[(hr3[0], hr3[1], hr2[1], hr[1], user)].add(item)
            except:
                pass
    return ret

def construct_2i_train(t2hr_dict, data):
    ret = {}
    for user in tqdm.tqdm(data):
        items = data[user]
        for item in items:
            try:
                hrs1 = t2hr_dict[item]
                hr1 = random.choice(hrs1)
                hrs2 = t2hr_dict[item]
                hr2 = random.choice(hrs2)
                assert hr1 != hr2
                if (hr1[0], hr1[1], hr2[0], hr2[1], user) not in ret:
                    ret[(hr1[0], hr1[1], hr2[0], hr2[1], user)] = set([item])
                else:
                    ret[(hr1[0], hr1[1], hr2[0], hr2[1], user)].add(item)
            except:
                pass
    return ret

def construct_2i_test(t2hr_dict_train, t2hr_dict_test, data):
    ret = {}
    for user in tqdm.tqdm(data):
        items = data[user]
        for item in items:
            try:
                hrs1 = t2hr_dict_test[item]
                hr1 = random.choice(hrs1)
                hrs2 = t2hr_dict_train[item]
                hr2 = random.choice(hrs2)
                assert hr1 != hr2
                if (hr1[0], hr1[1], hr2[0], hr2[1], user) not in ret:
                    ret[(hr1[0], hr1[1], hr2[0], hr2[1], user)] = set([item])
                else:
                    ret[(hr1[0], hr1[1], hr2[0], hr2[1], user)].add(item)
            except:
                pass
    return ret

def construct_3i_train(t2hr_dict, data):
    ret = {}
    for user in tqdm.tqdm(data):
        items = data[user]
        for item in items:
            try:
                hrs1 = t2hr_dict[item]
                hr1 = random.choice(hrs1)
                hrs2 = t2hr_dict[item]
                hr2 = random.choice(hrs2)
                hrs3 = t2hr_dict[item]
                hr3 = random.choice(hrs3)
                assert hr1 != hr2 != hr3
                if (hr1[0], hr1[1], hr2[0], hr2[1], hr3[0], hr3[1], user) not in ret:
                    ret[(hr1[0], hr1[1], hr2[0], hr2[1], hr3[0], hr3[1], user)] = set([item])
                else:
                    ret[(hr1[0], hr1[1], hr2[0], hr2[1], hr3[0], hr3[1], user)].add(item)
            except:
                pass
    return ret

def construct_3i_test(t2hr_dict_train, t2hr_dict_test, data):
    ret = {}
    for user in tqdm.tqdm(data):
        items = data[user]
        for item in items:
            try:
                hrs1 = t2hr_dict_test[item]
                hr1 = random.choice(hrs1)
                hrs2 = t2hr_dict_train[item]
                hr2 = random.choice(hrs2)
                hrs3 = t2hr_dict_train[item]
                hr3 = random.choice(hrs3)
                assert hr1 != hr2 != hr3
                if (hr1[0], hr1[1], hr2[0], hr2[1], hr3[0], hr3[1], user) not in ret:
                    ret[(hr1[0], hr1[1], hr2[0], hr2[1], hr3[0], hr3[1], user)] = set([item])
                else:
                    ret[(hr1[0], hr1[1], hr2[0], hr2[1], hr3[0], hr3[1], user)].add(item)
            except:
                pass
    return ret

def save_txt(path, obj):
    with open(path, 'w') as f:
        for line in obj.values:
            f.write(' '.join([str(x) for x in line.tolist()]) + '\n')

def get_k_test_data(data, k):
    all_queries = list(data.keys())
    ret = set([])
    while len(ret) < k:
        query = random.choice(all_queries)
        answer = random.choice(list(data[query]))
        query = [x for x in query]
        query.extend([answer])
        ret.add(tuple(query))
    return list(ret)

def get_baseline_data_train(path, data):
    ret = {}
    for k in data:
        items = data[k]
        for item in items:
            if k[-1] in ret:
                ret[k[-1]].add(item)
            else:
                ret[k[-1]] = set([item])
    with open(path, 'w') as f:
        for k in ret:
            items = list(ret[k])
            line = str(k) + ' ' + ' '.join([str(x) for x in items]) + '\n'
            f.write(line)

def get_baseline_data_test(path, data):
    ret = {}
    for line in data:
        if line[2] in ret:
            ret[line[2]].add(line[-1])
        else:
            ret[line[2]] = set([line[-1]])
    with open(path, 'w') as f:
        for k in ret:
            items = list(ret[k])
            line = str(k) + ' ' + ' '.join([str(x) for x in items]) + '\n'
            f.write(line)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--dataset', default='amazon-book', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--N_test', default=2000, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    path = cfg.data_root + cfg.dataset
    input_path = cfg.data_root + cfg.dataset + '/input/'
    if not os.path.exists(input_path):
        os.makedirs(input_path)
    kg_train, kg_test, train_dict, test_dict, i_count, e_count, r_count = read_data(path)
    save_txt(input_path + 'kg_train.txt', kg_train)
    save_txt(input_path + 'kg_test.txt', kg_test)
    kg_train_t2hr_dict = get_mapper(kg_train)
    kg_test_t2hr_dict = get_mapper(kg_test)
    
    # 1p Format: (e_1, r_1, u): {a}
    print('Constructing 1p...')
    data_1p_train = construct_1p(kg_train_t2hr_dict, train_dict)
    data_1p_test = construct_1p(kg_test_t2hr_dict, test_dict)
    data_1p_test = get_k_test_data(data_1p_test, k=cfg.N_test)
    get_baseline_data_train(input_path + 'baseline_train.txt', data_1p_train)
    get_baseline_data_test(input_path + 'baseline_test.txt', data_1p_test)
    print(f'Stats 1p: #Train: {len(data_1p_train)}, #Test: {len(data_1p_test)}')
    save_obj(data_1p_train, input_path + '1p_train.pkl')
    save_obj(data_1p_test, input_path + '1p_test.pkl')
    
    # 2p Format: (e_1, r_1, r_2, u): {a}
    print('Constructing 2p...')
    data_2p_train = construct_2p_train(kg_train_t2hr_dict, train_dict)
    data_2p_test = construct_2p_test(kg_train_t2hr_dict, kg_test_t2hr_dict, test_dict)
    data_2p_test = get_k_test_data(data_2p_test, k=cfg.N_test)
    print(f'Stats 2p: #Train: {len(data_2p_train)}, #Test: {len(data_2p_test)}')
    save_obj(data_2p_train, input_path + '2p_train.pkl')
    save_obj(data_2p_test, input_path + '2p_test.pkl')
    
    # 3p Format: (e_1, r_1, r_2, r_3, u): {a}
    print('Constructing 3p...')
    data_3p_train = construct_3p_train(kg_train_t2hr_dict, train_dict)
    data_3p_test = construct_3p_test(kg_train_t2hr_dict, kg_test_t2hr_dict, test_dict)
    data_3p_test = get_k_test_data(data_3p_test, k=cfg.N_test)
    print(f'Stats 3p: #Train: {len(data_3p_train)}, #Test: {len(data_3p_test)}')
    save_obj(data_3p_train, input_path + '3p_train.pkl')
    save_obj(data_3p_test, input_path + '3p_test.pkl')
    
    # 2i Format: (e_1, r_1, e_2, r_2, u): {a}
    print('Constructing 2i...')
    data_2i_train = construct_2i_train(kg_train_t2hr_dict, train_dict)
    data_2i_test = construct_2i_test(kg_train_t2hr_dict, kg_test_t2hr_dict, test_dict)
    data_2i_test = get_k_test_data(data_2i_test, k=cfg.N_test)
    print(f'Stats 2i: #Train: {len(data_2i_train)}, #Test: {len(data_2i_test)}')
    save_obj(data_2i_train, input_path + '2i_train.pkl')
    save_obj(data_2i_test, input_path + '2i_test.pkl')
    
    # 3i Format: (e_1, r_1, e_2, r_2, e_3, r_3, u): {a}
    print('Constructing 3i...')
    data_3i_train = construct_3i_train(kg_train_t2hr_dict, train_dict)
    data_3i_test = construct_3i_test(kg_train_t2hr_dict, kg_test_t2hr_dict, test_dict)
    data_3i_test = get_k_test_data(data_3i_test, k=cfg.N_test)
    print(f'Stats 3i: #Train: {len(data_3i_train)}, #Test: {len(data_3i_test)}')
    save_obj(data_3i_train, input_path + '3i_train.pkl')
    save_obj(data_3i_test, input_path + '3i_test.pkl')
    