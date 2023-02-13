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
import multiprocessing as mp

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
        for line in tqdm.tqdm(f):
            line = line.strip('\n').split(' ')
            kg.append([int(line[0]), int(line[1]), int(line[2])])
    kg = pd.DataFrame(kg, columns=['h', 'r', 't'])
    e_count = len(set(kg.t)|set(kg.h))
    print(f'Entity ids range 0 - {e_count - 1}')
    r_count = len(set(kg.r))
    print(f'Relation ids range 0 - {r_count - 1}')

    kg_train, kg_test = train_test_split(kg, test_size=0.05)
    
    rec_train_dict = {}
    with open(path + '/train.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            rec_train_dict[int(line[0])] = set([int(x) for x in line[1:]])

    rec_test_dict = {}
    with open(path + '/test.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            try:
                rec_test_dict[int(line[0])] = set([int(x) for x in line[1:]])
            except:
                pass
    
    return kg_train, kg_test, rec_train_dict, rec_test_dict, i_count, e_count, r_count

def get_mapper_t2hr(kg):
    t2hr_dict = {}
    for triple in tqdm.tqdm(kg.values):
        if triple[-1] not in t2hr_dict:
            t2hr_dict[triple[-1]] = set([(triple[0], triple[1])])
        else:
            t2hr_dict[triple[-1]].add((triple[0], triple[1]))
    return t2hr_dict

def get_mapper_hr2t(kg):
    hr2t_dict = {}
    for triple in tqdm.tqdm(kg.values):
        if (triple[0], triple[1]) not in hr2t_dict:
            hr2t_dict[(triple[0], triple[1])] = set([triple[-1]])
        else:
            hr2t_dict[(triple[0], triple[1])].add(triple[-1])
    return hr2t_dict

def construct_1p_train(t2hr_dict_train, hr2t_dict_train, rec_train_dict):
    ret = {}
    for user in tqdm.tqdm(rec_train_dict):
        items = rec_train_dict[user]
        for item in items:
            try:
                hrs = t2hr_dict_train[item]
                for hr in hrs:
                    if (hr[0], hr[1], user) not in ret:
                        ret[(hr[0], hr[1], user)] = {'both': set([item])}
                    else:
                        ret[(hr[0], hr[1], user)]['both'].add(item)
            except:
                pass
    
    for query in tqdm.tqdm(ret):
        ret[query]['lqa'] = hr2t_dict_train[(query[0], query[1])]
        ret[query]['rec'] = rec_train_dict[query[-1]]
        assert (ret[query]['lqa'] & ret[query]['rec']) == ret[query]['both']

    return ret

def construct_1p_test(t2hr_dict_test, rec_test_dict, k):
    users = list(rec_test_dict.keys())
    ret = []
    with tqdm.tqdm(total=k) as pbar:
        while len(ret) < k:
            try:
                user = np.random.choice(users)
                item = np.random.choice(list(rec_test_dict[user]))
                hrs = list(t2hr_dict_test[item])
                hr = hrs[np.random.choice(range(len(hrs)))]
                ret.append([hr[0], hr[1], user, item])
                pbar.update(1)
            except:
                pass

    return ret

def construct_2p_train(t2hr_dict_train, hr2t_dict_train, rec_train_dict, k):
    users = list(rec_test_dict.keys())
    ret = {}
    with tqdm.tqdm(total=k) as pbar:
        while len(ret) < k:
            try:
                user = np.random.choice(users)
                item = np.random.choice(list(rec_train_dict[user]))
                hrs = list(t2hr_dict_train[item])
                hr = hrs[np.random.choice(range(len(hrs)))]
                hrs2 = list(t2hr_dict_train[hr[0]])
                hr2 = hrs2[np.random.choice(range(len(hrs2)))]
                query = (hr2[0], hr2[1], hr[1], user)

                flag = 1
                ts = hr2t_dict_train[(query[0], query[1])]
                if len(ts) > 10000:
                    flag = 0
                else:
                    for t in ts:
                        try:
                            ts2 = hr2t_dict_train[(t, query[2])]
                        except:
                            ts2 = set()
                        if len(ts2) > 10000:
                            flag = 0
                            break

                if flag:
                    ret[query] = {'rec': rec_train_dict[query[-1]], 'lqa': set(), 'both': item}
                    for t in ts:
                        try:
                            ts2 = hr2t_dict_train[(t, query[2])]
                            for t2 in ts2:
                                ret[query]['lqa'].add(t2)
                        except:
                            pass

                    both = (ret[query]['lqa'] & ret[query]['rec'])
                    assert ret[query]['both'] in both
                    ret[query]['both'] = both
                    pbar.update(1)
            except:
                pass

    return ret

def construct_2p_test(t2hr_dict_test, t2hr_dict_train, rec_test_dict, k):
    users = list(rec_test_dict.keys())
    ret = []
    with tqdm.tqdm(total=k) as pbar:
        while len(ret) < k:
            try:
                user = np.random.choice(users)
                item = np.random.choice(list(rec_test_dict[user]))
                hrs = list(t2hr_dict_test[item])
                hr = hrs[np.random.choice(range(len(hrs)))]
                hrs2 = list(t2hr_dict_train[hr[0]])
                hr2 = hrs2[np.random.choice(range(len(hrs2)))]
                ret.append([hr2[0], hr2[1], hr[1], user, item])
                pbar.update(1)
            except:
                pass

    return ret

def construct_3p_train(t2hr_dict_train, hr2t_dict_train, rec_train_dict, k):
    users = list(rec_test_dict.keys())
    ret = {}
    with tqdm.tqdm(total=k) as pbar:
        while len(ret) < k:
            try:
                user = np.random.choice(users)
                item = np.random.choice(list(rec_train_dict[user]))
                hrs = list(t2hr_dict_train[item])
                hr = hrs[np.random.choice(range(len(hrs)))]
                hrs2 = list(t2hr_dict_train[hr[0]])
                hr2 = hrs2[np.random.choice(range(len(hrs2)))]
                hrs3 = list(t2hr_dict_train[hr2[0]])
                hr3 = hrs3[np.random.choice(range(len(hrs3)))]
                query = (hr3[0], hr3[1], hr2[1], hr[1], user)

                flag = 1
                ts = hr2t_dict_train[(query[0], query[1])]
                if len(ts) > 10000:
                    flag = 0
                else:
                    for t in ts:
                        try:
                            ts2 = hr2t_dict_train[(t, query[2])]
                        except:
                            ts2 = set()
                        if len(ts2) > 10000:
                            flag = 0
                            break
                        else:
                            for t2 in ts2:
                                try:
                                    ts3 = hr2t_dict_train[(t2, query[3])]
                                except:
                                    ts3 = set()
                                if len(ts3) > 10000:
                                    flag = 0
                                    break

                if flag:
                    ret[query] = {'rec': rec_train_dict[query[-1]], 'lqa': set(), 'both': item}
                    for t in ts:
                        try:
                            ts2 = hr2t_dict_train[(t, query[2])]
                        except:
                            ts2 = set()
                        for t2 in ts2:
                            try:
                                ts3 = hr2t_dict_train[(t2, query[3])]
                                for t3 in ts3:
                                    ret[query]['lqa'].add(t3)
                            except:
                                pass

                    both = (ret[query]['lqa'] & ret[query]['rec'])
                    assert ret[query]['both'] in both
                    ret[query]['both'] = both
                    pbar.update(1)
            except:
                pass

    return ret

def construct_3p_test(t2hr_dict_test, t2hr_dict_train, rec_test_dict, k):
    users = list(rec_test_dict.keys())
    ret = []
    with tqdm.tqdm(total=k) as pbar:
        while len(ret) < k:
            try:
                user = np.random.choice(users)
                item = np.random.choice(list(rec_test_dict[user]))
                hrs = list(t2hr_dict_test[item])
                hr = hrs[np.random.choice(range(len(hrs)))]
                hrs2 = list(t2hr_dict_train[hr[0]])
                hr2 = hrs2[np.random.choice(range(len(hrs2)))]
                hrs3 = list(t2hr_dict_train[hr2[0]])
                hr3 = hrs3[np.random.choice(range(len(hrs3)))]
                ret.append([hr3[0], hr3[1], hr2[1], hr[1], user, item])
                pbar.update(1)
            except:
                pass

    return ret

def construct_2i_train(t2hr_dict_train, hr2t_dict_train, rec_train_dict, k):
    users = list(rec_test_dict.keys())
    ret = {}
    with tqdm.tqdm(total=k) as pbar:
        while len(ret) < k:
            try:
                user = np.random.choice(users)
                item = np.random.choice(list(rec_train_dict[user]))
                hrs1 = list(t2hr_dict_train[item])
                hr1 = hrs1[np.random.choice(range(len(hrs1)))]
                hrs2 = list(t2hr_dict_train[item])
                hr2 = hrs2[np.random.choice(range(len(hrs2)))]
                assert hr1 != hr2
                query = (hr1[0], hr1[1], hr2[0], hr2[1], user)

                lqa_answers = hr2t_dict_train[(query[0], query[1])] & hr2t_dict_train[(query[2], query[3])]
                if len(lqa_answers) < 10000:
                    ret[query] = {'lqa': lqa_answers, 'rec': rec_train_dict[query[-1]], 'both': item}
                    both = (lqa_answers & ret[query]['rec'])
                    assert ret[query]['both'] in both
                    ret[query]['both'] = both
                    pbar.update(1)
            except:
                pass

    return ret

def construct_2i_test(t2hr_dict_test, t2hr_dict_train, rec_test_dict, k):
    users = list(rec_test_dict.keys())
    ret = []
    with tqdm.tqdm(total=k) as pbar:
        while len(ret) < k:
            try:
                user = np.random.choice(users)
                item = np.random.choice(list(rec_test_dict[user]))
                hrs1 = list(t2hr_dict_test[item])
                hr1 = hrs1[np.random.choice(range(len(hrs1)))]
                hrs2 = list(t2hr_dict_train[item])
                hr2 = hrs2[np.random.choice(range(len(hrs2)))]
                assert hr1 != hr2
                ret.append([hr1[0], hr1[1], hr2[0], hr2[1], user, item])
                pbar.update(1)
            except:
                pass

    return ret

def construct_3i_train(t2hr_dict_train, hr2t_dict_train, rec_train_dict, k):
    users = list(rec_test_dict.keys())
    ret = {}
    with tqdm.tqdm(total=k) as pbar:
        while len(ret) < k:
            try:
                user = np.random.choice(users)
                item = np.random.choice(list(rec_train_dict[user]))
                hrs1 = list(t2hr_dict_train[item])
                hr1 = hrs1[np.random.choice(range(len(hrs1)))]
                hrs2 = list(t2hr_dict_train[item])
                hr2 = hrs2[np.random.choice(range(len(hrs2)))]
                hrs3 = list(t2hr_dict_train[item])
                hr3 = hrs3[np.random.choice(range(len(hrs3)))]
                assert hr1 != hr2 != hr3
                query = (hr1[0], hr1[1], hr2[0], hr2[1], hr3[0], hr3[1], user)
                lqa_answers = hr2t_dict_train[(query[0], query[1])] & hr2t_dict_train[(query[2], query[3])] & hr2t_dict_train[(query[4], query[5])]
                if len(lqa_answers) < 10000:
                    ret[query] = {'lqa': lqa_answers, 'rec': rec_train_dict[query[-1]], 'both': item}
                    both = (lqa_answers & ret[query]['rec'])
                    assert ret[query]['both'] in both
                    ret[query]['both'] = both
                    pbar.update(1)
            except:
                pass

    return ret

def construct_3i_test(t2hr_dict_test, t2hr_dict_train, rec_test_dict, k):
    users = list(rec_test_dict.keys())
    ret = []
    with tqdm.tqdm(total=k) as pbar:
        while len(ret) < k:
            try:
                user = np.random.choice(users)
                item = np.random.choice(list(rec_test_dict[user]))
                hrs1 = list(t2hr_dict_test[item])
                hr1 = hrs1[np.random.choice(range(len(hrs1)))]
                hrs2 = list(t2hr_dict_train[item])
                hr2 = hrs2[np.random.choice(range(len(hrs2)))]
                hrs3 = list(t2hr_dict_train[item])
                hr3 = hrs3[np.random.choice(range(len(hrs3)))]
                assert hr1 != hr2 != hr3
                ret.append([hr1[0], hr1[1], hr2[0], hr2[1], hr3[0], hr3[1], user, item])
                pbar.update(1)
            except:
                pass

    return ret

def save_txt(path, obj):
    with open(path, 'w') as f:
        for line in obj.values:
            f.write(' '.join([str(x) for x in line.tolist()]) + '\n')

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--dataset', default='amazon-book', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--N_train', default=10000, type=int)
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
    kg_train, kg_test, rec_train_dict, rec_test_dict, i_count, e_count, r_count = read_data(path)
    save_txt(input_path + 'kg_train.txt', kg_train)
    save_txt(input_path + 'kg_test.txt', kg_test)
    kg_t2hr_dict_train = get_mapper_t2hr(kg_train)
    kg_hr2t_dict_train = get_mapper_hr2t(kg_train)
    kg_t2hr_dict_test = get_mapper_t2hr(kg_test)
    kg_hr2t_dict_test = get_mapper_hr2t(kg_test)
    
    # 1p Train Format: (e_1, r_1, u): {'rec': {rec answers}, 'lqa': {lqa answers}, 'both': {joint answers}}
    # 1p Test Format: [e_1, r_1, u, joint answer]
    print('Constructing 1p...')
    data_1p_train = construct_1p_train(kg_t2hr_dict_train, kg_hr2t_dict_train, rec_train_dict)
    data_1p_test = construct_1p_test(kg_t2hr_dict_test, rec_test_dict, k=cfg.N_test)
    print(f'Stats 1p: #Train: {len(data_1p_train)}, #Test: {len(data_1p_test)}')
    save_obj(data_1p_train, input_path + '1p_train.pkl')
    save_obj(data_1p_test, input_path + '1p_test.pkl')
    
    # 2p Train Format: (e_1, r_1, r_2, u): {'rec': {rec answers}, 'lqa': {lqa answers}, 'both': {joint answers}}
    # 2p Test Format: [e_1, r_1, r_2, u, joint answer]
    print('Constructing 2p...')
    data_2p_train = construct_2p_train(kg_t2hr_dict_train, kg_hr2t_dict_train, rec_train_dict, k=cfg.N_train)
    data_2p_test = construct_2p_test(kg_t2hr_dict_test, kg_t2hr_dict_train, rec_test_dict, k=cfg.N_test)
    print(f'Stats 2p: #Train: {len(data_2p_train)}, #Test: {len(data_2p_test)}')
    save_obj(data_2p_train, input_path + '2p_train.pkl')
    save_obj(data_2p_test, input_path + '2p_test.pkl')
    
    # 3p Train Format: (e_1, r_1, r_2, r_3, u): {'rec': {rec answers}, 'lqa': {lqa answers}, 'both': {joint answers}}
    # 3p Test Format: [e_1, r_1, r_2, r_3, u, joint answer]
    print('Constructing 3p...')
    data_3p_train = construct_3p_train(kg_t2hr_dict_train, kg_hr2t_dict_train, rec_train_dict, k=cfg.N_train)
    data_3p_test = construct_3p_test(kg_t2hr_dict_test, kg_t2hr_dict_train, rec_test_dict, k=cfg.N_test)
    print(f'Stats 3p: #Train: {len(data_3p_train)}, #Test: {len(data_3p_test)}')
    save_obj(data_3p_train, input_path + '3p_train.pkl')
    save_obj(data_3p_test, input_path + '3p_test.pkl')
    
    # 2i Train Format: (e_1, r_1, e_2, r_2, u): {'rec': {rec answers}, 'lqa': {lqa answers}, 'both': {joint answers}}
    # 2i Test Format: [e_1, r_1, e_2, r_2, u, joint answer]
    print('Constructing 2i...')
    data_2i_train = construct_2i_train(kg_t2hr_dict_train, kg_hr2t_dict_train, rec_train_dict, k=cfg.N_train)
    data_2i_test = construct_2i_test(kg_t2hr_dict_test, kg_t2hr_dict_train, rec_test_dict, k=cfg.N_test)
    print(f'Stats 2i: #Train: {len(data_2i_train)}, #Test: {len(data_2i_test)}')
    save_obj(data_2i_train, input_path + '2i_train.pkl')
    save_obj(data_2i_test, input_path + '2i_test.pkl')
    
    # 3i Train Format: (e_1, r_1, e_2, r_2, e_3, r_3, u): {'rec': {rec answers}, 'lqa': {lqa answers}, 'both': {joint answers}}
    # 3i Test Format: [e_1, r_1, e_2, r_2, e_3, r_3, u, joint answer]
    print('Constructing 3i...')
    data_3i_train = construct_3i_train(kg_t2hr_dict_train, kg_hr2t_dict_train, rec_train_dict, k=cfg.N_train)
    data_3i_test = construct_3i_test(kg_t2hr_dict_test, kg_t2hr_dict_train, rec_test_dict, k=cfg.N_test)
    print(f'Stats 3i: #Train: {len(data_3i_train)}, #Test: {len(data_3i_test)}')
    save_obj(data_3i_train, input_path + '3i_train.pkl')
    save_obj(data_3i_test, input_path + '3i_test.pkl')
    