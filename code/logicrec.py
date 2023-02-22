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
    N_item = -1
    with open(path + '/item_list.txt') as f:
        for line in f:
            N_item += 1

    N_ent = -1
    with open(path + '/entity_list.txt') as f:
        for line in f:
            N_ent += 1
    
    N_rel = -1
    with open(path + '/relation_list.txt') as f:
        for line in f:
            N_rel += 1
    N_rel = N_rel * 2

    filters = {}
    with open(path + '/train.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            filters[int(line[0])] = set([int(x) for x in line[1:]])

    N_user = len(filters.keys())
    print(f'N_item: {N_item}')
    print(f'N_ent: {N_ent}')
    print(f'N_rel: {N_rel}')
    print(f'N_user: {N_user}')

    return N_rel, N_item, N_ent, N_user, filters

def combine_data(data_1p, data_2p, data_3p, data_2i, data_3i, N_item, k):
    item_candidates = set(range(N_item))
    ret = []
    counter = 0
    # 1p: e1, r1, u, ph, ph, ph, ph, i, lqa, rec, both, 1
    for query in tqdm.tqdm(data_1p):
        # counter += 1
        # if counter < 100001:
        query_as_list = list(query)
        items = data_1p[query]['both']
        for item in items:
            query_as_list.extend([0, 0, 0, 0, item, 1, 1, 1, 1])
            ret.append(query_as_list)
            query_as_list = list(query)
        lqa_answers = (data_1p[query]['lqa'] & item_candidates)
        if len(lqa_answers) > k:
            lqa_answers = np.random.choice(list(lqa_answers), k)
        for lqa_answer in lqa_answers:
            query_as_list.extend([0, 0, 0, 0, lqa_answer, 1, 0, 0, 1])
            ret.append(query_as_list)
            query_as_list = list(query)
        rec_answers = data_1p[query]['rec']
        if len(rec_answers) > k:
            rec_answers = np.random.choice(list(rec_answers), k)
        for rec_answer in rec_answers:
            query_as_list.extend([0, 0, 0, 0, rec_answer, 0, 1, 0, 1])
            ret.append(query_as_list)
            query_as_list = list(query)

    # 2p: e1, r1, r2, u, ph, ph, ph, i, lqa, rec, both, 2
    for query in tqdm.tqdm(data_2p):
        query_as_list = list(query)
        items = data_2p[query]['both']
        for item in items:
            query_as_list.extend([0, 0, 0, item, 1, 1, 1, 2])
            ret.append(query_as_list)
            query_as_list = list(query)
        lqa_answers = (data_2p[query]['lqa'] & item_candidates)
        if len(lqa_answers) > k:
            lqa_answers = np.random.choice(list(lqa_answers), k)
        for lqa_answer in lqa_answers:
            query_as_list.extend([0, 0, 0, lqa_answer, 1, 0, 0, 2])
            ret.append(query_as_list)
            query_as_list = list(query)
        rec_answers = data_2p[query]['rec']
        if len(rec_answers) > k:
            rec_answers = np.random.choice(list(rec_answers), k)
        for rec_answer in rec_answers:
            query_as_list.extend([0, 0, 0, rec_answer, 0, 1, 0, 2])
            ret.append(query_as_list)
            query_as_list = list(query)

    # 3p: e1, r1, r2, r3, u, ph, ph, i, lqa, rec, both, 3
    for query in tqdm.tqdm(data_3p):
        query_as_list = list(query)
        items = data_3p[query]['both']
        for item in items:
            query_as_list.extend([0, 0, item, 1, 1, 1, 3])
            ret.append(query_as_list)
            query_as_list = list(query)
        lqa_answers = (data_3p[query]['lqa'] & item_candidates)
        if len(lqa_answers) > k:
            lqa_answers = np.random.choice(list(lqa_answers), k)
        for lqa_answer in lqa_answers:
            query_as_list.extend([0, 0, lqa_answer, 1, 0, 0, 3])
            ret.append(query_as_list)
            query_as_list = list(query)
        rec_answers = data_3p[query]['rec']
        if len(rec_answers) > k:
            rec_answers = np.random.choice(list(rec_answers), k)
        for rec_answer in rec_answers:
            query_as_list.extend([0, 0, rec_answer, 0, 1, 0, 3])
            ret.append(query_as_list)
            query_as_list = list(query)

    # 2i: e1, r1, e2, r2, u, ph, ph, i, lqa, rec, both, 4
    for query in tqdm.tqdm(data_2i):
        query_as_list = list(query)
        items = data_2i[query]['both']
        for item in items:
            query_as_list.extend([0, 0, item, 1, 1, 1, 4])
            ret.append(query_as_list)
            query_as_list = list(query)
        lqa_answers = (data_2i[query]['lqa'] & item_candidates)
        if len(lqa_answers) > k:
            lqa_answers = np.random.choice(list(lqa_answers), k)
        for lqa_answer in lqa_answers:
            query_as_list.extend([0, 0, lqa_answer, 1, 0, 0, 4])
            ret.append(query_as_list)
            query_as_list = list(query)
        rec_answers = data_2i[query]['rec']
        if len(rec_answers) > k:
            rec_answers = np.random.choice(list(rec_answers), k)
        for rec_answer in rec_answers:
            query_as_list.extend([0, 0, rec_answer, 0, 1, 0, 4])
            ret.append(query_as_list)
            query_as_list = list(query)

    # 3i: e1, r1, e2, r2, e3, r3, u, i, lqa, rec, both, 5
    for query in tqdm.tqdm(data_3i):
        query_as_list = list(query)
        items = data_3i[query]['both']
        for item in items:
            query_as_list.extend([item, 1, 1, 1, 5])
            ret.append(query_as_list)
            query_as_list = list(query)
        lqa_answers = (data_3i[query]['lqa'] & item_candidates)
        if len(lqa_answers) > k:
            lqa_answers = np.random.choice(list(lqa_answers), k)
        for lqa_answer in lqa_answers:
            query_as_list.extend([lqa_answer, 1, 0, 0, 5])
            ret.append(query_as_list)
            query_as_list = list(query)
        rec_answers = data_3i[query]['rec']
        if len(rec_answers) > k:
            rec_answers = np.random.choice(list(rec_answers), k)
        for rec_answer in rec_answers:
            query_as_list.extend([rec_answer, 0, 1, 0, 5])
            ret.append(query_as_list)
            query_as_list = list(query)
        
    return torch.tensor(ret)

class LQADatasetTrain(torch.utils.data.Dataset):
    def __init__(self, N_ent, data, cfg):
        super().__init__()
        self.N_ent = N_ent
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        neg = pos.clone()
        neg_item = np.random.choice(self.N_ent)
        neg[-5: -1] = torch.tensor([neg_item, 0, 0, 0])

        return torch.cat([pos.unsqueeze(dim=0), neg.unsqueeze(dim=0)], dim=0)

def my_collate_fn(batch):
    return torch.cat(batch, dim=0)

class LQADatasetTest(torch.utils.data.Dataset):
    def __init__(self, N_item, data, cfg, stage, query_type):
        super().__init__()
        self.N_item = N_item
        self.stage = stage
        self.query_type = query_type
        self.data = self._get_data(data)

    def _get_data(self, data):
        ret = []
        for pos in data:
            pos_as_list = list(pos)[:-1]
            pos_item = pos[-1]
            if self.query_type == '1p':
                pos_as_list.extend([0, 0, 0, 0, pos_item, 1, 1, 1, 1])
            elif self.query_type == '2p':
                pos_as_list.extend([0, 0, 0, pos_item, 1, 1, 1, 2])
            elif self.query_type == '3p':
                pos_as_list.extend([0, 0, pos_item, 1, 1, 1, 3])
            elif self.query_type == '2i':
                pos_as_list.extend([0, 0, pos_item, 1, 1, 1, 4])
            elif self.query_type == '3i':
                pos_as_list.extend([pos_item, 1, 1, 1, 5])
            elif self.query_type == 'pi':
                pos_as_list.extend([0, pos_item, 1, 1, 1, 6])
            elif self.query_type == 'ip':
                pos_as_list.extend([0, pos_item, 1, 1, 1, 7])
            elif self.query_type == '2u':
                pos_as_list.extend([0, 0, pos_item, 1, 1, 1, 8])
            elif self.query_type == 'up':
                pos_as_list.extend([0, pos_item, 1, 1, 1, 9])
            ret.append(pos_as_list)

        if self.stage == 'valid':
            return torch.tensor(ret)[:len(ret) // 2]
        elif self.stage == 'test':
            return torch.tensor(ret)[len(ret) // 2:]
        else:
            raise ValueError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        queries = pos[:-5].unsqueeze(dim=0).expand(self.N_item, len(pos) - 5)
        items = torch.arange(self.N_item).unsqueeze(dim=1)
        labels = pos[-4:].unsqueeze(dim=0).expand(self.N_item, 4)
        return torch.cat([queries, items, labels], dim=-1), pos

class IntersectionNet(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, self.dim)
        self.layer2 = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = torch.nn.functional.relu(self.layer1(embeddings))
        attention = torch.nn.functional.softmax(self.layer2(layer1_act), dim=1)
        embedding = torch.sum(attention * embeddings, dim=1)
        return embedding

class Expert(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.xavier_uniform_(self.layer1.weight)

    def forward(self, emb):
        return torch.nn.functional.relu(self.layer1(emb))

class Gate(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, 3)
        torch.nn.init.xavier_uniform_(self.layer1.weight)

    def forward(self, emb):
        return torch.softmax(self.layer1(emb), dim=-1)

class Tower(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim * 2, self.dim)
        self.layer2 = torch.nn.Linear(self.dim, 1)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, emb):
        return self.layer2(torch.nn.functional.relu(self.layer1(emb)))

class LogicRecModel(torch.nn.Module):
    def __init__(self, N_user, N_ent, N_rel, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.gamma = cfg.gamma
        self.intersection = IntersectionNet(cfg.emb_dim)

        self.e_embedding = torch.nn.Embedding(N_ent, cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(N_rel + 1, cfg.emb_dim)
        self.u_embedding = torch.nn.Embedding(N_user, cfg.emb_dim)

        self.expert_1 = Expert(cfg.emb_dim)
        self.expert_2 = Expert(cfg.emb_dim)
        self.gate_lqa = Gate(cfg.emb_dim)
        self.gate_rec = Gate(cfg.emb_dim)
        self.gate_both = Gate(cfg.emb_dim)

        self.expert_1_add = Expert(cfg.emb_dim)
        self.expert_2_add = Expert(cfg.emb_dim)
        self.gate_lqa_add = Gate(cfg.emb_dim)
        self.gate_rec_add = Gate(cfg.emb_dim)
        self.gate_both_add = Gate(cfg.emb_dim)

        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.u_embedding.weight.data)

        self.criterion = torch.nn.BCEWithLogitsLoss()

    def projection(self, emb_1, emb_2):
        return emb_1 + emb_2

    def _cal_logit_vec(self, a_emb, q_emb):
        distance = a_emb - q_emb
        logits = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logits
    
    def _mmoe(self, q_emb, uq_emb, qu_emb, part):
        if part == 0:
            expert_1_emb = self.expert_1(qu_emb)
            expert_2_emb = self.expert_2(qu_emb)
            gate_q = self.gate_lqa(q_emb)
            gate_uq = self.gate_rec(uq_emb)
            gate_qu = self.gate_both(qu_emb)
        elif part == 1:
            expert_1_emb = self.expert_1_add(qu_emb)
            expert_2_emb = self.expert_2_add(qu_emb)
            gate_q = self.gate_lqa_add(q_emb)
            gate_uq = self.gate_rec_add(uq_emb)
            gate_qu = self.gate_both_add(qu_emb)
        else:
            raise ValueError
        q_emb = expert_1_emb * gate_q[:, 0].unsqueeze(dim=-1) + expert_2_emb * gate_q[:, 1].unsqueeze(dim=-1)
        uq_emb = expert_1_emb * gate_uq[:, 0].unsqueeze(dim=-1) + expert_2_emb * gate_uq[:, 1].unsqueeze(dim=-1)
        qu_emb = expert_1_emb * gate_qu[:, 0].unsqueeze(dim=-1) + expert_2_emb * gate_qu[:, 1].unsqueeze(dim=-1)
        return q_emb, uq_emb, qu_emb

    def forward_1p(self, data):
        e_emb = self.e_embedding(data[:, 0])
        r_emb = self.r_embedding(data[:, 1])
        u_emb = self.u_embedding(data[:, 2])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        q_emb = self.projection(e_emb, r_emb)
        uq_emb = self.projection(u_emb, ur_emb)
        qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                                uq_emb.unsqueeze(dim=1)], dim=1))

        q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)

        logits_q = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
        logits_uq = self._cal_logit_vec(a_emb, uq_emb).unsqueeze(dim=1)
        logits_qu = self._cal_logit_vec(a_emb, qu_emb).unsqueeze(dim=1)

        return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_2p(self, data):
        e_emb = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        r_emb_2 = self.r_embedding(data[:, 2])
        u_emb = self.u_embedding(data[:, 3])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        q_emb = self.projection(self.projection(e_emb, r_emb_1), r_emb_2)
        uq_emb = self.projection(u_emb, ur_emb)
        qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                                uq_emb.unsqueeze(dim=1)], dim=1))

        q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)

        logits_q = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
        logits_uq = self._cal_logit_vec(a_emb, uq_emb).unsqueeze(dim=1)
        logits_qu = self._cal_logit_vec(a_emb, qu_emb).unsqueeze(dim=1)

        return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_3p(self, data):
        e_emb = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        r_emb_2 = self.r_embedding(data[:, 2])
        r_emb_3 = self.r_embedding(data[:, 3])
        u_emb = self.u_embedding(data[:, 4])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        q_emb = self.projection(self.projection(self.projection(e_emb, r_emb_1), r_emb_2), r_emb_3)
        uq_emb = self.projection(u_emb, ur_emb)
        qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                                uq_emb.unsqueeze(dim=1)], dim=1))

        q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)

        logits_q = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
        logits_uq = self._cal_logit_vec(a_emb, uq_emb).unsqueeze(dim=1)
        logits_qu = self._cal_logit_vec(a_emb, qu_emb).unsqueeze(dim=1)

        return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_2i(self, data):
        e_emb_1 = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        e_emb_2 = self.e_embedding(data[:, 2])
        r_emb_2 = self.r_embedding(data[:, 3])
        u_emb = self.u_embedding(data[:, 4])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        q_emb_1 = self.projection(e_emb_1, r_emb_1)
        q_emb_2 = self.projection(e_emb_2, r_emb_2)
        q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                q_emb_2.unsqueeze(dim=1)], dim=1))
        uq_emb = self.projection(u_emb, ur_emb)
        qu_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                q_emb_2.unsqueeze(dim=1), 
                                                uq_emb.unsqueeze(dim=1)], dim=1))
        
        q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)

        logits_q = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
        logits_uq = self._cal_logit_vec(a_emb, uq_emb).unsqueeze(dim=1)
        logits_qu = self._cal_logit_vec(a_emb, qu_emb).unsqueeze(dim=1)

        return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_3i(self, data):
        e_emb_1 = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        e_emb_2 = self.e_embedding(data[:, 2])
        r_emb_2 = self.r_embedding(data[:, 3])
        e_emb_3 = self.e_embedding(data[:, 4])
        r_emb_3 = self.r_embedding(data[:, 5])
        u_emb = self.u_embedding(data[:, 6])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        q_emb_1 = self.projection(e_emb_1, r_emb_1)
        q_emb_2 = self.projection(e_emb_2, r_emb_2)
        q_emb_3 = self.projection(e_emb_3, r_emb_3)
        q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                q_emb_2.unsqueeze(dim=1), 
                                                q_emb_3.unsqueeze(dim=1)], dim=1))
        uq_emb = self.projection(u_emb, ur_emb)
        qu_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                q_emb_2.unsqueeze(dim=1), 
                                                q_emb_3.unsqueeze(dim=1),
                                                uq_emb.unsqueeze(dim=1)], dim=1))

        q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)

        logits_q = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
        logits_uq = self._cal_logit_vec(a_emb, uq_emb).unsqueeze(dim=1)
        logits_qu = self._cal_logit_vec(a_emb, qu_emb).unsqueeze(dim=1)

        return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_pi(self, data):
        e_emb_1 = self.e_embedding(data[:, 0])
        r_emb_11 = self.r_embedding(data[:, 1])
        r_emb_12 = self.r_embedding(data[:, 2])
        e_emb_2 = self.e_embedding(data[:, 3])
        r_emb_2 = self.r_embedding(data[:, 4])
        u_emb = self.u_embedding(data[:, 5])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        q_emb_1 = self.projection(self.projection(e_emb_1, r_emb_11), r_emb_12)
        q_emb_2 = self.projection(e_emb_2, r_emb_2)
        q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                q_emb_2.unsqueeze(dim=1)], dim=1))
        uq_emb = self.projection(u_emb, ur_emb)
        qu_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                q_emb_2.unsqueeze(dim=1), 
                                                uq_emb.unsqueeze(dim=1)], dim=1))

        q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)

        logits_q = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
        logits_uq = self._cal_logit_vec(a_emb, uq_emb).unsqueeze(dim=1)
        logits_qu = self._cal_logit_vec(a_emb, qu_emb).unsqueeze(dim=1)

        return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_ip(self, data):
        e_emb_1 = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        e_emb_2 = self.e_embedding(data[:, 2])
        r_emb_2 = self.r_embedding(data[:, 3])
        r_emb = self.r_embedding(data[:, 4])
        u_emb = self.u_embedding(data[:, 5])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        q_emb_1 = self.projection(e_emb_1, r_emb_1)
        q_emb_2 = self.projection(e_emb_2, r_emb_2)
        q_emb_mid = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                    q_emb_2.unsqueeze(dim=1)], dim=1))
        q_emb = self.projection(q_emb_mid, r_emb)
        uq_emb = self.projection(u_emb, ur_emb)
        qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                                uq_emb.unsqueeze(dim=1)], dim=1))
        
        q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)

        logits_q = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
        logits_uq = self._cal_logit_vec(a_emb, uq_emb).unsqueeze(dim=1)
        logits_qu = self._cal_logit_vec(a_emb, qu_emb).unsqueeze(dim=1)

        return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_2u(self, data):
        e_emb_1 = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        e_emb_2 = self.e_embedding(data[:, 2])
        r_emb_2 = self.r_embedding(data[:, 3])
        u_emb = self.u_embedding(data[:, 4])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        q_emb_1 = self.projection(e_emb_1, r_emb_1)
        q_emb_2 = self.projection(e_emb_2, r_emb_2)
        uq_emb_1 = self.projection(u_emb, ur_emb)
        uq_emb_2 = self.projection(u_emb, ur_emb)
        qu_emb_1 = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                uq_emb_1.unsqueeze(dim=1)], dim=1))
        qu_emb_2 = self.intersection(torch.cat([q_emb_2.unsqueeze(dim=1), 
                                                uq_emb_1.unsqueeze(dim=1)], dim=1))

        q_emb_1, uq_emb_1, qu_emb_1 = self._mmoe(q_emb_1, uq_emb_1, qu_emb_1, part=0)
        q_emb_2, uq_emb_2, qu_emb_2 = self._mmoe(q_emb_2, uq_emb_2, qu_emb_2, part=0)

        logits_q_1 = self._cal_logit_vec(a_emb, q_emb_1).unsqueeze(dim=1)
        logits_q_2 = self._cal_logit_vec(a_emb, q_emb_2).unsqueeze(dim=1)
        logits_q = torch.cat([logits_q_1, logits_q_2], dim=-1).max(dim=-1)[0].unsqueeze(dim=1)
        logits_uq_1 = self._cal_logit_vec(a_emb, uq_emb_1).unsqueeze(dim=1)
        logits_uq_2 = self._cal_logit_vec(a_emb, uq_emb_2).unsqueeze(dim=1)
        logits_uq = torch.cat([logits_uq_1, logits_uq_2], dim=-1).max(dim=-1)[0].unsqueeze(dim=1)
        logits_qu_1 = self._cal_logit_vec(a_emb, qu_emb_1).unsqueeze(dim=1)
        logits_qu_2 = self._cal_logit_vec(a_emb, qu_emb_2).unsqueeze(dim=1)
        logits_qu = torch.cat([logits_qu_1, logits_qu_2], dim=-1).max(dim=-1)[0].unsqueeze(dim=1)

        return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_up(self, data):
        e_emb_1 = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        e_emb_2 = self.e_embedding(data[:, 2])
        r_emb_2 = self.r_embedding(data[:, 3])
        r_emb = self.r_embedding(data[:, 4])
        u_emb = self.u_embedding(data[:, 5])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        q_emb_1 = self.projection(self.projection(e_emb_1, r_emb_1), r_emb)
        q_emb_2 = self.projection(self.projection(e_emb_2, r_emb_2), r_emb)
        uq_emb_1 = self.projection(u_emb, ur_emb)
        uq_emb_2 = self.projection(u_emb, ur_emb)
        qu_emb_1 = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                uq_emb_1.unsqueeze(dim=1)], dim=1))
        qu_emb_2 = self.intersection(torch.cat([q_emb_2.unsqueeze(dim=1), 
                                                uq_emb_2.unsqueeze(dim=1)], dim=1))
        
        q_emb_1, uq_emb_1, qu_emb_1 = self._mmoe(q_emb_1, uq_emb_1, qu_emb_1, part=0)
        q_emb_2, uq_emb_2, qu_emb_2 = self._mmoe(q_emb_2, uq_emb_2, qu_emb_2, part=0)

        logits_q_1 = self._cal_logit_vec(a_emb, q_emb_1).unsqueeze(dim=1)
        logits_q_2 = self._cal_logit_vec(a_emb, q_emb_2).unsqueeze(dim=1)
        logits_q = torch.cat([logits_q_1, logits_q_2], dim=-1).max(dim=-1)[0].unsqueeze(dim=1)
        logits_uq_1 = self._cal_logit_vec(a_emb, uq_emb_1).unsqueeze(dim=1)
        logits_uq_2 = self._cal_logit_vec(a_emb, uq_emb_2).unsqueeze(dim=1)
        logits_uq = torch.cat([logits_uq_1, logits_uq_2], dim=-1).max(dim=-1)[0].unsqueeze(dim=1)
        logits_qu_1 = self._cal_logit_vec(a_emb, qu_emb_1).unsqueeze(dim=1)
        logits_qu_2 = self._cal_logit_vec(a_emb, qu_emb_2).unsqueeze(dim=1)
        logits_qu = torch.cat([logits_qu_1, logits_qu_2], dim=-1).max(dim=-1)[0].unsqueeze(dim=1)

        return torch.cat([logits_q, logits_uq, logits_qu], dim=1)


    def get_loss(self, data):
        data_1p = torch.index_select(data, 0, (data[:, -1] == 1).nonzero().squeeze(-1))
        data_2p = torch.index_select(data, 0, (data[:, -1] == 2).nonzero().squeeze(-1))
        data_3p = torch.index_select(data, 0, (data[:, -1] == 3).nonzero().squeeze(-1))
        data_2i = torch.index_select(data, 0, (data[:, -1] == 4).nonzero().squeeze(-1))
        data_3i = torch.index_select(data, 0, (data[:, -1] == 5).nonzero().squeeze(-1))

        preds = []
        labels = []
        if len(data_1p):
            preds_1p = self.forward_1p(data_1p)
            labels_1p = data_1p[:, -4:-1]
            preds.append(preds_1p)
            labels.append(labels_1p)
        if len(data_2p):
            preds_2p = self.forward_2p(data_2p)
            labels_2p = data_2p[:, -4:-1]
            preds.append(preds_2p)
            labels.append(labels_2p)
        if len(data_3p):
            preds_3p = self.forward_3p(data_3p)
            labels_3p = data_3p[:, -4:-1]
            preds.append(preds_3p)
            labels.append(labels_3p)
        if len(data_2i):
            preds_2i = self.forward_2i(data_2i)
            labels_2i = data_2i[:, -4:-1]
            preds.append(preds_2i)
            labels.append(labels_2i)
        if len(data_3i):
            preds_3i = self.forward_3i(data_3i)
            labels_3i = data_3i[:, -4:-1]
            preds.append(preds_3i)
            labels.append(labels_3i)

        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0).float()

        return self.criterion(preds, labels)

def get_rank(pos, logits, flt):
    ranking = torch.argsort(logits, descending=True)
    rank = (ranking == pos[0][-5]).nonzero().item() + 1
    ranking_better = ranking[:rank - 1]
    if flt != None:
        for e in flt:
            if (ranking_better == e).sum() == 1:
                rank -= 1
    return rank

def ndcg(true, rank, k):
    pred = torch.zeros(k)
    discount = 1 / torch.log2(torch.arange(k) + 2)
    if rank > k:
        return 0
    else:
        pred[rank - 1] = 1
        idcg = (true * discount).sum()
        dcg = (pred * discount).sum()
        return (dcg / idcg).item()

def evaluate(dataloader, model, device, train_dict, flag):
    r = []
    rr = []
    h10 = []
    h20 = []
    h50 = []
    ndcg10 = []
    ndcg20 = []
    ndcg50 = []
    true_10 = torch.zeros(10)
    true_10[0] = 1
    true_20 = torch.zeros(20)
    true_20[0] = 1
    true_50 = torch.zeros(50)
    true_50[0] = 1
    model.eval()
    with torch.no_grad():
        for possible, pos in dataloader:
            possible = possible[0].to(device)
            if flag == '1p':
                logits = model.forward_1p(possible)[:, -1]
                flt = train_dict[pos[0][2].item()]
            elif flag == '2p':
                logits = model.forward_2p(possible)[:, -1]
                flt = train_dict[pos[0][3].item()]
            elif flag == '3p':
                logits = model.forward_3p(possible)[:, -1]
                flt = train_dict[pos[0][4].item()]
            elif flag == '2i':
                logits = model.forward_2i(possible)[:, -1]
                flt = train_dict[pos[0][4].item()]
            elif flag == '3i':
                logits = model.forward_3i(possible)[:, -1]
                flt = train_dict[pos[0][6].item()]
            elif flag == 'pi':
                logits = model.forward_pi(possible)[:, -1]
                flt = train_dict[pos[0][5].item()]
            elif flag == 'ip':
                logits = model.forward_ip(possible)[:, -1]
                flt = train_dict[pos[0][5].item()]
            elif flag == '2u':
                logits = model.forward_2u(possible)[:, -1]
                flt = train_dict[pos[0][4].item()]
            elif flag == 'up':
                logits = model.forward_up(possible)[:, -1]
                flt = train_dict[pos[0][5].item()]
            else:
                raise ValueError
            rank = get_rank(pos, logits, flt)
            r.append(rank)
            rr.append(1/rank)
            if rank <= 10:
                h10.append(1)
            else:
                h10.append(0)
            if rank <= 20:
                h20.append(1)
            else:
                h20.append(0)
            if rank <= 50:
                h50.append(1)
            else:
                h50.append(0)

            ndcg10.append(ndcg(true_10, rank, 10))
            ndcg20.append(ndcg(true_20, rank, 20))
            ndcg50.append(ndcg(true_50, rank, 50))
    
    results = [r, rr, h10, h20, h50, ndcg10, ndcg20, ndcg50]
    
    r = int(sum(results[0])/len(results[0]))
    rr = round(sum(results[1])/len(results[1]), 3)
    h10 = round(sum(results[2])/len(results[2]), 3)
    h20 = round(sum(results[3])/len(results[3]), 3)
    h50 = round(sum(results[4])/len(results[4]), 3)
    ndcg10 = round(sum(results[5])/len(results[5]), 3)
    ndcg20 = round(sum(results[6])/len(results[6]), 3)
    ndcg50 = round(sum(results[7])/len(results[7]), 3)
    
    print(f'#{flag}#  MR: {r}, MRR: {rr}, Hit@10: {h10}, Hit@20: {h20}, Hit@50: {h50}, nDCG@10: {ndcg10}, nDCG@20: {ndcg20}, nDCG@50: {ndcg50}', flush=True)
    
    return r, rr, h10, h20, h50, ndcg10, ndcg20, ndcg50

def iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    # Supporting Arguments
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--cached', default=0, type=int)
    parser.add_argument('--dataset', default='amazon-book', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--tolerance', default=5, type=int)
    parser.add_argument('--valid_interval', default=5000, type=int)
    # Tunable Arguments
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--bs', default=1024, type=int)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--gamma', default=12, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    input_path = cfg.data_root + cfg.dataset
    save_root = f'../tmp/{cfg.dataset}_{cfg.emb_dim}_{cfg.lr}_{cfg.bs}_{cfg.wd}_{cfg.gamma}/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    N_rel, N_item, N_ent, N_user, filters = read_data(input_path)
    test_1p = load_obj(input_path + '/input/1p_test.pkl')
    test_2p = load_obj(input_path + '/input/2p_test.pkl')
    test_3p = load_obj(input_path + '/input/3p_test.pkl')
    test_2i = load_obj(input_path + '/input/2i_test.pkl')
    test_3i = load_obj(input_path + '/input/3i_test.pkl')
    test_pi = load_obj(input_path + '/input/pi_test.pkl')
    test_ip = load_obj(input_path + '/input/ip_test.pkl')
    test_2u = load_obj(input_path + '/input/2u_test.pkl')
    test_up = load_obj(input_path + '/input/up_test.pkl')
    
    if not cfg.cached:
        train_1p = load_obj(input_path + '/input/1p_train.pkl')
        print('Read 1p done')
        train_2p = load_obj(input_path + '/input/2p_train.pkl')
        print('Read 2p done')
        train_3p = load_obj(input_path + '/input/3p_train.pkl')
        print('Read 3p done')
        train_2i = load_obj(input_path + '/input/2i_train.pkl')
        print('Read 2i done')
        train_3i = load_obj(input_path + '/input/3i_train.pkl')
        print('Read 3i done')
        data = combine_data(train_1p, train_2p, train_3p, train_2i, train_3i, N_item, k=cfg.k)
        torch.save(data, input_path + '/input/input.th')
    else:
        data = torch.load(input_path + '/input/input.th')

    lqa_dataset_train = LQADatasetTrain(N_ent, data, cfg)
    lqa_dataset_1p_valid = LQADatasetTest(N_item, test_1p, cfg, stage='valid', query_type='1p')
    lqa_dataset_1p_test = LQADatasetTest(N_item, test_1p, cfg, stage='test', query_type='1p')
    lqa_dataset_2p_valid = LQADatasetTest(N_item, test_2p, cfg, stage='valid', query_type='2p')
    lqa_dataset_2p_test = LQADatasetTest(N_item, test_2p, cfg, stage='test', query_type='2p')
    lqa_dataset_3p_valid = LQADatasetTest(N_item, test_3p, cfg, stage='valid', query_type='3p')
    lqa_dataset_3p_test = LQADatasetTest(N_item, test_3p, cfg, stage='test', query_type='3p')
    lqa_dataset_2i_valid = LQADatasetTest(N_item, test_2i, cfg, stage='valid', query_type='2i')
    lqa_dataset_2i_test = LQADatasetTest(N_item, test_2i, cfg, stage='test', query_type='2i')
    lqa_dataset_3i_valid = LQADatasetTest(N_item, test_3i, cfg, stage='valid', query_type='3i')
    lqa_dataset_3i_test = LQADatasetTest(N_item, test_3i, cfg, stage='test', query_type='3i')
    lqa_dataset_pi_valid = LQADatasetTest(N_item, test_pi, cfg, stage='valid', query_type='pi')
    lqa_dataset_pi_test = LQADatasetTest(N_item, test_pi, cfg, stage='test', query_type='pi')
    lqa_dataset_ip_valid = LQADatasetTest(N_item, test_ip, cfg, stage='valid', query_type='ip')
    lqa_dataset_ip_test = LQADatasetTest(N_item, test_ip, cfg, stage='test', query_type='ip')
    lqa_dataset_2u_valid = LQADatasetTest(N_item, test_2u, cfg, stage='valid', query_type='2u')
    lqa_dataset_2u_test = LQADatasetTest(N_item, test_2u, cfg, stage='test', query_type='2u')
    lqa_dataset_up_valid = LQADatasetTest(N_item, test_up, cfg, stage='valid', query_type='up')
    lqa_dataset_up_test = LQADatasetTest(N_item, test_up, cfg, stage='test', query_type='up')

    lqa_dataloader_train = torch.utils.data.DataLoader(dataset=lqa_dataset_train,
                                                       batch_size=cfg.bs,
                                                       num_workers=cfg.num_workers,
                                                       shuffle=True,
                                                       drop_last=True,
                                                       collate_fn=my_collate_fn)
    lqa_dataloader_1p_valid = torch.utils.data.DataLoader(dataset=lqa_dataset_1p_valid,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_1p_test = torch.utils.data.DataLoader(dataset=lqa_dataset_1p_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_2p_valid = torch.utils.data.DataLoader(dataset=lqa_dataset_2p_valid,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_2p_test = torch.utils.data.DataLoader(dataset=lqa_dataset_2p_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_3p_valid = torch.utils.data.DataLoader(dataset=lqa_dataset_3p_valid,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_3p_test = torch.utils.data.DataLoader(dataset=lqa_dataset_3p_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_2i_valid = torch.utils.data.DataLoader(dataset=lqa_dataset_2i_valid,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_2i_test = torch.utils.data.DataLoader(dataset=lqa_dataset_2i_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_3i_valid = torch.utils.data.DataLoader(dataset=lqa_dataset_3i_valid,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_3i_test = torch.utils.data.DataLoader(dataset=lqa_dataset_3i_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_pi_valid = torch.utils.data.DataLoader(dataset=lqa_dataset_pi_valid,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_pi_test = torch.utils.data.DataLoader(dataset=lqa_dataset_pi_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_ip_valid = torch.utils.data.DataLoader(dataset=lqa_dataset_ip_valid,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_ip_test = torch.utils.data.DataLoader(dataset=lqa_dataset_ip_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_2u_valid = torch.utils.data.DataLoader(dataset=lqa_dataset_2u_valid,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_2u_test = torch.utils.data.DataLoader(dataset=lqa_dataset_2u_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_up_valid = torch.utils.data.DataLoader(dataset=lqa_dataset_up_valid,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    lqa_dataloader_up_test = torch.utils.data.DataLoader(dataset=lqa_dataset_up_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=False,
                                                drop_last=False)
    
    model = LogicRecModel(N_user, N_ent, N_rel, cfg)
    model = model.to(device)

    if cfg.verbose:
        lqa_dataloader_1p_valid = tqdm.tqdm(lqa_dataloader_1p_valid)
        lqa_dataloader_1p_test = tqdm.tqdm(lqa_dataloader_1p_test)
        lqa_dataloader_2p_valid = tqdm.tqdm(lqa_dataloader_2p_valid)
        lqa_dataloader_2p_test = tqdm.tqdm(lqa_dataloader_2p_test)
        lqa_dataloader_3p_valid = tqdm.tqdm(lqa_dataloader_3p_valid)
        lqa_dataloader_3p_test = tqdm.tqdm(lqa_dataloader_3p_test)
        lqa_dataloader_2i_valid = tqdm.tqdm(lqa_dataloader_2i_valid)
        lqa_dataloader_2i_test = tqdm.tqdm(lqa_dataloader_2i_test)
        lqa_dataloader_3i_valid = tqdm.tqdm(lqa_dataloader_3i_valid)
        lqa_dataloader_3i_test = tqdm.tqdm(lqa_dataloader_3i_test)
        lqa_dataloader_pi_valid = tqdm.tqdm(lqa_dataloader_pi_valid)
        lqa_dataloader_pi_test = tqdm.tqdm(lqa_dataloader_pi_test)
        lqa_dataloader_ip_valid = tqdm.tqdm(lqa_dataloader_ip_valid)
        lqa_dataloader_ip_test = tqdm.tqdm(lqa_dataloader_ip_test)
        lqa_dataloader_2u_valid = tqdm.tqdm(lqa_dataloader_2u_valid)
        lqa_dataloader_2u_test = tqdm.tqdm(lqa_dataloader_2u_test)
        lqa_dataloader_up_valid = tqdm.tqdm(lqa_dataloader_up_valid)
        lqa_dataloader_up_test = tqdm.tqdm(lqa_dataloader_up_test)
    valid_dataloaders = [lqa_dataloader_1p_valid, 
                        lqa_dataloader_2p_valid,
                        lqa_dataloader_3p_valid,
                        lqa_dataloader_2i_valid,
                        lqa_dataloader_3i_valid,
                        lqa_dataloader_pi_valid,
                        lqa_dataloader_ip_valid,
                        lqa_dataloader_2u_valid,
                        lqa_dataloader_up_valid
                        ]
    test_dataloaders = [lqa_dataloader_1p_test, 
                        lqa_dataloader_2p_test,
                        lqa_dataloader_3p_test,
                        lqa_dataloader_2i_test,
                        lqa_dataloader_3i_test,
                        lqa_dataloader_pi_test,
                        lqa_dataloader_ip_test,
                        lqa_dataloader_2u_test,
                        lqa_dataloader_up_test
                        ]
    query_types = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip', '2u', 'up']
    lqa_dataloader_train = iterator(lqa_dataloader_train)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    
    if cfg.verbose:
        ranger = tqdm.tqdm(range(cfg.max_steps))
    else:
        ranger = range(cfg.max_steps)
    tolerance = cfg.tolerance
    max_value = 0
    avg_loss = []
    for step in ranger:
        model.train()

        loss = model.get_loss(next(lqa_dataloader_train).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss.append(loss.item())
        
        if (step + 1) % cfg.valid_interval == 0:
            print(f'Loss: {round(sum(avg_loss) / len(avg_loss), 4)}', flush=True)
            avg_loss = []
            print('Validating...', flush=True)
            results = []
            for i in range(len(query_types)):
                flag = query_types[i]
                r, rr, h10, h20, h50, ndcg10, ndcg20, ndcg50 = evaluate(valid_dataloaders[i], model, device, filters, flag)
                results.append([r, rr, h10, h20, h50, ndcg10, ndcg20, ndcg50])
                
            mean_results = torch.tensor(results).mean(dim=0).numpy().tolist()
            print(f'Mean: MR: {int(mean_results[0])}, \
                    MRR: {round(mean_results[1], 3)}, \
                    Hit@10: {round(mean_results[2], 3)}, \
                    Hit@20: {round(mean_results[3], 3)}, \
                    Hit@50: {round(mean_results[4], 3)}, \
                    nDCG@10: {round(mean_results[5], 3)}, \
                    nDCG@20: {round(mean_results[6], 3)}, \
                    nDCG@50: {round(mean_results[7], 3)}', flush=True)
            value = round(mean_results[1], 3)
            if value >= max_value:
                max_value = value
                tolerance = cfg.tolerance
            else:
                tolerance -= 1
            print(f'Tolerance: {tolerance}', flush=True)
            torch.save(model.state_dict(), save_root + (str(step + 1)))

        if (tolerance == 0) or ((step + 1) == cfg.max_steps):
            print('Testing...', flush=True)
            print(f'Best performance at epoch {step - cfg.tolerance * cfg.valid_interval + 1}', flush=True)
            model.load_state_dict(torch.load(save_root + str(step - cfg.tolerance * cfg.valid_interval + 1)))
            results = []
            for i in range(len(query_types)):
                flag = query_types[i]
                r, rr, h10, h20, h50, ndcg10, ndcg20, ndcg50 = evaluate(test_dataloaders[i], model, device, filters, flag)
                results.append([r, rr, h10, h20, h50, ndcg10, ndcg20, ndcg50])
            mean_results = torch.tensor(results).mean(dim=0).numpy().tolist()
            print(f'Mean: MR: {int(mean_results[0])}, \
                    MRR: {round(mean_results[1], 3)}, \
                    Hit@10: {round(mean_results[2], 3)}, \
                    Hit@20: {round(mean_results[3], 3)}, \
                    Hit@50: {round(mean_results[4], 3)}, \
                    nDCG@10: {round(mean_results[5], 3)}, \
                    nDCG@20: {round(mean_results[6], 3)}, \
                    nDCG@50: {round(mean_results[7], 3)}', flush=True)
            break
