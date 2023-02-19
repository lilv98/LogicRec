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
    all_i = []
    counter = 0
    with open(path + '/item_list.txt') as f:
        for line in f:
            counter += 1
            if counter > 1:
                all_i.append(line)
    N_item = int(all_i[-1].strip('\n').split(' ')[1]) + 1

    all_e = []
    counter = 0
    with open(path + '/entity_list.txt') as f:
        for line in f:
            counter += 1
            if counter > 1:
                all_e.append(line)
    N_ent = len(all_e)
    assert all_e[N_item - 1].strip('\n').split(' ')[0] == all_i[-1].strip('\n').split(' ')[2]
    
    N_rel = -1
    with open(path + '/relation_list.txt') as f:
        for line in f:
            N_rel += 1

    N_user = -1
    with open(path + '/user_list.txt') as f:
        for line in f:
            N_user += 1

    print(f'N_item: {N_item}')
    print(f'N_ent: {N_ent}')
    print(f'N_rel: {N_rel}')
    print(f'N_user: {N_user}')

    filters = {}
    with open(path + '/train.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            filters[int(line[0])] = set([int(x) for x in line[1:]])

    return N_rel, N_item, N_ent, N_user, filters

def combine_data(data_1p, data_2p, data_3p, data_2i, data_3i, N_item, k):
    item_candidates = set(range(N_item))
    ret = []
    counter = 0
    # 1p: e1, r1, u, ph, ph, ph, ph, i, lqa, rec, both, 1
    for query in tqdm.tqdm(data_1p):
        # counter += 1
        # if counter < 10001:
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
        self.num_ng = cfg.num_ng
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
        self.num_ng = cfg.num_ng
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

class BoxOffsetIntersection(torch.nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, self.dim)
        self.layer2 = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.xavier_uniform_(self.layer1.weight.data)
        torch.nn.init.xavier_uniform_(self.layer2.weight.data)

    def forward(self, embeddings):
        layer1_act = torch.nn.functional.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=1)
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=1)
        return offset * gate

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)

class BetaIntersection(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, self.dim // 2)
        self.layer2 = torch.nn.Linear(self.dim // 2, self.dim // 2)

        torch.nn.init.xavier_uniform_(self.layer1.weight.data)
        torch.nn.init.xavier_uniform_(self.layer2.weight.data)

    def forward(self, alpha_embeddings, beta_embeddings):
        all_embeddings = torch.cat([alpha_embeddings, beta_embeddings], dim=-1)
        # (batch_size, num_conj, dim)
        layer1_act = torch.nn.functional.relu(self.layer1(all_embeddings))
        # (batch_size, num_conj, dim)
        attention = torch.nn.functional.softmax(self.layer2(layer1_act), dim=1)
        alpha_embedding = torch.sum(attention * alpha_embeddings, dim=1)
        beta_embedding = torch.sum(attention * beta_embeddings, dim=1)

        return alpha_embedding, beta_embedding

class BetaProjection(torch.nn.Module):
    
    def __init__(self, emb_dim, regularizer):
        super(BetaProjection, self).__init__()
        self.emb_dim = emb_dim
        self.layer1 = torch.nn.Linear(self.emb_dim * 2, self.emb_dim) 
        self.layer2 = torch.nn.Linear(self.emb_dim, self.emb_dim) 
        self.regularizer = regularizer
        torch.nn.init.xavier_uniform_(self.layer1.weight.data)
        torch.nn.init.xavier_uniform_(self.layer2.weight.data)

    def forward(self, e_embedding, r_embedding):
        x = torch.cat([e_embedding, r_embedding], dim=-1)
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = self.regularizer(x)
        return x

class Expert(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, self.dim)
        # self.layer2 = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        # torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, emb):
        # return self.layer2(torch.nn.functional.relu(self.layer1(emb)))
        return torch.nn.functional.relu(self.layer1(emb))

class Gate(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, 3)
        # self.layer1 = torch.nn.Linear(self.dim, self.dim)
        # self.layer2 = torch.nn.Linear(self.dim, 3)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        # torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, emb):
        # return torch.softmax(self.layer2(torch.nn.functional.relu(self.layer1(emb))), dim=-1)
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
        self.which = cfg.which
        self.emb_dim = cfg.emb_dim
        self.num_ng = cfg.num_ng
        self.gamma = cfg.gamma

        if self.which == 'vec':
            self.emb_dim = cfg.emb_dim
            self.intersection = IntersectionNet(self.emb_dim)
        
        elif 'box' in self.which:
            self.emb_dim = cfg.emb_dim
            self.intersection = IntersectionNet(self.emb_dim)
            self.cen = cfg.cen
            self.offset_embedding = torch.nn.Embedding(N_rel + 1, self.emb_dim)
            self.offset_net = BoxOffsetIntersection(self.emb_dim)

        elif 'beta' in self.which:
            self.emb_dim = cfg.emb_dim * 2
            self.intersection = BetaIntersection(self.emb_dim)
            self.regularizer = Regularizer(1, 0.05, 1e9)
            self.beta_proj_net = BetaProjection(self.emb_dim, self.regularizer)

        else:
            raise ValueError

        self.e_embedding = torch.nn.Embedding(N_ent, self.emb_dim)
        self.r_embedding = torch.nn.Embedding(N_rel + 1, self.emb_dim)
        self.u_embedding = torch.nn.Embedding(N_user, self.emb_dim)

        if 'mmoe' in self.which:
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
        if self.which == 'beta':
            return self.beta_proj_net(emb_1, emb_2)
        else:
            return emb_1 + emb_2

    def _cal_logit_vec(self, a_emb, q_emb):
        distance = a_emb - q_emb
        logits = self.gamma - torch.norm(distance, p=1, dim=-1)
        return logits

    def _cal_logit_box(self, a_emb, q_emb, q_emb_offset):
        delta = (a_emb - q_emb).abs()
        distance_out = torch.nn.functional.relu(delta - q_emb_offset)
        distance_in = torch.min(delta, q_emb_offset)
        logits = self.gamma - torch.norm(distance_out, p=1, dim=-1) - self.cen * torch.norm(distance_in, p=1, dim=-1)
        return logits

    def _cal_logit_beta(self, a_emb, query_dist):
        a_emb = self.regularizer(a_emb)
        alpha_embedding, beta_embedding = torch.chunk(a_emb, 2, dim=-1)
        entity_dist = torch.distributions.beta.Beta(alpha_embedding, beta_embedding)
        logits = self.gamma - torch.norm(torch.distributions.kl.kl_divergence(entity_dist, query_dist), p=1, dim=-1)
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

        if self.which == 'vec':
            q_emb_1 = self.projection(e_emb, r_emb)
            q_emb_2 = self.projection(u_emb, ur_emb)
            q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                 q_emb_2.unsqueeze(dim=1)], dim=1))
            logits = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
            return torch.cat([logits, logits, logits], dim=1)

        elif 'box' in self.which:
            r_emb_offset = self.offset_embedding(data[:, 1])
            ur_emb_offset = self.offset_embedding.weight[-1].unsqueeze(dim=0)

            q_emb = self.projection(e_emb, r_emb)
            uq_emb = self.projection(u_emb, ur_emb)
            qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                                  uq_emb.unsqueeze(dim=1)], dim=1))

            q_emb_offset = self.projection(torch.zeros_like(r_emb), r_emb_offset)
            uq_emb_offset = self.projection(torch.zeros_like(r_emb), ur_emb_offset)
            qu_emb_offset = self.offset_net(torch.cat([q_emb_offset.unsqueeze(dim=1), 
                                                       uq_emb_offset.unsqueeze(dim=1)], dim=1))
            
            if 'mmoe' in self.which:
                q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)
                q_emb_offset, uq_emb_offset, qu_emb_offset = self._mmoe(q_emb_offset, uq_emb_offset, qu_emb_offset, part=1)

            logits_q = self._cal_logit_box(a_emb, q_emb, q_emb_offset).unsqueeze(dim=1)
            logits_uq = self._cal_logit_box(a_emb, uq_emb, uq_emb_offset).unsqueeze(dim=1)
            logits_qu = self._cal_logit_box(a_emb, qu_emb, qu_emb_offset).unsqueeze(dim=1)

            return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

        elif 'beta' in self.which:
            q_emb = self.projection(self.regularizer(e_emb), r_emb)
            uq_emb = self.projection(self.regularizer(u_emb), ur_emb.expand_as(u_emb))
            q_emb_alpha, q_emb_beta = torch.chunk(q_emb, 2, dim=-1)
            uq_emb_alpha, uq_emb_beta = torch.chunk(uq_emb, 2, dim=-1)

            qu_emb_alpha = torch.cat([q_emb_alpha.unsqueeze(dim=1), 
                                      uq_emb_alpha.unsqueeze(dim=1)], dim=1)
            qu_emb_beta = torch.cat([q_emb_beta.unsqueeze(dim=1), 
                                     uq_emb_beta.unsqueeze(dim=1)], dim=1)
            qu_emb_alpha, qu_emb_beta = self.intersection(qu_emb_alpha, qu_emb_beta)

            if 'mmoe' in self.which:
                q_emb_alpha, uq_emb_alpha, qu_emb_alpha = self._mmoe(q_emb_alpha, uq_emb_alpha, qu_emb_alpha, part=0)
                q_emb_beta, uq_emb_beta, qu_emb_beta = self._mmoe(q_emb_beta, uq_emb_beta, qu_emb_beta, part=1)

            q_dist = torch.distributions.beta.Beta(self.regularizer(q_emb_alpha), self.regularizer(q_emb_beta))
            uq_dist = torch.distributions.beta.Beta(self.regularizer(uq_emb_alpha), self.regularizer(uq_emb_beta))
            qu_dist = torch.distributions.beta.Beta(self.regularizer(qu_emb_alpha), self.regularizer(qu_emb_beta))
            logits_q = self._cal_logit_beta(a_emb, q_dist).unsqueeze(dim=1)
            logits_uq = self._cal_logit_beta(a_emb, uq_dist).unsqueeze(dim=1)
            logits_qu = self._cal_logit_beta(a_emb, qu_dist).unsqueeze(dim=1)
            
            return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_2p(self, data):
        e_emb = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        r_emb_2 = self.r_embedding(data[:, 2])
        u_emb = self.u_embedding(data[:, 3])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        if self.which == 'vec':
            q_emb_1 = self.projection(self.projection(e_emb, r_emb_1), r_emb_2)
            q_emb_2 = self.projection(u_emb, ur_emb)
            q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                 q_emb_2.unsqueeze(dim=1)], dim=1))
            logits = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
            return torch.cat([logits, logits, logits], dim=1)
        
        elif 'box' in self.which:
            r_emb_1_offset = self.offset_embedding(data[:, 1])
            r_emb_2_offset = self.offset_embedding(data[:, 2])
            ur_emb_offset = self.offset_embedding.weight[-1].unsqueeze(dim=0)

            q_emb = self.projection(self.projection(e_emb, r_emb_1), r_emb_2)
            uq_emb = self.projection(u_emb, ur_emb)
            qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                                  uq_emb.unsqueeze(dim=1)], dim=1))
            
            q_emb_offset = self.projection(self.projection(torch.zeros_like(r_emb_1), r_emb_1_offset), r_emb_2_offset)
            uq_emb_offset = self.projection(torch.zeros_like(r_emb_1), ur_emb_offset)
            qu_emb_offset = self.offset_net(torch.cat([q_emb_offset.unsqueeze(dim=1), 
                                                       uq_emb_offset.unsqueeze(dim=1)], dim=1))
            
            if 'mmoe' in self.which:
                q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)
                q_emb_offset, uq_emb_offset, qu_emb_offset = self._mmoe(q_emb_offset, uq_emb_offset, qu_emb_offset, part=1)

            logits_q = self._cal_logit_box(a_emb, q_emb, q_emb_offset).unsqueeze(dim=1)
            logits_uq = self._cal_logit_box(a_emb, uq_emb, uq_emb_offset).unsqueeze(dim=1)
            logits_qu = self._cal_logit_box(a_emb, qu_emb, qu_emb_offset).unsqueeze(dim=1)
            
            return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

        elif 'beta' in self.which:
            q_emb = self.projection(self.projection(self.regularizer(e_emb), r_emb_1), r_emb_2)
            uq_emb = self.projection(self.regularizer(u_emb), ur_emb.expand_as(u_emb))
            q_emb_alpha, q_emb_beta = torch.chunk(q_emb, 2, dim=-1)
            uq_emb_alpha, uq_emb_beta = torch.chunk(uq_emb, 2, dim=-1)

            qu_emb_alpha = torch.cat([q_emb_alpha.unsqueeze(dim=1), 
                                      uq_emb_alpha.unsqueeze(dim=1)], dim=1)
            qu_emb_beta = torch.cat([q_emb_beta.unsqueeze(dim=1), 
                                     uq_emb_beta.unsqueeze(dim=1)], dim=1)
            qu_emb_alpha, qu_emb_beta = self.intersection(qu_emb_alpha, qu_emb_beta)

            if 'mmoe' in self.which:
                q_emb_alpha, uq_emb_alpha, qu_emb_alpha = self._mmoe(q_emb_alpha, uq_emb_alpha, qu_emb_alpha, part=0)
                q_emb_beta, uq_emb_beta, qu_emb_beta = self._mmoe(q_emb_beta, uq_emb_beta, qu_emb_beta, part=1)

            q_dist = torch.distributions.beta.Beta(self.regularizer(q_emb_alpha), self.regularizer(q_emb_beta))
            uq_dist = torch.distributions.beta.Beta(self.regularizer(uq_emb_alpha), self.regularizer(uq_emb_beta))
            qu_dist = torch.distributions.beta.Beta(self.regularizer(qu_emb_alpha), self.regularizer(qu_emb_beta))
            logits_q = self._cal_logit_beta(a_emb, q_dist).unsqueeze(dim=1)
            logits_uq = self._cal_logit_beta(a_emb, uq_dist).unsqueeze(dim=1)
            logits_qu = self._cal_logit_beta(a_emb, qu_dist).unsqueeze(dim=1)
            
            return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_3p(self, data):
        e_emb = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        r_emb_2 = self.r_embedding(data[:, 2])
        r_emb_3 = self.r_embedding(data[:, 3])
        u_emb = self.u_embedding(data[:, 4])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        if self.which == 'vec':
            q_emb_1 = self.projection(self.projection(self.projection(e_emb, r_emb_1), r_emb_2), r_emb_3)
            q_emb_2 = self.projection(u_emb, ur_emb)
            q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                 q_emb_2.unsqueeze(dim=1)], dim=1))
            logits = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
            return torch.cat([logits, logits, logits], dim=1)

        elif 'box' in self.which:
            r_emb_1_offset = self.offset_embedding(data[:, 1])
            r_emb_2_offset = self.offset_embedding(data[:, 2])
            r_emb_3_offset = self.offset_embedding(data[:, 3])
            ur_emb_offset = self.offset_embedding.weight[-1].unsqueeze(dim=0)

            q_emb = self.projection(self.projection(self.projection(e_emb, r_emb_1), r_emb_2), r_emb_3)
            uq_emb = self.projection(u_emb, ur_emb)
            qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                                  uq_emb.unsqueeze(dim=1)], dim=1))
            
            q_emb_offset = self.projection(self.projection(self.projection(torch.zeros_like(r_emb_1), r_emb_1_offset), r_emb_2_offset), r_emb_3_offset)
            uq_emb_offset = self.projection(torch.zeros_like(r_emb_1), ur_emb_offset)
            qu_emb_offset = self.offset_net(torch.cat([q_emb_offset.unsqueeze(dim=1), 
                                                       uq_emb_offset.unsqueeze(dim=1)], dim=1))

            if 'mmoe' in self.which:
                q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)
                q_emb_offset, uq_emb_offset, qu_emb_offset = self._mmoe(q_emb_offset, uq_emb_offset, qu_emb_offset, part=1)

            logits_q = self._cal_logit_box(a_emb, q_emb, q_emb_offset).unsqueeze(dim=1)
            logits_uq = self._cal_logit_box(a_emb, uq_emb, uq_emb_offset).unsqueeze(dim=1)
            logits_qu = self._cal_logit_box(a_emb, qu_emb, qu_emb_offset).unsqueeze(dim=1)
            
            return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

        elif 'beta' in self.which:
            q_emb = self.projection(self.projection(self.projection(self.regularizer(e_emb), r_emb_1), r_emb_2), r_emb_3)
            uq_emb = self.projection(self.regularizer(u_emb), ur_emb.expand_as(u_emb))
            q_emb_alpha, q_emb_beta = torch.chunk(q_emb, 2, dim=-1)
            uq_emb_alpha, uq_emb_beta = torch.chunk(uq_emb, 2, dim=-1)

            qu_emb_alpha = torch.cat([q_emb_alpha.unsqueeze(dim=1), 
                                      uq_emb_alpha.unsqueeze(dim=1)], dim=1)
            qu_emb_beta = torch.cat([q_emb_beta.unsqueeze(dim=1), 
                                     uq_emb_beta.unsqueeze(dim=1)], dim=1)
            qu_emb_alpha, qu_emb_beta = self.intersection(qu_emb_alpha, qu_emb_beta)

            if 'mmoe' in self.which:
                q_emb_alpha, uq_emb_alpha, qu_emb_alpha = self._mmoe(q_emb_alpha, uq_emb_alpha, qu_emb_alpha, part=0)
                q_emb_beta, uq_emb_beta, qu_emb_beta = self._mmoe(q_emb_beta, uq_emb_beta, qu_emb_beta, part=1)

            q_dist = torch.distributions.beta.Beta(self.regularizer(q_emb_alpha), self.regularizer(q_emb_beta))
            uq_dist = torch.distributions.beta.Beta(self.regularizer(uq_emb_alpha), self.regularizer(uq_emb_beta))
            qu_dist = torch.distributions.beta.Beta(self.regularizer(qu_emb_alpha), self.regularizer(qu_emb_beta))
            logits_q = self._cal_logit_beta(a_emb, q_dist).unsqueeze(dim=1)
            logits_uq = self._cal_logit_beta(a_emb, uq_dist).unsqueeze(dim=1)
            logits_qu = self._cal_logit_beta(a_emb, qu_dist).unsqueeze(dim=1)
            
            return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

    def forward_2i(self, data):
        e_emb_1 = self.e_embedding(data[:, 0])
        r_emb_1 = self.r_embedding(data[:, 1])
        e_emb_2 = self.e_embedding(data[:, 2])
        r_emb_2 = self.r_embedding(data[:, 3])
        u_emb = self.u_embedding(data[:, 4])
        ur_emb = self.r_embedding.weight[-1].unsqueeze(dim=0)
        a_emb = self.e_embedding(data[:, -5])

        if self.which == 'vec':
            q_emb_1 = self.projection(e_emb_1, r_emb_1)
            q_emb_2 = self.projection(e_emb_2, r_emb_2)
            q_emb_3 = self.projection(u_emb, ur_emb)
            q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                 q_emb_2.unsqueeze(dim=1), 
                                                 q_emb_3.unsqueeze(dim=1)], dim=1))
            logits = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
            return torch.cat([logits, logits, logits], dim=1)

        elif 'box' in self.which:
            r_emb_1_offset = self.offset_embedding(data[:, 1])
            r_emb_2_offset = self.offset_embedding(data[:, 3])
            ur_emb_offset = self.offset_embedding.weight[-1].unsqueeze(dim=0)
            
            q_emb_1 = self.projection(e_emb_1, r_emb_1)
            q_emb_2 = self.projection(e_emb_2, r_emb_2)
            q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                 q_emb_2.unsqueeze(dim=1)], dim=1))
            uq_emb = self.projection(u_emb, ur_emb)
            qu_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                  q_emb_2.unsqueeze(dim=1), 
                                                  uq_emb.unsqueeze(dim=1)], dim=1))
            
            q_emb_1_offset = self.projection(torch.zeros_like(r_emb_1), r_emb_1_offset)
            q_emb_2_offset = self.projection(torch.zeros_like(r_emb_2), r_emb_2_offset)
            q_emb_offset = self.intersection(torch.cat([q_emb_1_offset.unsqueeze(dim=1), 
                                                        q_emb_2_offset.unsqueeze(dim=1)], dim=1))
            uq_emb_offset = self.projection(torch.zeros_like(r_emb_1), ur_emb_offset)
            qu_emb_offset = self.intersection(torch.cat([q_emb_1_offset.unsqueeze(dim=1), 
                                                         q_emb_2_offset.unsqueeze(dim=1),
                                                         uq_emb_offset.unsqueeze(dim=1)], dim=1))

            if 'mmoe' in self.which:
                q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)
                q_emb_offset, uq_emb_offset, qu_emb_offset = self._mmoe(q_emb_offset, uq_emb_offset, qu_emb_offset, part=1)

            logits_q = self._cal_logit_box(a_emb, q_emb, q_emb_offset).unsqueeze(dim=1)
            logits_uq = self._cal_logit_box(a_emb, uq_emb, uq_emb_offset).unsqueeze(dim=1)
            logits_qu = self._cal_logit_box(a_emb, qu_emb, qu_emb_offset).unsqueeze(dim=1)
            
            return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

        elif 'beta' in self.which:
            q_emb_1 = self.projection(self.regularizer(e_emb_1), r_emb_1)
            q_emb_2 = self.projection(self.regularizer(e_emb_2), r_emb_2)
            uq_emb = self.projection(self.regularizer(u_emb), ur_emb.expand_as(u_emb))

            q_emb_1_alpha, q_emb_1_beta = torch.chunk(q_emb_1, 2, dim=-1)
            q_emb_2_alpha, q_emb_2_beta = torch.chunk(q_emb_2, 2, dim=-1)
            uq_emb_alpha, uq_emb_beta = torch.chunk(uq_emb, 2, dim=-1)

            q_emb_alpha = torch.cat([q_emb_1_alpha.unsqueeze(dim=1), 
                                     q_emb_2_alpha.unsqueeze(dim=1)], dim=1)
            q_emb_beta = torch.cat([q_emb_1_beta.unsqueeze(dim=1), 
                                    q_emb_2_beta.unsqueeze(dim=1)], dim=1)
            q_emb_alpha, q_emb_beta = self.intersection(q_emb_alpha, q_emb_beta)
            qu_emb_alpha = torch.cat([q_emb_1_alpha.unsqueeze(dim=1), 
                                     q_emb_2_alpha.unsqueeze(dim=1),
                                     uq_emb_alpha.unsqueeze(dim=1)], dim=1)
            qu_emb_beta = torch.cat([q_emb_1_beta.unsqueeze(dim=1), 
                                    q_emb_2_beta.unsqueeze(dim=1),
                                    uq_emb_beta.unsqueeze(dim=1)], dim=1)
            qu_emb_alpha, qu_emb_beta = self.intersection(qu_emb_alpha, qu_emb_beta)

            if 'mmoe' in self.which:
                q_emb_alpha, uq_emb_alpha, qu_emb_alpha = self._mmoe(q_emb_alpha, uq_emb_alpha, qu_emb_alpha, part=0)
                q_emb_beta, uq_emb_beta, qu_emb_beta = self._mmoe(q_emb_beta, uq_emb_beta, qu_emb_beta, part=1)

            q_dist = torch.distributions.beta.Beta(self.regularizer(q_emb_alpha), self.regularizer(q_emb_beta))
            uq_dist = torch.distributions.beta.Beta(self.regularizer(uq_emb_alpha), self.regularizer(uq_emb_beta))
            qu_dist = torch.distributions.beta.Beta(self.regularizer(qu_emb_alpha), self.regularizer(qu_emb_beta))
            logits_q = self._cal_logit_beta(a_emb, q_dist).unsqueeze(dim=1)
            logits_uq = self._cal_logit_beta(a_emb, uq_dist).unsqueeze(dim=1)
            logits_qu = self._cal_logit_beta(a_emb, qu_dist).unsqueeze(dim=1)

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

        if self.which == 'vec':
            q_emb_1 = self.projection(e_emb_1, r_emb_1)
            q_emb_2 = self.projection(e_emb_2, r_emb_2)
            q_emb_3 = self.projection(e_emb_3, r_emb_3)
            q_emb_4 = self.projection(u_emb, ur_emb)
            q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                                 q_emb_2.unsqueeze(dim=1), 
                                                 q_emb_3.unsqueeze(dim=1),
                                                 q_emb_4.unsqueeze(dim=1)], dim=1))
            logits = self._cal_logit_vec(a_emb, q_emb).unsqueeze(dim=1)
            return torch.cat([logits, logits, logits], dim=1)

        elif 'box' in self.which:
            r_emb_1_offset = self.offset_embedding(data[:, 1])
            r_emb_2_offset = self.offset_embedding(data[:, 3])
            r_emb_3_offset = self.offset_embedding(data[:, 5])
            ur_emb_offset = self.offset_embedding.weight[-1].unsqueeze(dim=0)
            
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
            
            q_emb_1_offset = self.projection(torch.zeros_like(r_emb_1), r_emb_1_offset)
            q_emb_2_offset = self.projection(torch.zeros_like(r_emb_2), r_emb_2_offset)
            q_emb_3_offset = self.projection(torch.zeros_like(r_emb_3), r_emb_3_offset)
            q_emb_offset = self.intersection(torch.cat([q_emb_1_offset.unsqueeze(dim=1), 
                                                        q_emb_2_offset.unsqueeze(dim=1),
                                                        q_emb_3_offset.unsqueeze(dim=1)], dim=1))
            uq_emb_offset = self.projection(torch.zeros_like(r_emb_1), ur_emb_offset)
            qu_emb_offset = self.intersection(torch.cat([q_emb_1_offset.unsqueeze(dim=1), 
                                                         q_emb_2_offset.unsqueeze(dim=1),
                                                         q_emb_3_offset.unsqueeze(dim=1),
                                                         uq_emb_offset.unsqueeze(dim=1)], dim=1))
            
            if 'mmoe' in self.which:
                q_emb, uq_emb, qu_emb = self._mmoe(q_emb, uq_emb, qu_emb, part=0)
                q_emb_offset, uq_emb_offset, qu_emb_offset = self._mmoe(q_emb_offset, uq_emb_offset, qu_emb_offset, part=1)

            logits_q = self._cal_logit_box(a_emb, q_emb, q_emb_offset).unsqueeze(dim=1)
            logits_uq = self._cal_logit_box(a_emb, uq_emb, uq_emb_offset).unsqueeze(dim=1)
            logits_qu = self._cal_logit_box(a_emb, qu_emb, qu_emb_offset).unsqueeze(dim=1)
            
            return torch.cat([logits_q, logits_uq, logits_qu], dim=1)

        elif 'beta' in self.which:
            q_emb_1 = self.projection(self.regularizer(e_emb_1), r_emb_1)
            q_emb_2 = self.projection(self.regularizer(e_emb_2), r_emb_2)
            q_emb_3 = self.projection(self.regularizer(e_emb_3), r_emb_3)
            uq_emb = self.projection(self.regularizer(u_emb), ur_emb.expand_as(u_emb))

            q_emb_1_alpha, q_emb_1_beta = torch.chunk(q_emb_1, 2, dim=-1)
            q_emb_2_alpha, q_emb_2_beta = torch.chunk(q_emb_2, 2, dim=-1)
            q_emb_3_alpha, q_emb_3_beta = torch.chunk(q_emb_3, 2, dim=-1)
            uq_emb_alpha, uq_emb_beta = torch.chunk(uq_emb, 2, dim=-1)

            q_emb_alpha = torch.cat([q_emb_1_alpha.unsqueeze(dim=1), 
                                     q_emb_2_alpha.unsqueeze(dim=1),
                                     q_emb_3_alpha.unsqueeze(dim=1)], dim=1)
            q_emb_beta = torch.cat([q_emb_1_beta.unsqueeze(dim=1), 
                                    q_emb_2_beta.unsqueeze(dim=1),
                                    q_emb_3_beta.unsqueeze(dim=1)], dim=1)
            q_emb_alpha, q_emb_beta = self.intersection(q_emb_alpha, q_emb_beta)
            qu_emb_alpha = torch.cat([q_emb_1_alpha.unsqueeze(dim=1), 
                                     q_emb_2_alpha.unsqueeze(dim=1),
                                     q_emb_3_alpha.unsqueeze(dim=1),
                                     uq_emb_alpha.unsqueeze(dim=1)], dim=1)
            qu_emb_beta = torch.cat([q_emb_1_beta.unsqueeze(dim=1), 
                                    q_emb_2_beta.unsqueeze(dim=1),
                                    q_emb_3_beta.unsqueeze(dim=1),
                                    uq_emb_beta.unsqueeze(dim=1)], dim=1)
            qu_emb_alpha, qu_emb_beta = self.intersection(qu_emb_alpha, qu_emb_beta)

            if 'mmoe' in self.which:
                q_emb_alpha, uq_emb_alpha, qu_emb_alpha = self._mmoe(q_emb_alpha, uq_emb_alpha, qu_emb_alpha, part=0)
                q_emb_beta, uq_emb_beta, qu_emb_beta = self._mmoe(q_emb_beta, uq_emb_beta, qu_emb_beta, part=1)

            q_dist = torch.distributions.beta.Beta(self.regularizer(q_emb_alpha), self.regularizer(q_emb_beta))
            uq_dist = torch.distributions.beta.Beta(self.regularizer(uq_emb_alpha), self.regularizer(uq_emb_beta))
            qu_dist = torch.distributions.beta.Beta(self.regularizer(qu_emb_alpha), self.regularizer(qu_emb_beta))
            logits_q = self._cal_logit_beta(a_emb, q_dist).unsqueeze(dim=1)
            logits_uq = self._cal_logit_beta(a_emb, uq_dist).unsqueeze(dim=1)
            logits_qu = self._cal_logit_beta(a_emb, qu_dist).unsqueeze(dim=1)

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

        if self.which.split('-')[-1] == 'only':
            return self.criterion(preds[:, -1], labels[:, -1])
        elif self.which.split('-')[-1] == 'all':
            return self.criterion(preds, labels)
        elif self.which.split('-')[-1] == 'lqa':
            return (self.criterion(preds[:, -1], labels[:, -1]) + self.criterion(preds[:, 0], labels[:, 0]) ) / 2
        elif self.which.split('-')[-1] == 'rec':
            return (self.criterion(preds[:, -1], labels[:, -1]) + self.criterion(preds[:, 1], labels[:, 1]) ) / 2
        else:
            raise ValueError

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
    
    print(f'MR: {r}, MRR: {rr}, Hit@10: {h10}, Hit@20: {h20}, Hit@50: {h50}, nDCG@10: {ndcg10}, nDCG@20: {ndcg20}, nDCG@50: {ndcg50}', flush=True)
    
    return r, rr, h10, h20, h50, ndcg10, ndcg20, ndcg50

def iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--cached', default=0, type=int)
    parser.add_argument('--dataset', default='amazon-book', type=str)
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
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    input_path = cfg.data_root + cfg.dataset
    save_root = f'../tmp/{cfg.which}_{cfg.dataset}_{cfg.emb_dim}_{cfg.lr}_{cfg.bs}_{cfg.wd}/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    N_rel, N_item, N_ent, N_user, filters = read_data(input_path)
    test_1p = load_obj(input_path + '/input/1p_test.pkl')
    test_2p = load_obj(input_path + '/input/2p_test.pkl')
    test_3p = load_obj(input_path + '/input/3p_test.pkl')
    test_2i = load_obj(input_path + '/input/2i_test.pkl')
    test_3i = load_obj(input_path + '/input/3i_test.pkl')
    
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
    valid_dataloaders = [lqa_dataloader_1p_valid, 
                        lqa_dataloader_2p_valid,
                        lqa_dataloader_3p_valid,
                        lqa_dataloader_2i_valid,
                        lqa_dataloader_3i_valid
                        ]
    test_dataloaders = [lqa_dataloader_1p_test, 
                        lqa_dataloader_2p_test,
                        lqa_dataloader_3p_test,
                        lqa_dataloader_2i_test,
                        lqa_dataloader_3i_test
                        ]
    query_types = ['1p', '2p', '3p', '2i', '3i']
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
            mmrr = []
            for i in range(len(query_types)):
                flag = query_types[i]
                print(f'{flag}:')
                r, rr, h10, h20, h50, ndcg10, ndcg20, ndcg50 = evaluate(valid_dataloaders[i], model, device, filters, flag)
                mmrr.append(rr)
            value = sum(mmrr) / len(mmrr)
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
            for i in range(len(query_types)):
                flag = query_types[i]
                print(f'{flag}:', flush=True)
                r, rr, h10, h20, h50, ndcg10, ndcg20, ndcg50 = evaluate(test_dataloaders[i], model, device, filters, flag)
            break
