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

class LQADatasetTrain(torch.utils.data.Dataset):
    def __init__(self, N_item, N_ent, data, cfg):
        super().__init__()
        self.N_item = N_item
        self.N_ent = N_ent
        self.num_ng = cfg.num_ng
        self.data_dict = data
        self.item_candidates = set(range(self.N_item))
        self.all_candidates = set(range(self.N_ent))
        self.data = self._get_data(data)

    def __len__(self):
        return len(self.data)
    
    def _get_data(self, data):
        ret = []
        for query in data:
            query_as_list = list(query)
            items = data[query]['both']
            for item in items:
                query_as_list.extend([item])
                ret.append(query_as_list)
                query_as_list = list(query)
        return torch.tensor(ret)

    def __getitem__(self, idx):
        pos = self.data[idx]
        query = tuple(pos[:-1].numpy().tolist())
        answers = self.data_dict[query]
        lqa_answers = (answers['lqa'] & self.item_candidates)
        rec_answers = answers['rec']
        # query_expanded = pos[:-1].expand(self.num_ng, -1)

        lqa_not_rec_cadidates = lqa_answers - rec_answers
        rec_not_lqa_cadidates = rec_answers - lqa_answers
        
        if len(lqa_not_rec_cadidates):
            lqa_not_rec_answer = torch.tensor(random.sample(lqa_not_rec_cadidates, k=1))
            lqa_sample = torch.cat([pos[:-1], lqa_not_rec_answer, torch.tensor([1, 0, 0])], dim=0).unsqueeze(dim=0)
        else:
            lqa_sample = torch.cat([pos[:-1], torch.full((4, ), 2).long()], dim=0).unsqueeze(dim=0)
        
        if len(rec_not_lqa_cadidates):
            rec_not_lqa_answer = torch.tensor(random.sample(rec_not_lqa_cadidates, k=1))
            rec_sample = torch.cat([pos[:-1], rec_not_lqa_answer, torch.tensor([0, 1, 0])], dim=0).unsqueeze(dim=0)
        else:
            rec_sample = torch.cat([pos[:-1], torch.full((4, ), 2).long()], dim=0).unsqueeze(dim=0)

        # neg_candidates = self.all_candidates - lqa_answers - rec_answers
        neg_candidates = self.all_candidates
        neg_answer = torch.tensor(random.sample(neg_candidates, k=1))
        neg_sample = torch.cat([pos[:-1], neg_answer, torch.tensor([0, 0, 0])], dim=0).unsqueeze(dim=0)

        # neg_candidates = list(self.all_candidates - lqa_answers - rec_answers)
        # neg_answers = torch.tensor(np.random.choice(neg_candidates, self.num_ng)).unsqueeze(dim=1)
        # # neg_answers = torch.tensor(np.random.choice(self.N_ent, self.num_ng)).unsqueeze(dim=1)
        # neg_samples = torch.cat([query_expanded, neg_answers, torch.zeros(self.num_ng, 3).long()], dim=1)

        pos = torch.cat([pos, torch.ones(3).long()], dim=0).unsqueeze(dim=0)
        sample = torch.cat([pos, lqa_sample, rec_sample, neg_sample], dim=0)

        return sample


class LQADatasetTest(torch.utils.data.Dataset):
    def __init__(self, N_item, data, cfg, stage):
        super().__init__()
        self.N_item = N_item
        self.num_ng = cfg.num_ng
        self.stage = stage
        self.data = self._get_data(data)

    def _get_data(self, data):
        ret = []
        for query in data:
            query_as_list = list(query)
            ret.append(query_as_list)
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
        queries = pos[:-1].unsqueeze(dim=0).expand(self.N_item, len(pos) - 1)
        items = torch.arange(self.N_item).unsqueeze(dim=1)
        return torch.cat([queries, items], dim=-1), pos

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
        self.layer2 = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, emb):
        return self.layer2(torch.nn.functional.relu(self.layer1(emb)))

class Gate(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, self.dim)
        self.layer2 = torch.nn.Linear(self.dim, 2)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, emb):
        return torch.softmax(self.layer2(torch.nn.functional.relu(self.layer1(emb))), dim=-1)

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
        self.num_ng = cfg.num_ng
        self.gamma = cfg.gamma
        self.e_embedding = torch.nn.Embedding(N_ent, cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(N_rel, cfg.emb_dim)
        self.u_embedding = torch.nn.Embedding(N_user, cfg.emb_dim)
        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.u_embedding.weight.data)
        self.intersection = IntersectionNet(cfg.emb_dim)
        self.expert_1 = Expert(cfg.emb_dim)
        self.expert_2 = Expert(cfg.emb_dim)
        self.gate_lqa = Gate(cfg.emb_dim)
        self.gate_rec = Gate(cfg.emb_dim)
        self.gate_both = Gate(cfg.emb_dim)
        self.tower_lqa = Tower(cfg.emb_dim)
        self.tower_rec = Tower(cfg.emb_dim)
        self.tower_both = Tower(cfg.emb_dim)

    def projection(self, emb_1, emb_2):
        return emb_1 + emb_2

    def forward_multitask(self, q_emb, u_emb, qu_emb, a_emb):
        # emb_1 = self.expert_1(qu_emb)
        # emb_2 = self.expert_2(qu_emb)
        # gate_lqa = self.gate_lqa(qu_emb)
        # gate_rec = self.gate_rec(qu_emb)
        # gate_both = self.gate_both(qu_emb)
        # emb_lqa = emb_1 * gate_lqa[:, 0].unsqueeze(dim=-1) + emb_2 * gate_lqa[:, 1].unsqueeze(dim=-1)
        # emb_rec = emb_1 * gate_rec[:, 0].unsqueeze(dim=-1) + emb_2 * gate_rec[:, 1].unsqueeze(dim=-1)
        # emb_both = emb_1 * gate_both[:, 0].unsqueeze(dim=-1) + emb_2 * gate_both[:, 1].unsqueeze(dim=-1)
        # preds_lqa = - torch.norm(a_emb - q_emb.unsqueeze(dim=1), p=1, dim=-1)
        # preds_rec = - torch.norm(a_emb - u_emb.unsqueeze(dim=1), p=1, dim=-1)
        # preds_both = - torch.norm(a_emb - qu_emb.unsqueeze(dim=1), p=1, dim=-1)
        # preds_lqa = self.tower_lqa(torch.cat([emb_lqa.unsqueeze(dim=1).expand_as(a_emb), a_emb], dim=-1)).squeeze(dim=-1)
        # preds_rec = self.tower_rec(torch.cat([emb_rec.unsqueeze(dim=1).expand_as(a_emb), a_emb], dim=-1)).squeeze(dim=-1)
        # preds_both = self.tower_both(torch.cat([emb_both.unsqueeze(dim=1).expand_as(a_emb), a_emb], dim=-1)).squeeze(dim=-1)

        # preds_lqa = self.tower_lqa(torch.cat([q_emb.unsqueeze(dim=1).expand_as(a_emb), a_emb], dim=-1)).squeeze(dim=-1)
        # preds_rec = self.tower_rec(torch.cat([u_emb.unsqueeze(dim=1).expand_as(a_emb), a_emb], dim=-1)).squeeze(dim=-1)
        # preds_both = self.tower_both(torch.cat([qu_emb.unsqueeze(dim=1).expand_as(a_emb), a_emb], dim=-1)).squeeze(dim=-1)
        preds_lqa = self.tower_lqa(torch.cat([qu_emb.unsqueeze(dim=1).expand_as(a_emb), a_emb], dim=-1)).squeeze(dim=-1)
        preds_rec = self.tower_rec(torch.cat([qu_emb.unsqueeze(dim=1).expand_as(a_emb), a_emb], dim=-1)).squeeze(dim=-1)
        preds_both = self.tower_both(torch.cat([qu_emb.unsqueeze(dim=1).expand_as(a_emb), a_emb], dim=-1)).squeeze(dim=-1)
        return preds_lqa, preds_rec, preds_both

    def forward_1p(self, data):
        e_emb = self.e_embedding(data[:, 0, 0])
        r_emb = self.r_embedding(data[:, 0, 1])
        u_emb = self.u_embedding(data[:, 0, 2])
        a_emb = self.e_embedding(data[:, :, 3])

        q_emb = self.projection(e_emb, r_emb)
        qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                              u_emb.unsqueeze(dim=1)], dim=1))

        return self.forward_multitask(q_emb, u_emb, qu_emb, a_emb)
        # return self.forward_multitask(q_emb, u_emb, a_emb)

    def forward_2p(self, data):
        e_emb = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        r_emb_2 = self.r_embedding(data[:, 0, 2])
        u_emb = self.u_embedding(data[:, 0, 3])
        a_emb = self.e_embedding(data[:, :, 4])

        q_emb = self.projection(e_emb, r_emb_1)
        q_emb = self.projection(q_emb, r_emb_2)
        qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                              u_emb.unsqueeze(dim=1)], dim=1))

        return self.forward_multitask(q_emb, u_emb, qu_emb, a_emb)
        # return self.forward_multitask(q_emb, u_emb, a_emb)

    def forward_3p(self, data):
        e_emb = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        r_emb_2 = self.r_embedding(data[:, 0, 2])
        r_emb_3 = self.r_embedding(data[:, 0, 3])
        u_emb = self.u_embedding(data[:, 0, 4])
        a_emb = self.e_embedding(data[:, :, 5])

        q_emb = self.projection(e_emb, r_emb_1)
        q_emb = self.projection(q_emb, r_emb_2)
        q_emb = self.projection(q_emb, r_emb_3)
        qu_emb = self.intersection(torch.cat([q_emb.unsqueeze(dim=1), 
                                              u_emb.unsqueeze(dim=1)], dim=1))
        
        return self.forward_multitask(q_emb, u_emb, qu_emb, a_emb)
        # return self.forward_multitask(q_emb, u_emb, a_emb)

    def forward_2i(self, data):
        e_emb_1 = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        e_emb_2 = self.e_embedding(data[:, 0, 2])
        r_emb_2 = self.r_embedding(data[:, 0, 3])
        u_emb = self.u_embedding(data[:, 0, 4])
        a_emb = self.e_embedding(data[:, :, 5])

        q_emb_1 = self.projection(e_emb_1, r_emb_1)
        q_emb_2 = self.projection(e_emb_2, r_emb_2)
        q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                             q_emb_2.unsqueeze(dim=1)], dim=1))
        qu_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                              q_emb_2.unsqueeze(dim=1),
                                              u_emb.unsqueeze(dim=1)], dim=1))
        
        return self.forward_multitask(q_emb, u_emb, qu_emb, a_emb)
        # return self.forward_multitask(q_emb, u_emb, a_emb)

    def forward_3i(self, data):
        e_emb_1 = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        e_emb_2 = self.e_embedding(data[:, 0, 2])
        r_emb_2 = self.r_embedding(data[:, 0, 3])
        e_emb_3 = self.e_embedding(data[:, 0, 4])
        r_emb_3 = self.r_embedding(data[:, 0, 5])
        u_emb = self.u_embedding(data[:, 0, 6])
        a_emb = self.e_embedding(data[:, :, 7])

        q_emb_1 = self.projection(e_emb_1, r_emb_1)
        q_emb_2 = self.projection(e_emb_2, r_emb_2)
        q_emb_3 = self.projection(e_emb_3, r_emb_3)
        q_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                             q_emb_2.unsqueeze(dim=1),
                                             q_emb_3.unsqueeze(dim=1)], dim=1))
        qu_emb = self.intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                              q_emb_2.unsqueeze(dim=1),
                                              q_emb_3.unsqueeze(dim=1),
                                              u_emb.unsqueeze(dim=1)], dim=1))

        return self.forward_multitask(q_emb, u_emb, qu_emb, a_emb)
        # return self.forward_multitask(q_emb, u_emb, a_emb)


    def get_loss(self, data, flag):
        if flag == '1p':
            preds_lqa, preds_rec, preds_both = self.forward_1p(data)
        elif flag == '2p':
            preds_lqa, preds_rec, preds_both = self.forward_2p(data)
        elif flag == '3p':
            preds_lqa, preds_rec, preds_both = self.forward_3p(data)
        elif flag == '2i':
            preds_lqa, preds_rec, preds_both = self.forward_2i(data)
        elif flag == '3i':
            preds_lqa, preds_rec, preds_both = self.forward_3i(data)
        else:
            raise ValueError

        labels_lqa = data[:, :, -3]
        labels_rec = data[:, :, -2]

        valid_lqa = (labels_lqa[:, 1] != 2).nonzero().squeeze(dim=-1)
        valid_rec = (labels_rec[:, 2] != 2).nonzero().squeeze(dim=-1)
        valid_lqa_and_rec = ((labels_lqa[:, 1] != 2) * (labels_rec[:, 2] != 2)).nonzero().squeeze(dim=-1)

        # 0 - 3
        loss_both_03 = - torch.nn.functional.logsigmoid(preds_both[:, 0] - preds_both[:, 3]).mean()
        return loss_both_03
        
        # # 0 - 1
        # loss_both_01 = - torch.nn.functional.logsigmoid(torch.index_select(preds_both, 0, valid_lqa)[:, 0] - torch.index_select(preds_both, 0, valid_lqa)[:, 1]).mean()
        # # 0 - 2
        # loss_both_02 = - torch.nn.functional.logsigmoid(torch.index_select(preds_both, 0, valid_rec)[:, 0] - torch.index_select(preds_both, 0, valid_rec)[:, 2]).mean()

        # # 1 - 2
        # loss_lqa_12 = - torch.nn.functional.logsigmoid(torch.index_select(preds_lqa, 0, valid_lqa_and_rec)[:, 1] - torch.index_select(preds_lqa, 0, valid_lqa_and_rec)[:, 2]).mean()
        # # # 1 - 3
        # # loss_lqa_13 = - torch.nn.functional.logsigmoid(torch.index_select(preds_lqa, 0, valid_lqa)[:, 1] - torch.index_select(preds_lqa, 0, valid_lqa)[:, 3]).mean()

        # # 2 - 1
        # loss_rec_21 = - torch.nn.functional.logsigmoid(torch.index_select(preds_rec, 0, valid_lqa_and_rec)[:, 2] - torch.index_select(preds_rec, 0, valid_lqa_and_rec)[:, 1]).mean()
        # # # 2 - 3
        # # loss_rec_23 = - torch.nn.functional.logsigmoid(torch.index_select(preds_rec, 0, valid_rec)[:, 2] - torch.index_select(preds_rec, 0, valid_rec)[:, 3]).mean()

        # loss_both = (loss_both_03 + loss_both_01 + loss_both_02) / 3
        # # loss_both = loss_both_03
        # loss_lqa = loss_lqa_12
        # loss_rec = loss_rec_21
        # # loss_lqa = (loss_lqa_12 + loss_lqa_13) / 2
        # # loss_rec = (loss_rec_21 + loss_rec_23) / 2

        # return (loss_lqa + loss_rec + self.gamma * loss_both) / (2 + self.gamma)

def get_rank(pos, logits, flt):
    ranking = torch.argsort(logits, descending=True)
    rank = (ranking == pos[0, -1]).nonzero().item() + 1
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
    ndcg10 = []
    ndcg20 = []
    true_10 = torch.zeros(10)
    true_10[0] = 1
    true_20 = torch.zeros(20)
    true_20[0] = 1
    model.eval()
    with torch.no_grad():
        for possible, pos in dataloader:
            possible = possible.to(device)
            if flag == '1p':
                logits = model.forward_1p(possible)[-1][0]
            elif flag == '2p':
                logits = model.forward_2p(possible)[-1][0]
            elif flag == '3p':
                logits = model.forward_3p(possible)[-1][0]
            elif flag == '2i':
                logits = model.forward_2i(possible)[-1][0]
            elif flag == '3i':
                logits = model.forward_3i(possible)[-1][0]
            else:
                raise ValueError
            flt = train_dict[pos[0][-2].item()]
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

            ndcg10.append(ndcg(true_10, rank, 10))
            ndcg20.append(ndcg(true_20, rank, 20))
    
    results = [r, rr, h10, h20, ndcg10, ndcg20]
    
    r = int(sum(results[0])/len(results[0]))
    rr = round(sum(results[1])/len(results[1]), 3)
    h10 = round(sum(results[2])/len(results[2]), 3)
    h20 = round(sum(results[3])/len(results[3]), 3)
    ndcg10 = round(sum(results[4])/len(results[4]), 3)
    ndcg20 = round(sum(results[5])/len(results[5]), 3)
    
    print(f'MR: {r}, MRR: {rr}, Hit@10: {h10}, Hit@20: {h20}, nDCG@10: {ndcg10}, nDCG@20: {ndcg20}', flush=True)
    
    return r, rr, h10, h20, ndcg10, ndcg20

def iterator(dataloader):
    while True:
        for data in dataloader:
            yield data

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--dataset', default='amazon-book', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_ng', default=8, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--alpha', default=1, type=int)
    parser.add_argument('--gamma', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--add_loss', default=1, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--tolerance', default=5, type=int)
    parser.add_argument('--valid_interval', default=1000, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    input_path = cfg.data_root + cfg.dataset
    save_root = f'../tmp/logicrec_{cfg.dataset}_{cfg.emb_dim}_{cfg.num_ng}_{cfg.lr}_{cfg.bs}/'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    N_rel, N_item, N_ent, N_user, filters = read_data(input_path)
    
    train_1p, test_1p = load_obj(input_path + '/input/1p_train.pkl'), load_obj(input_path + '/input/1p_test.pkl')
    print('Read 1p done')
    train_2p, test_2p = load_obj(input_path + '/input/2p_train.pkl'), load_obj(input_path + '/input/2p_test.pkl')
    print('Read 2p done')
    train_3p, test_3p = load_obj(input_path + '/input/3p_train.pkl'), load_obj(input_path + '/input/3p_test.pkl')
    print('Read 3p done')
    train_2i, test_2i = load_obj(input_path + '/input/2i_train.pkl'), load_obj(input_path + '/input/2i_test.pkl')
    print('Read 2i done')
    train_3i, test_3i = load_obj(input_path + '/input/3i_train.pkl'), load_obj(input_path + '/input/3i_test.pkl')
    print('Read 3i done')

    lqa_dataset_1p_train = LQADatasetTrain(N_item, N_ent, train_1p, cfg)
    lqa_dataset_1p_valid = LQADatasetTest(N_item, test_1p, cfg, stage='valid')
    lqa_dataset_1p_test = LQADatasetTest(N_item, test_1p, cfg, stage='test')
    lqa_dataset_2p_train = LQADatasetTrain(N_item, N_ent, train_2p, cfg)
    lqa_dataset_2p_valid = LQADatasetTest(N_item, test_2p, cfg, stage='valid')
    lqa_dataset_2p_test = LQADatasetTest(N_item, test_2p, cfg, stage='test')
    lqa_dataset_3p_train = LQADatasetTrain(N_item, N_ent, train_3p, cfg)
    lqa_dataset_3p_valid = LQADatasetTest(N_item, test_3p, cfg, stage='valid')
    lqa_dataset_3p_test = LQADatasetTest(N_item, test_3p, cfg, stage='test')
    lqa_dataset_2i_train = LQADatasetTrain(N_item, N_ent, train_2i, cfg)
    lqa_dataset_2i_valid = LQADatasetTest(N_item, test_2i, cfg, stage='valid')
    lqa_dataset_2i_test = LQADatasetTest(N_item, test_2i, cfg, stage='test')
    lqa_dataset_3i_train = LQADatasetTrain(N_item, N_ent, train_3i, cfg)
    lqa_dataset_3i_valid = LQADatasetTest(N_item, test_3i, cfg, stage='valid')
    lqa_dataset_3i_test = LQADatasetTest(N_item, test_3i, cfg, stage='test')
    
    lqa_dataloader_1p_train = torch.utils.data.DataLoader(dataset=lqa_dataset_1p_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
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
    lqa_dataloader_2p_train = torch.utils.data.DataLoader(dataset=lqa_dataset_2p_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
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
    lqa_dataloader_3p_train = torch.utils.data.DataLoader(dataset=lqa_dataset_3p_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
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
    lqa_dataloader_2i_train = torch.utils.data.DataLoader(dataset=lqa_dataset_2i_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
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
    lqa_dataloader_3i_train = torch.utils.data.DataLoader(dataset=lqa_dataset_3i_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
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
    lqa_dataloader_1p_train = iterator(lqa_dataloader_1p_train)
    lqa_dataloader_2p_train = iterator(lqa_dataloader_2p_train)
    lqa_dataloader_3p_train = iterator(lqa_dataloader_3p_train)
    lqa_dataloader_2i_train = iterator(lqa_dataloader_2i_train)
    lqa_dataloader_3i_train = iterator(lqa_dataloader_3i_train)
    
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

        loss_1p = model.get_loss(next(lqa_dataloader_1p_train).to(device), flag='1p')
        loss_2p = model.get_loss(next(lqa_dataloader_2p_train).to(device), flag='2p')
        loss_3p = model.get_loss(next(lqa_dataloader_3p_train).to(device), flag='3p')
        loss_2i = model.get_loss(next(lqa_dataloader_2i_train).to(device), flag='2i')
        loss_3i = model.get_loss(next(lqa_dataloader_3i_train).to(device), flag='3i')

        loss = (cfg.alpha * loss_1p + loss_2p + loss_3p + loss_2i + loss_3i) / (4 + cfg.alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        avg_loss.append(loss.item())
        
        if (step + 1) % cfg.valid_interval == 0:
            print(f'Loss: {round(sum(avg_loss) / len(avg_loss), 4)}')
            avg_loss = []
            print('Validating...')
            mmrr = []
            for i in range(len(query_types)):
                flag = query_types[i]
                print(f'{flag}:')
                r, rr, h10, h20, ndcg10, ndcg20 = evaluate(valid_dataloaders[i], model, device, filters, flag)
                mmrr.append(rr)
            value = sum(mmrr) / len(mmrr)
            if value >= max_value:
                max_value = value
                tolerance = cfg.tolerance
            else:
                tolerance -= 1
            print(f'Tolerance: {tolerance}')
            torch.save(model.state_dict(), save_root + (str(step + 1)))

        if (tolerance == 0) or ((step + 1) == cfg.max_steps):
            print('Testing...')
            print(f'Best performance at epoch {step - cfg.tolerance * cfg.valid_interval + 1}')
            model.load_state_dict(torch.load(save_root + str(step - cfg.tolerance * cfg.valid_interval + 1)))
            for i in range(len(query_types)):
                flag = query_types[i]
                print(f'{flag}:')
                r, rr, h10, h20, ndcg10, ndcg20 = evaluate(test_dataloaders[i], model, device, filters, flag)
            break
