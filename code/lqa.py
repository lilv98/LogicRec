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
    
    N_rel = 0
    with open(path + '/relation_list.txt') as f:
        for line in f:
            N_rel += 1
    print(f'N_rel: {N_rel}')
    print(f'N_item: {N_item}')
    print(f'N_ent: {N_ent}')
    
    train_dict = {}
    with open(path + '/input/baseline_train.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            train_dict[int(line[0])] = [int(x) for x in line[1:]]
    
    test_dict = {}
    with open(path + '/input/baseline_test.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            test_dict[int(line[0])] = [int(x) for x in line[1:]]
            
    assert len(set(test_dict.keys()) | set(train_dict.keys())) == len(train_dict)
    N_user = len(train_dict)
    return N_rel, N_item, N_ent, N_user, train_dict

class LQADatasetTrain(torch.utils.data.Dataset):
    def __init__(self, N_item, data, cfg):
        super().__init__()
        self.N_item = N_item
        self.num_ng = cfg.num_ng
        self.data = self._get_data(data)

    def _get_data(self, data):
        ret = []
        for query in data:
            query_as_list = list(query)
            items = data[query]
            for item in items:
                query_as_list.extend([item])
                ret.append(query_as_list)
                query_as_list = list(query)
        return torch.tensor(ret)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pos = self.data[idx]
        neg_answers = torch.tensor(np.random.choice(self.N_item, self.num_ng)).unsqueeze(dim=1)
        query = pos[:-1].expand(self.num_ng, -1)
        negs = torch.cat([query, neg_answers], dim=1)
        sample = torch.cat([pos.unsqueeze(dim=0), negs], dim=0)
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

class CenterIntersection(torch.nn.Module):

    def __init__(self, dim):
        super(CenterIntersection, self).__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, self.dim)
        self.layer2 = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        layer1_act = torch.nn.functional.relu(self.layer1(embeddings)) 
        # (batchsize, num_conj, dim)
        attention = torch.nn.functional.softmax(self.layer2(layer1_act), dim=1) 
        # (batchsize, num_conj, dim)
        embedding = torch.sum(attention * embeddings, dim=1)
        return embedding

class BoxOffsetIntersection(torch.nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, self.dim)
        self.layer2 = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, embeddings):
        pdb.set_trace()
        layer1_act = torch.nn.functional.relu(self.layer1(embeddings))
        layer1_mean = torch.mean(layer1_act, dim=0) 
        gate = torch.sigmoid(self.layer2(layer1_mean))
        offset, _ = torch.min(embeddings, dim=0)

        return offset * gate

class LogicRecModel(torch.nn.Module):
    def __init__(self, N_user, N_ent, N_rel, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.gamma = cfg.gamma
        self.base_model = cfg.base_model
        self.e_embedding = torch.nn.Embedding(N_ent, cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(N_rel, cfg.emb_dim)
        self.u_embedding = torch.nn.Embedding(N_user, cfg.emb_dim)
        self.center_net = CenterIntersection(cfg.emb_dim)
        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.u_embedding.weight.data)
        if self.base_model == 'box':
            self.offset_embedding = torch.nn.Embedding(N_rel, cfg.emb_dim)
            self.offset_net = BoxOffsetIntersection(N_ent)

    def _projection(self, e_emb, r_emb, offset_emb=None):
        if self.base_model == 'vec':
            return e_emb + r_emb
        elif self.base_model == 'box':
            pdb.set_trace()
        else:
            raise ValueError

    def _intersection(self, embs):
        # (batchsize, n_conj, emb_dim)
        if self.base_model == 'vec':
            return self.center_net(embs)
        else:
            raise ValueError
        
    def _cal_logit(self, q_emb, a_emb):
        if self.base_model == 'vec':
            # logit = (q_emb * a_emb).sum(dim=-1)
            distance = q_emb - a_emb
            logit = self.gamma - torch.norm(distance, p=1, dim=-1)
            return logit
        else:
            raise ValueError
    
    def forward_1p(self, data):
        e_emb = self.e_embedding(data[:, 0, 0])
        r_emb = self.r_embedding(data[:, 0, 1])
        u_emb = self.u_embedding(data[:, 0, -2])
        a_emb = self.e_embedding(data[:, :, -1])

        ur_emb = self.r_embedding.weight[-1]
        q_emb_1 = self._projection(e_emb, r_emb, torch.zeros_like(e_emb))
        q_emb_2 = self._projection(u_emb, ur_emb, torch.zeros_like(e_emb))
        q_emb = self._intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                              q_emb_2.unsqueeze(dim=1)], dim=1))

        return self._cal_logit(q_emb.unsqueeze(dim=1), a_emb)

    def forward_2p(self, data):
        e_emb = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        r_emb_2 = self.r_embedding(data[:, 0, 2])
        u_emb = self.u_embedding(data[:, 0, -2])
        a_emb = self.e_embedding(data[:, :, -1])

        ur_emb = self.r_embedding.weight[-1]
        q_emb_1 = self._projection(self._projection(e_emb, r_emb_1), r_emb_2)
        q_emb_2 = self._projection(u_emb, ur_emb)
        q_emb = self._intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                              q_emb_2.unsqueeze(dim=1)], dim=1))

        return self._cal_logit(q_emb.unsqueeze(dim=1), a_emb)

    def forward_3p(self, data):
        e_emb = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        r_emb_2 = self.r_embedding(data[:, 0, 2])
        r_emb_3 = self.r_embedding(data[:, 0, 3])
        u_emb = self.u_embedding(data[:, 0, -2])
        a_emb = self.e_embedding(data[:, :, -1])
        
        ur_emb = self.r_embedding.weight[-1]
        q_emb_1 = self._projection(self._projection(self._projection(e_emb, r_emb_1), r_emb_2), r_emb_3)
        q_emb_2 = self._projection(u_emb, ur_emb)
        q_emb = self._intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                              q_emb_2.unsqueeze(dim=1)], dim=1))

        return self._cal_logit(q_emb.unsqueeze(dim=1), a_emb)

    def forward_2i(self, data):
        e_emb_1 = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        e_emb_2 = self.e_embedding(data[:, 0, 2])
        r_emb_2 = self.r_embedding(data[:, 0, 3])
        u_emb = self.u_embedding(data[:, 0, -2])
        a_emb = self.e_embedding(data[:, :, -1])

        ur_emb = self.r_embedding.weight[-1]
        q_emb_1 = self._projection(e_emb_1, r_emb_1)
        q_emb_2 = self._projection(e_emb_2, r_emb_2)
        q_emb_3 = self._projection(u_emb, ur_emb)
        q_emb = self._intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                              q_emb_2.unsqueeze(dim=1), 
                                              q_emb_3.unsqueeze(dim=1)], dim=1))

        return self._cal_logit(q_emb.unsqueeze(dim=1), a_emb)

    def forward_3i(self, data):
        e_emb_1 = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        e_emb_2 = self.e_embedding(data[:, 0, 2])
        r_emb_2 = self.r_embedding(data[:, 0, 3])
        e_emb_3 = self.e_embedding(data[:, 0, 4])
        r_emb_3 = self.r_embedding(data[:, 0, 5])
        u_emb = self.u_embedding(data[:, 0, -2])
        a_emb = self.e_embedding(data[:, :, -1])

        ur_emb = self.r_embedding.weight[-1]
        q_emb_1 = self._projection(e_emb_1, r_emb_1)
        q_emb_2 = self._projection(e_emb_2, r_emb_2)
        q_emb_3 = self._projection(e_emb_3, r_emb_3)
        q_emb_4 = self._projection(u_emb, ur_emb)
        q_emb = self._intersection(torch.cat([q_emb_1.unsqueeze(dim=1), 
                                              q_emb_2.unsqueeze(dim=1), 
                                              q_emb_3.unsqueeze(dim=1),
                                              q_emb_4.unsqueeze(dim=1)], dim=1))

        return self._cal_logit(q_emb.unsqueeze(dim=1), a_emb)

    def get_loss(self, data, flag):
        if flag == '1p':
            logits = self.forward_1p(data)
        elif flag == '2p':
            logits = self.forward_2p(data)
        elif flag == '3p':
            logits = self.forward_3p(data)
        elif flag == '2i':
            logits = self.forward_2i(data)
        elif flag == '3i':
            logits = self.forward_3i(data)
        else:
            raise ValueError
        
        # return - torch.nn.functional.logsigmoid(logits[:, 0].unsqueeze(dim=-1) - logits[:, 1:]).mean()
        return ( - torch.nn.functional.logsigmoid(logits[:, 0]).mean() - torch.log(1 - torch.sigmoid(logits[:, 1:]) + 1e-10).mean()) / 2

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
        pred[i - 1] = 1
        idcg = (true * discount).sum()
        dcg = (pred * discount).sum()
        return (idcg / dcg).item()
        
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
                logits = model.forward_1p(possible)[0]
            elif flag == '2p':
                logits = model.forward_2p(possible)[0]
            elif flag == '3p':
                logits = model.forward_3p(possible)[0]
            elif flag == '2i':
                logits = model.forward_2i(possible)[0]
            elif flag == '3i':
                logits = model.forward_3i(possible)[0]
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
    
    print(f'MR: {r}', flush=True)
    print(f'MRR: {rr}', flush=True)
    print(f'Hit@10: {h10}', flush=True)
    print(f'Hit@20: {h20}', flush=True)
    print(f'nDCG@10: {ndcg10}', flush=True)
    print(f'nDCG@20: {ndcg20}', flush=True)
    
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
    parser.add_argument('--gamma', default=12, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--lr', default=1e-2, type=int)
    parser.add_argument('--wd', default=1e-5, type=int)
    parser.add_argument('--max_steps', default=100000, type=int)
    # vec, box, beta, gamma, fuzzy, cqd
    parser.add_argument('--base_model', default='vec', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--bs', default=1024, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--tolerance', default=3, type=int)
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
    N_rel, N_item, N_ent, N_user, train_dict = read_data(input_path)
    
    train_1p, test_1p = load_obj(input_path + '/input/1p_train.pkl'), load_obj(input_path + '/input/1p_test.pkl')
    train_2p, test_2p = load_obj(input_path + '/input/2p_train.pkl'), load_obj(input_path + '/input/2p_test.pkl')
    train_3p, test_3p = load_obj(input_path + '/input/3p_train.pkl'), load_obj(input_path + '/input/3p_test.pkl')
    train_2i, test_2i = load_obj(input_path + '/input/2i_train.pkl'), load_obj(input_path + '/input/2i_test.pkl')
    train_3i, test_3i = load_obj(input_path + '/input/3i_train.pkl'), load_obj(input_path + '/input/3i_test.pkl')
    
    lqa_dataset_1p_train = LQADatasetTrain(N_item, train_1p, cfg)
    lqa_dataset_1p_valid = LQADatasetTest(N_item, test_1p, cfg, stage='valid')
    lqa_dataset_1p_test = LQADatasetTest(N_item, test_1p, cfg, stage='test')
    lqa_dataset_2p_train = LQADatasetTrain(N_item, train_2p, cfg)
    lqa_dataset_2p_valid = LQADatasetTest(N_item, test_2p, cfg, stage='valid')
    lqa_dataset_2p_test = LQADatasetTest(N_item, test_2p, cfg, stage='test')
    lqa_dataset_3p_train = LQADatasetTrain(N_item, train_3p, cfg)
    lqa_dataset_3p_valid = LQADatasetTest(N_item, test_3p, cfg, stage='valid')
    lqa_dataset_3p_test = LQADatasetTest(N_item, test_3p, cfg, stage='test')
    lqa_dataset_2i_train = LQADatasetTrain(N_item, train_2i, cfg)
    lqa_dataset_2i_valid = LQADatasetTest(N_item, test_2i, cfg, stage='valid')
    lqa_dataset_2i_test = LQADatasetTest(N_item, test_2i, cfg, stage='test')
    lqa_dataset_3i_train = LQADatasetTrain(N_item, train_3i, cfg)
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
        
        loss = loss_1p + loss_2p + loss_3p + loss_2i + loss_3i
        
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
                r, rr, h10, h20, ndcg10, ndcg20 = evaluate(valid_dataloaders[i], model, device, train_dict, flag)
                mmrr.append(rr)
            value = sum(mmrr) / len(mmrr)
            if value >= max_value:
                max_value = value
                tolerance = cfg.tolerance
            else:
                tolerance -= 1

        if (tolerance == 0) or ((step + 1) == cfg.max_steps):
            print('Testing...')
            for i in range(len(query_types)):
                flag = query_types[i]
                print(f'{flag}:')
                r, rr, h10, h20, ndcg10, ndcg20 = evaluate(test_dataloaders[i], model, device, train_dict, flag)
            break
