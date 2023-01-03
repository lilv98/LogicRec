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

    kg = []
    with open(path + '/input/kg_train.txt') as f:
        for line in f:
            line = line.strip('\n').split(' ')
            kg.append([int(line[0]), int(line[1]), int(line[2])])
    kg = pd.DataFrame(kg, columns=['h', 'r', 't'])
    
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
    return N_rel, N_item, N_ent, N_user, kg, train_dict, test_dict

class KGEDataset(torch.utils.data.Dataset):
    def __init__(self, N_ent, data, cfg):
        super().__init__()
        self.N_ent = N_ent
        self.num_ng = cfg.num_ng
        self.data = torch.tensor(data.values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        head, rel, tail = self.data[idx]
        negs = torch.tensor(np.random.choice(self.N_ent, self.num_ng))
        neg_t, neg_h = negs[:self.num_ng // 2].unsqueeze(dim=1), negs[self.num_ng // 2:].unsqueeze(dim=1)
        neg_t = torch.cat([torch.tensor([head, rel]).expand(self.num_ng // 2, -1), neg_t], dim=1)
        neg_h = torch.cat([neg_h, torch.tensor([rel, tail]).expand(self.num_ng // 2, -1)], dim=1)
        sample = torch.cat([torch.tensor([head, rel, tail]).unsqueeze(0), neg_t, neg_h], dim=0)
        return sample

class RSDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, N_user, N_item, data, cfg):
        super().__init__()
        self.N_user = N_user
        self.N_item = N_item
        self.num_ng = cfg.num_ng
        self.data = self._get_data(data)

    def _get_data(self, data):
        ret = []
        for user in data:
            items = data[user]
            for item in items:
                ret.append([user, item])
        return torch.tensor(ret)            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item = self.data[idx]
        neg_user = torch.tensor(np.random.choice(self.N_user, self.num_ng // 2)).unsqueeze(dim=1)
        neg_item = torch.tensor(np.random.choice(self.N_item, self.num_ng // 2)).unsqueeze(dim=1)
        neg_i = torch.cat([user.unsqueeze(dim=0).expand(self.num_ng // 2, 1), neg_item], dim=1)
        neg_u = torch.cat([neg_user, item.unsqueeze(dim=0).expand(self.num_ng // 2, 1)], dim=1)
        sample = torch.cat([torch.tensor([user, item]).unsqueeze(dim=0), neg_i, neg_u], dim=0)
        return sample

class RSDatasetTest(torch.utils.data.Dataset):
    def __init__(self, N_item, data):
        super().__init__()
        self.N_item = N_item
        self.data = self._get_data(data)

    def _get_data(self, data):
        ret = []
        for user in data:
            items = data[user]
            for item in items:
                ret.append([user, item])
        return torch.tensor(ret)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos = self.data[idx]
        users = pos[0].unsqueeze(dim=0).expand(N_item, 1)
        items = torch.arange(N_item).unsqueeze(dim=1)
        return torch.cat([users, items], dim=-1), pos

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

class RSModel(torch.nn.Module):
    def __init__(self, N_user, N_item, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.rs_base_model = cfg.rs_base_model
        self.u_embedding = torch.nn.Embedding(N_user, cfg.emb_dim)
        self.i_embedding = torch.nn.Embedding(N_item, cfg.emb_dim)
        torch.nn.init.xavier_uniform_(self.u_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.i_embedding.weight.data)

    def _BPRMF(self, u_emb, i_emb):
        return (u_emb * i_emb).sum(dim=-1)

    def forward(self, data):
        u_emb = self.u_embedding(data[:, :, 0])
        i_emb = self.i_embedding(data[:, :, 1])
        if self.rs_base_model == 'BPRMF':
            return self._BPRMF(u_emb, i_emb)
        else:
            raise ValueError

    def get_loss(self, data):
        logits = self.forward(data)
        return - torch.nn.functional.logsigmoid(logits[:, 0].unsqueeze(dim=-1) - logits[:, 1:]).mean()

class BasicIntersection(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.layer1 = torch.nn.Linear(self.dim, self.dim)
        self.layer2 = torch.nn.Linear(self.dim, self.dim)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, q_emb_1, q_emb_2):
        embeddings = torch.cat([q_emb_1.unsqueeze(dim=1), q_emb_2.unsqueeze(dim=1)], dim=1)
        layer1_act = torch.nn.functional.relu(self.layer1(embeddings))
        attention = torch.nn.functional.softmax(self.layer2(layer1_act), dim=0)
        return torch.sum(attention * embeddings, dim=1)


class LogicRecModel(torch.nn.Module):
    def __init__(self, N_user, N_ent, N_rel, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.gamma = cfg.gamma
        self.kge_base_model = cfg.kge_base_model
        self.rs_base_model = cfg.rs_base_model
        self.lqa_base_model = cfg.lqa_base_model
        self.e_embedding = torch.nn.Embedding(N_ent, cfg.emb_dim)
        self.r_embedding = torch.nn.Embedding(N_rel + 1, cfg.emb_dim)
        self.u_embedding = torch.nn.Embedding(N_user, cfg.emb_dim)
        self.basic_intersection = BasicIntersection(cfg.emb_dim)
        torch.nn.init.xavier_uniform_(self.e_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.r_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.u_embedding.weight.data)

    def _DistMult(self, h_emb, r_emb, t_emb):
        return (h_emb * r_emb * t_emb).sum(dim=-1)
    
    def _BPRMF(self, u_emb, i_emb):
        return (u_emb * i_emb).sum(dim=-1)

    def _projection(self, e_emb, r_emb):
        if self.lqa_base_model == 'GQE':
            return e_emb + r_emb
        else:
            raise ValueError

    def _intersection(self, q_emb_1, q_emb_2):
        if self.lqa_base_model == 'GQE':
            return self.basic_intersection(q_emb_1, q_emb_2)
        else:
            raise ValueError

    def _lqa(self, q_emb, a_emb):
        if self.lqa_base_model == 'GQE':
            return self.gamma - torch.norm(q_emb - a_emb, p=1, dim=-1)
        else:
            raise ValueError

    def forward_kge(self, data):
        h_emb = self.e_embedding(data[:, :, 0])
        r_emb = self.r_embedding(data[:, :, 1])
        t_emb = self.e_embedding(data[:, :, 2])
        if self.kge_base_model == 'DistMult':
            return self._DistMult(h_emb, r_emb, t_emb)
        else:
            raise ValueError

    def forward_rs(self, data):
        u_emb = self.u_embedding(data[:, :, 0])
        i_emb = self.e_embedding(data[:, :, 1])
        if self.rs_base_model == 'BPRMF':
            return self._BPRMF(u_emb, i_emb)
        else:
            raise ValueError
    
    def forward_1p(self, data):
        e_emb = self.e_embedding(data[:, 0, 0])
        r_emb = self.r_embedding(data[:, 0, 1])
        u_emb = self.u_embedding(data[:, 0, -2])
        ur_emb = self.r_embedding.weight[-1]
        a_emb = self.e_embedding(data[:, :, -1])
        q_emb_1 = self._projection(e_emb, r_emb)
        q_emb_2 = self._projection(u_emb, ur_emb)
        q_emb = self._intersection(q_emb_1, q_emb_2)
        return self._lqa(q_emb.unsqueeze(dim=1), a_emb)

    def forward_2p(self, data):
        e_emb = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        r_emb_2 = self.r_embedding(data[:, 0, 2])
        u_emb = self.u_embedding(data[:, 0, -2])
        ur_emb = self.r_embedding.weight[-1]
        a_emb = self.e_embedding(data[:, :, -1])
        q_emb_1 = self._projection(self._projection(e_emb, r_emb_1), r_emb_2)
        q_emb_2 = self._projection(u_emb, ur_emb)
        q_emb = self._intersection(q_emb_1, q_emb_2)
        return self._lqa(q_emb.unsqueeze(dim=1), a_emb)

    def forward_3p(self, data):
        e_emb = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        r_emb_2 = self.r_embedding(data[:, 0, 2])
        r_emb_3 = self.r_embedding(data[:, 0, 3])
        u_emb = self.u_embedding(data[:, 0, -2])
        ur_emb = self.r_embedding.weight[-1]
        a_emb = self.e_embedding(data[:, :, -1])
        q_emb_1 = self._projection(self._projection(self._projection(e_emb, r_emb_1), r_emb_2), r_emb_3)
        q_emb_2 = self._projection(u_emb, ur_emb)
        q_emb = self._intersection(q_emb_1, q_emb_2)
        return self._lqa(q_emb.unsqueeze(dim=1), a_emb)

    def forward_2i(self, data):
        e_emb_1 = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        e_emb_2 = self.e_embedding(data[:, 0, 2])
        r_emb_2 = self.r_embedding(data[:, 0, 3])
        u_emb = self.u_embedding(data[:, 0, -2])
        ur_emb = self.r_embedding.weight[-1]
        a_emb = self.e_embedding(data[:, :, -1])
        q_emb_1 = self._projection(e_emb_1, r_emb_1)
        q_emb_2 = self._projection(e_emb_2, r_emb_2)
        q_emb_3 = self._projection(u_emb, ur_emb)
        q_emb = self._intersection(q_emb_1, q_emb_2)
        q_emb = self._intersection(q_emb, q_emb_3)
        return self._lqa(q_emb.unsqueeze(dim=1), a_emb)

    def forward_3i(self, data):
        e_emb_1 = self.e_embedding(data[:, 0, 0])
        r_emb_1 = self.r_embedding(data[:, 0, 1])
        e_emb_2 = self.e_embedding(data[:, 0, 2])
        r_emb_2 = self.r_embedding(data[:, 0, 3])
        e_emb_3 = self.e_embedding(data[:, 0, 4])
        r_emb_3 = self.r_embedding(data[:, 0, 5])
        u_emb = self.u_embedding(data[:, 0, -2])
        ur_emb = self.r_embedding.weight[-1]
        a_emb = self.e_embedding(data[:, :, -1])
        q_emb_1 = self._projection(e_emb_1, r_emb_1)
        q_emb_2 = self._projection(e_emb_2, r_emb_2)
        q_emb_3 = self._projection(e_emb_3, r_emb_3)
        q_emb_4 = self._projection(u_emb, ur_emb)
        q_emb = self._intersection(q_emb_1, q_emb_2)
        q_emb = self._intersection(q_emb, q_emb_3)
        q_emb = self._intersection(q_emb, q_emb_4)
        return self._lqa(q_emb.unsqueeze(dim=1), a_emb)

    def get_loss(self, data, flag):
        if flag == 'kge':
            logits = self.forward_kge(data)
        elif flag == 'rs':
            logits = self.forward_rs(data)
        elif flag == '1p':
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
        return - torch.nn.functional.logsigmoid(logits[:, 0].unsqueeze(dim=-1) - logits[:, 1:]).mean()


def get_rank(pos, logits, flt):
    ranking = torch.argsort(logits, descending=True)
    rank = (ranking == pos[0, 1]).nonzero().item() + 1
    ranking_better = ranking[:rank - 1]
    if flt != None:
        for e in flt:
            if (ranking_better == e).sum() == 1:
                rank -= 1
    return rank

def evaluate(dataloader, model, device, train_dict):
    r = []
    rr = []
    h1 = []
    h3 = []
    h10 = []
    model.eval()
    with torch.no_grad():
        for possible, pos in dataloader:
            possible = possible.to(device)
            logits = model.forward_rs(possible)[0]
            flt = train_dict[pos[0][0].item()]
            rank = get_rank(pos, logits, flt)
            r.append(rank)
            rr.append(1/rank)
            if rank == 1:
                h1.append(1)
            else:
                h1.append(0)
            if rank <= 3:
                h3.append(1)
            else:
                h3.append(0)
            if rank <= 10:
                h10.append(1)
            else:
                h10.append(0)
    
    results = [r, rr, h1, h3, h10]
    
    r = int(sum(results[0])/len(results[0]))
    rr = round(sum(results[1])/len(results[1]), 3)
    h1 = round(sum(results[2])/len(results[2]), 3)
    h3 = round(sum(results[3])/len(results[3]), 3)
    h10 = round(sum(results[4])/len(results[4]), 3)
    
    print(r, flush=True)
    print(rr, flush=True)
    print(h1, flush=True)
    print(h3, flush=True)
    print(h10, flush=True)
    
    return r, rr, h1, h3, h10


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--dataset', default='amazon-book', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_ng', default=8, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--gamma', default=12, type=float)
    parser.add_argument('--lr', default=1e-2, type=int)
    parser.add_argument('--wd', default=1e-5, type=int)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--rs_base_model', default='BPRMF', type=str)
    parser.add_argument('--kge_base_model', default='DistMult', type=str)
    parser.add_argument('--lqa_base_model', default='GQE', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--bs', default=1024, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--tolerance', default=5, type=int)
    parser.add_argument('--valid_interval', default=5, type=int)
    return parser.parse_args(args)

if __name__ == '__main__':
    cfg = parse_args()
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)
    seed_everything(cfg.seed)
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    input_path = cfg.data_root + cfg.dataset
    N_rel, N_item, N_ent, N_user, kg, train_dict, test_dict = read_data(input_path)
    
    train_1p, test_1p = load_obj(input_path + '/input/1p_train.pkl'), load_obj(input_path + '/input/1p_test.pkl')
    train_2p, test_2p = load_obj(input_path + '/input/2p_train.pkl'), load_obj(input_path + '/input/2p_test.pkl')
    train_3p, test_3p = load_obj(input_path + '/input/3p_train.pkl'), load_obj(input_path + '/input/3p_test.pkl')
    train_2i, test_2i = load_obj(input_path + '/input/2i_train.pkl'), load_obj(input_path + '/input/2i_test.pkl')
    train_3i, test_3i = load_obj(input_path + '/input/3i_train.pkl'), load_obj(input_path + '/input/3i_test.pkl')
    
    kge_dataset = KGEDataset(N_ent, kg, cfg)
    rs_dataset_train = RSDatasetTrain(N_user, N_item, train_dict, cfg)
    rs_dataset_test = RSDatasetTest(N_item, test_dict)
    lqa_dataset_1p_train = LQADatasetTrain(N_item, train_1p, cfg)
    lqa_dataset_2p_train = LQADatasetTrain(N_item, train_2p, cfg)
    lqa_dataset_3p_train = LQADatasetTrain(N_item, train_3p, cfg)
    lqa_dataset_2i_train = LQADatasetTrain(N_item, train_2i, cfg)
    lqa_dataset_3i_train = LQADatasetTrain(N_item, train_3i, cfg)

    kge_dataloader = torch.utils.data.DataLoader(dataset=kge_dataset,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
    rs_dataloader_train = torch.utils.data.DataLoader(dataset=rs_dataset_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
    rs_dataloader_test = torch.utils.data.DataLoader(dataset=rs_dataset_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
    lqa_dataloader_1p_train = torch.utils.data.DataLoader(dataset=lqa_dataset_1p_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
    lqa_dataloader_2p_train = torch.utils.data.DataLoader(dataset=lqa_dataset_2p_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
    lqa_dataloader_3p_train = torch.utils.data.DataLoader(dataset=lqa_dataset_3p_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
    lqa_dataloader_2i_train = torch.utils.data.DataLoader(dataset=lqa_dataset_2i_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
    lqa_dataloader_3i_train = torch.utils.data.DataLoader(dataset=lqa_dataset_3i_train,
                                                batch_size=cfg.bs,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
    
    model = LogicRecModel(N_user, N_ent, N_rel, cfg)
    model = model.to(device)
    
    if cfg.verbose:
        kge_dataloader = tqdm.tqdm(kge_dataloader)
        rs_dataloader_train = tqdm.tqdm(rs_dataloader_train)
        rs_dataloader_test = tqdm.tqdm(rs_dataloader_test)
        lqa_dataloader_1p_train = tqdm.tqdm(lqa_dataloader_1p_train)
        lqa_dataloader_2p_train = tqdm.tqdm(lqa_dataloader_2p_train)
        lqa_dataloader_3p_train = tqdm.tqdm(lqa_dataloader_3p_train)
        lqa_dataloader_2i_train = tqdm.tqdm(lqa_dataloader_2i_train)
        lqa_dataloader_3i_train = tqdm.tqdm(lqa_dataloader_3i_train)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    
    tolerance = cfg.tolerance
    max_value = 0
    for epoch in range(cfg.max_epochs):
        print(f'Common -- Epoch {epoch + 1}:')
        model.train()
        avg_loss = []
        for batch in zip(kge_dataloader, 
                         rs_dataloader_train,
                         lqa_dataloader_1p_train,
                         lqa_dataloader_2p_train,
                         lqa_dataloader_3p_train,
                         lqa_dataloader_2i_train,
                         lqa_dataloader_3i_train
                         ):
            batch_kge = batch[0].to(device)
            batch_rs = batch[1].to(device)
            batch_1p = batch[2].to(device)
            batch_2p = batch[3].to(device)
            batch_3p = batch[4].to(device)
            batch_2i = batch[5].to(device)
            batch_3i = batch[6].to(device)
            
            loss_kge = model.get_loss(batch_kge, flag='kge')
            loss_rs = model.get_loss(batch_rs, flag='rs')
            loss_1p = model.get_loss(batch_1p, flag='1p')
            loss_2p = model.get_loss(batch_2p, flag='2p')
            loss_3p = model.get_loss(batch_3p, flag='3p')
            loss_2i = model.get_loss(batch_2i, flag='2i')
            loss_3i = model.get_loss(batch_3i, flag='3i')
            
            loss = loss_kge + loss_rs + loss_1p + loss_2p + loss_3p + loss_2i + loss_3i
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss.append(loss.item())
        print(f'Loss: {round(sum(avg_loss) / len(avg_loss), 4)}')
        
        if (epoch + 1) % cfg.valid_interval == 0:
            r, rr, h1, h3, h10 = evaluate(rs_dataloader_test, model, device, train_dict)
            if rr >= max_value:
                max_value = rr
                tolerance = cfg.tolerance
            else:
                tolerance -= 1

        if (tolerance == 0) or ((epoch + 1) == cfg.max_epochs):
            break
    # pdb.set_trace()