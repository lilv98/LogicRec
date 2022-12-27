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
        sample = torch.cat([torch.tensor([user, item]).unsqueeze(0), neg_i, neg_u], dim=0)
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
        pdb.set_trace()
        # neg_i = torch.cat([user.expand(self.num_ng // 2), neg_item], dim=0).view(-1, 2).t()
        # neg_u = torch.cat([neg_user, item.expand(self.num_ng // 2)], dim=0).view(-1, 2).t()
        # sample = torch.cat([torch.tensor([user, item]).unsqueeze(0), neg_i, neg_u], dim=0)
        # return sample

class RSModel(torch.nn.Module):
    def __init__(self, N_user, N_item, cfg):
        super().__init__()
        self.emb_dim = cfg.emb_dim
        self.rs_base_model = cfg.rs_base_model
        self.u_embedding = torch.nn.Embedding(N_user, cfg.emb_dim)
        self.i_embedding = torch.nn.Embedding(N_item, cfg.emb_dim)
        torch.nn.init.xavier_uniform_(self.u_embedding.weight.data)
        torch.nn.init.xavier_uniform_(self.i_embedding.weight.data)

    def _BPR(self, u_emb, i_emb):
        return (u_emb * i_emb).sum(dim=-1)

    def forward(self, data):
        u_emb = self.u_embedding(data[:, :, 0])
        i_emb = self.i_embedding(data[:, :, 1])
        if self.rs_base_model == 'BPR':
            return self._BPR(u_emb, i_emb)
        else:
            raise ValueError

    def get_loss(self, data):
        logits = self.forward(data)
        return - torch.nn.functional.logsigmoid(logits[:, 0].unsqueeze(dim=-1) - logits[:, 1:]).mean()

def evaluate(logits):
    pdb.set_trace()

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../data/', type=str)
    parser.add_argument('--dataset', default='amazon-book', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_ng', default=8, type=int)
    parser.add_argument('--emb_dim', default=256, type=int)
    parser.add_argument('--lr', default=1e-3, type=int)
    parser.add_argument('--wd', default=0, type=int)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--rs_base_model', default='BPR', type=str)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--bs', default=1024, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
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
    kge_dataset = KGEDataset(N_ent, kg, cfg)
    rs_dataset_train = RSDatasetTrain(N_user, N_item, train_dict, cfg)
    rs_dataset_test = RSDatasetTest(N_item, test_dict)

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
    rs_dataset_test = torch.utils.data.DataLoader(dataset=rs_dataset_test,
                                                batch_size=1,
                                                num_workers=cfg.num_workers,
                                                shuffle=True,
                                                drop_last=True)
    
    model_rs = RSModel(N_user, N_item, cfg)
    model_rs = model_rs.to(device)
    
    if cfg.verbose:
        kge_dataloader = tqdm.tqdm(kge_dataloader)
        rs_dataloader_train = tqdm.tqdm(rs_dataloader_train)
    
    optimizer_rs = torch.optim.Adam(model_rs.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    for epoch in range(cfg.max_epochs):
        print(f'Common -- Epoch {epoch + 1}:')
        model_rs.train()
        avg_loss = []
        for batch in zip(kge_dataloader, rs_dataloader_train):
            batch_rs = batch[1].to(device)
            loss_rs = model_rs.get_loss(batch_rs)
            optimizer_rs.zero_grad()
            loss_rs.backward()
            optimizer_rs.step()
            avg_loss.append(loss_rs.item())
        print(f'Loss: {round(sum(avg_loss) / len(avg_loss), 4)}')
    # pdb.set_trace()