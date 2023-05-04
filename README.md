# LogicRec

This is a demonstrative implementation of our SIGIR 2023 [paper](https://arxiv.org/abs/2304.11722) LogicRec: Recommendation with Users' Logical Requirements

## Abstract

Users may demand recommendations with highly personalized requirements involving logical operations, e.g., the intersection of two requirements, where such requirements naturally form structured logical queries on knowledge graphs (KGs). To date, existing recommender systems lack the capability to tackle users' complex logical requirements. In this work, we formulate the problem of recommendation with users' logical requirements (LogicRec) and construct benchmark datasets for LogicRec. Furthermore, we propose an initial solution for LogicRec based on $\textit{logical requirement}$ retrieval and $\textit{user preference}$ retrieval, where we face two challenges. First, KGs are incomplete in nature. Therefore, there are always missing true facts, which entails that the answers to logical requirements can not be completely found in KGs. In this case, item selection based on the answers to logical queries is not applicable. We thus resort to logical query embedding (LQE) to jointly infer missing facts and retrieve items based on logical requirements. Second, answer sets are under-exploited. Existing LQE methods can only deal with query-answer pairs, where queries in our case are the intersected user preferences and logical requirements. However, the logical requirements and user preferences have different answer sets, offering us richer knowledge about the requirements and preferences by providing requirement-item and preference-item pairs. Thus, we design a multi-task knowledge-sharing mechanism to exploit these answer sets collectively. Extensive experimental results demonstrate the significance of the LogicRec task and the effectiveness of our proposed method.

## Requirements
* python == 3.8.5
* torch == 1.8.1
* numpy == 1.19.2
* pandas == 1.0.1
* tqdm == 4.61.0

## Datasets
Download and unzip the preprocessed [amazon-book](https://drive.google.com/file/d/10sLVpfbBEBLp-MFc7_Bfz75puYEIaMSM/view?usp=share_link) and [yelp2018](https://drive.google.com/file/d/1NYYzSOmuLZ37PIYc5OIoLAcVDqrZ0rd5/view?usp=share_link) datasets to **./data/**. If you wanna run query generation, please run the following commands after you have downloaded the datasets:

```powershell
python ppc_amazon_book.py 
```

```powershell
python ppc_yelp2018.py 
```


## Run

Tunable Arguments:
- emb_dim: embedding dimension
- lr: learning rate
- bs: batchsize
- wd: weight decay
- gamma: the margin to compute logits

Supporting Arguments:
- data_root: your preferred directory to store data
- cached: using the cached input or not
- dataset: amazon-book or yelp2018
- seed: random seed
- k: number of rec and lqa answer for each logicrec answer
- max_steps: maximum training steps
- num_workers: number of torch dataloader threads
- verbose: to show the progress bar or not
- gpu: an available gpu id
- tolerance: number of validations until executing early stop
- valid_interval: number of epochs between validations

Please run the code via:

```powershell
nohup python logicrec.py --verbose 0 --dataset amazon-book --cached 0 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --emb_dim 256 >amazon-book.log 2>&1 &
```

```powershell
nohup python logicrec.py --verbose 0 --dataset yelp2018 --cached 0 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --emb_dim 256 >amazon-book.log 2>&1 &
```

Please set the argument **cached** to 1 after first run for each dataset.

## Reference

```
@article{tang2023logicrec,
  title={LogicRec: Recommendation with Users' Logical Requirements},
  author={Tang, Zhenwei and Floto, Griffin and Toroghi, Armin and Pei, Shichao and Zhang, Xiangliang and Sanner, Scott},
  journal={arXiv preprint arXiv:2304.11722},
  year={2023}
}
```
