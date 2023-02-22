# LogicRec

This is the implementation of LogicRec: Recommendation with Users' Logical Requirements.

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