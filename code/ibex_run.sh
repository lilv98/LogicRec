#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J LogicRec
#SBATCH -o LogicRec.%J.out
#SBATCH -e LogicRec.%J.err
#SBATCH --time=24:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# python gathered.py --verbose 0 --dataset amazon-book --which box-mmoe-all --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 256
# python gathered.py --verbose 0 --dataset amazon-book --which box-mmoe-rec --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 256
# python gathered.py --verbose 0 --dataset amazon-book --which box-mmoe-lqa --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 256
# python gathered.py --verbose 0 --dataset amazon-book --which box-only --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 256

python gathered.py --verbose 0 --dataset amazon-book --which beta-mmoe-all --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 128
python gathered.py --verbose 0 --dataset amazon-book --which beta-mmoe-rec --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 128
python gathered.py --verbose 0 --dataset amazon-book --which beta-mmoe-lqa --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 128
python gathered.py --verbose 0 --dataset amazon-book --which beta-only --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 128

# python gathered.py --verbose 0 --dataset amazon-book --which box-all --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 256
# python gathered.py --verbose 0 --dataset amazon-book --which beta-all --cached 1 --valid_interval 5000 --max_steps 1000000 --num_workers 4 --bs 4096 --tolerance 3 --wd 0 --gamma 1 --emb_dim 128