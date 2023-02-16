#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J LogicRec
#SBATCH -o LogicRec.%J.out
#SBATCH -e LogicRec.%J.err
#SBATCH --time=12:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=[v100]

python logicrec.py --verbose 0 --dataset amazon-book --num_workers 16 --bs 64 --tolerance 10
# python lqa.py --verbose 0 --base_model vec --dataset amazon-book
# python lqa.py --verbose 0 --base_model box --dataset amazon-book
# python lqa.py --verbose 0 --base_model beta --dataset amazon-book
# python lqa.py --verbose 0 --base_model vec --dataset last-fm
# python lqa.py --verbose 0 --base_model box --dataset last-fm
# python lqa.py --verbose 0 --base_model beta --dataset last-fm
