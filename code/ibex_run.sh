#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J LogicRec
#SBATCH -o LogicRec.%J.out
#SBATCH -e LogicRec.%J.err
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --constraint=[v100]

# python logicrec.py --verbose 0 --add_loss 1 --lr 0.01
# python ppc.py --dataset amazon-book
python ppc.py --dataset last-fm
