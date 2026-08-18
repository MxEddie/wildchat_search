#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH -t 4:00:00  # time requested in hour:minute:seconds

EXPORT HF_TOKEN=$4
#source .venv/bin/activate #activate the virtual environment

python3 src/search_dataset.py \
    -of $1 \
    --limit $2 \
    --dataset $3 \


