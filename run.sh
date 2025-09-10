#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH -t 4:00:00  # time requested in hour:minute:seconds

export HF_TOKEN="" #huggingface token

export TOKENIZERS_PARALLELISM=false

source .venv/bin/activate #activate the virtual environment

python3 src/search_dataset.py \
    --task $1 \
    -of $2 \
    --usecase $3 \
    --limit $4 \
    --dataset $5 \
    --match_only $6 \
    $7 
    #--country $7 \

