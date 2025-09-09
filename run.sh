#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH -t 4:00:00  # time requested in hour:minute:seconds

export HF_TOKEN="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"

export TOKENIZERS_PARALLELISM=false


source .new_wildchat/bin/activate

python3 src/search_dataset.py \
    --task $1 \
    -of $2 \
    --usecase $3 \
    --limit $4 \
    --dataset $5 \
    $6
    #--country $7 \

