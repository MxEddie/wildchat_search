#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH -t 4:00:00  # time requested in hour:minute:seconds

export HF_TOKEN="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"

source .wildchat/bin/activate

if [ $5 = "promptgen" ]
then 
python3 src/prep_phrases.py $1
else 
    if test -f ./{$1}prompts.txt; then
    echo "Prompts found"
    else 
    python3 src/prep_phrases.py $1
    fi
fi 

if [ -z "$4" ]
then 
python3 src/search_dataset.py --task $1 --country $2 -of $3 

else
python3 src/search_dataset.py --task $1 --country $2 -of $3 --limit $4 
fi 
