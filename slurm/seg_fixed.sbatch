#!/bin/bash
#SBATCH --job-name=seg_fixed
#SBATCH --partition=gpu               # <- partition of your cluster
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --array=0-20
# run from the repository root:  sbatch slurm/seg_fixed.sbatch
set -e
mkdir -p logs
source configs/env.sh

# Fixed split (train / val / test folders of cloudy_split): 7 inputs x 3 seeds = 21 array tasks (--array=0-20)
SETUPS=(orig bicubic ediffsr esrgan orig_ediffsr orig_esrgan resid_ediffsr)
IDX=${SLURM_ARRAY_TASK_ID:-0}
SEED=$((IDX / 7))
SETUP=${SETUPS[$((IDX % 7))]}
python -W ignore src/segmentation/seg_experiment.py --setup $SETUP --seed $SEED --epochs 30 --pretrained imagenet
