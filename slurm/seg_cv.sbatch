#!/bin/bash
#SBATCH --job-name=seg_cv
#SBATCH --partition=gpu               # <- partition of your cluster
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --array=0-7
# run from the repository root:  sbatch slurm/seg_cv.sbatch
set -e
mkdir -p logs
source configs/env.sh

# Scene-level cross-validation: 4 folds x 2 seeds = 8 array tasks (--array=0-7); every task trains the six inputs of the paper one after another.
FOLDS=(A B C D)
IDX=${SLURM_ARRAY_TASK_ID:-0}
FOLD=${FOLDS[$((IDX % 4))]}
SEED=$((IDX / 4))
for SETUP in orig bicubic ediffsr esrgan orig_ediffsr orig_esrgan; do
    python -W ignore src/segmentation/seg_experiment.py --setup $SETUP --fold $FOLD --seed $SEED --epochs 30 --pretrained imagenet
done
