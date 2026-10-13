#!/bin/bash
#SBATCH --job-name=sr_eval
#SBATCH --partition=gpu               # <- partition of your cluster
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

# run from the repository root:  sbatch slurm/sr_eval_sen2venus.sbatch
set -e
mkdir -p logs
source configs/env.sh

# Table 1 and 2: EDiffSR, ESRGAN and bicubic on the same Sen2VENuS validation patches
N=${1:-1024}     # number of validation patches (evenly spaced over valid.csv)
TAG=${2:-main}
python -W ignore src/sr_benchmark/sr_eval_unified.py --stage ediffsr --n $N --tag $TAG
python -W ignore src/sr_benchmark/sr_eval_unified.py --stage esrgan  --n $N --tag $TAG
python -W ignore src/sr_benchmark/sr_eval_unified.py --stage esrgan4 --n $N --tag $TAG
python -W ignore src/sr_benchmark/sr_eval_unified.py --stage metrics --n $N --tag $TAG
mkdir -p $WORK_DIR/tables
python src/sr_benchmark/sr_eval_tables.py $TAG $WORK_DIR/tables
