#!/bin/bash
#SBATCH --job-name=erqa_s2v
#SBATCH --partition=gpu               # <- partition of your cluster
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

# run from the repository root:  sbatch slurm/erqa_sen2venus.sbatch
set -e
mkdir -p logs
source configs/env.sh

# Table 3 and Figure 4: ERQA against the VENuS reference (480 validation patches of 12 scenes)
python src/sr_benchmark/erqa_s2v_select.py
python -W ignore src/sr_benchmark/erqa_s2v_infer.py --model ediffsr
python -W ignore src/sr_benchmark/erqa_s2v_infer.py --model esrgan
python -W ignore src/sr_benchmark/erqa_s2v.py
python -W ignore src/sr_benchmark/make_fig_erqa_s2v.py
