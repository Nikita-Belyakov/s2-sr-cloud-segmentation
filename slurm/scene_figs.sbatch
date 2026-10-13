#!/bin/bash
#SBATCH --job-name=scene_figs
#SBATCH --partition=gpu               # <- partition of your cluster
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out

# run from the repository root:  sbatch slurm/scene_figs.sbatch
set -e
mkdir -p logs
source configs/env.sh

# Figures 2 and 3: the same validation scenes as in the draft, EDiffSR / ESRGAN / VENuS
python -W ignore src/sr_benchmark/scene_figs.py --stage ediffsr
python -W ignore src/sr_benchmark/scene_figs.py --stage esrgan
python -W ignore src/sr_benchmark/scene_figs.py --stage plot
