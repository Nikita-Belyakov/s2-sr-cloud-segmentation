#!/bin/bash
#SBATCH --job-name=cloudy_sr
#SBATCH --partition=gpu               # <- partition of your cluster
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --array=0-11
# run from the repository root:  sbatch slurm/cloudy_sr.sbatch
set -e
mkdir -p logs
source configs/env.sh

# SR of the cloudy patches by both models (the output is split-wise: $WORK_DIR/cloudy_split_sr/<model>/<split>/<patch>.tif).
# For 12 parallel shards submit with --array=0-11 ; the shard index is SLURM_ARRAY_TASK_ID.
SHARD=${SLURM_ARRAY_TASK_ID:-0}
NSH=${NSHARDS:-12}
python -W ignore src/cloudy_sr/ediffsr_cloudy_infer.py \
    --split_root $WORK_DIR/cloudy_split --out_root $WORK_DIR/cloudy_split_sr/ediffsr_x2 \
    --weights ${EDIFFSR_8CH:-$WEIGHTS_DIR/ediffsr_8ch.pth} --dtype uint16 --batch 1 --workers 2 --seed $SHARD \
    --shard $SHARD/$NSH --skip_existing
python -W ignore src/cloudy_sr/esrgan_cloudy_infer.py \
    --split_root $WORK_DIR/cloudy_split --out_root $WORK_DIR/cloudy_split_sr/esrgan_x4_to512 \
    --weights ${ESRGAN_8CH:-$WEIGHTS_DIR/esrgan_8ch.pth} --norm train --out_size 512 --dtype uint16 \
    --shard $SHARD/$NSH --skip_existing
