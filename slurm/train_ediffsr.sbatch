#!/bin/bash
#SBATCH --job-name=ediffsr_train
#SBATCH --partition=gpu               # <- partition of your cluster
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

# Training of one EDiffSR model (4 GPUs). Run from the repository root:
#   sbatch slurm/train_ediffsr.sbatch setting2x_sen2venus_l1.yml          (8 bands, joint)
#   sbatch slurm/train_ediffsr.sbatch setting2xRGB_NIR_val_sen2venus.yml   (RGBNir, x2)
#   sbatch slurm/train_ediffsr.sbatch setting4xIR_sen2venus.yml            (IR, x4)
set -e
mkdir -p logs
source configs/env.sh
CFG=${1:-setting2x_sen2venus_l1.yml}
export S2V_NPZ=${S2V_NPZ:-$S2V_ROOT/npz}
export S2V_SPLITS=${S2V_SPLITS:-$S2V_ROOT/splits}
cd ediffsr/codes/config/sisr
python -m torch.distributed.launch --nproc_per_node=4 --master_port=13122 --use_env train_sen2venus.py \
    -opt options/train/$CFG --launcher pytorch
