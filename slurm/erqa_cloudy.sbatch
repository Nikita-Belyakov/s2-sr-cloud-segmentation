#!/bin/bash
#SBATCH --job-name=erqa_cloudy
#SBATCH --partition=gpu               # <- partition of your cluster
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# run from the repository root:  sbatch slurm/erqa_cloudy.sbatch
set -e
mkdir -p logs
source configs/env.sh

# Table 4 and Figure 6, Figure 5: ERQA of the SR results against the interpolated source, spectra of the classes
python -W ignore src/cloudy_sr/erqa_cloudy.py test
python -W ignore src/cloudy_sr/erqa_cloudy_viz.py
python -W ignore src/cloudy_sr/make_fig_erqa_cloudy.py
python -W ignore src/cloudy_sr/spectral_analysis.py 2
python -W ignore src/cloudy_sr/make_fig_spectra.py 2
