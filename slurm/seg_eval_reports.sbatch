#!/bin/bash
#SBATCH --job-name=seg_eval
#SBATCH --partition=gpu               # <- partition of your cluster
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out

# run from the repository root:  sbatch slurm/seg_eval_reports.sbatch
set -e
mkdir -p logs
source configs/env.sh

# Evaluation of the trained models and the reports that the tables are built from (run after seg_fixed and seg_cv)
python -W ignore src/segmentation/seg_aggregate.py          # seg_report/seg_report.md  (fixed split)
python -W ignore src/segmentation/seg_ensemble.py           # probability ensembles of the fixed split
python -W ignore src/segmentation/seg_boundary_eval.py      # boundary / object-size metrics of the fixed split
python -W ignore src/segmentation/seg_cv_report.py          # cv_report.md (cross-validation + boundary metrics)
python -W ignore src/segmentation/seg_small_eval.py fixed
python -W ignore src/segmentation/seg_small_eval.py cv
python -W ignore src/segmentation/seg_small_report.py       # small_report.md
python -W ignore src/segmentation/seg_thin_eval.py fixed
python -W ignore src/segmentation/seg_thin_eval.py cv
python -W ignore src/segmentation/seg_thin_report.py        # thin_report.md
python -W ignore src/segmentation/make_fig_cv.py            # Figure 7
python src/segmentation/build_tables.py                     # Tables 3-10 (LaTeX) -> $WORK_DIR/tables
