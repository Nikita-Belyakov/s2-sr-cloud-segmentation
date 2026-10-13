# Copy to configs/env.sh and edit. Sourced by the SLURM scripts in slurm/ (and can be sourced by hand).
# All variables are optional: the defaults are folders inside the repository (see s2sr/paths.py).

export WORK_DIR=$PWD/work                      # outputs and the cloudy data folders are here
export S2V_ROOT=/path/to/sen2venus             # contains npz/ and splits/
export WEIGHTS_DIR=/path/to/weights            # ediffsr_8ch.pth ediffsr_rgbnir.pth ediffsr_ir.pth esrgan_8ch.pth esrgan_rgbnir.pth esrgan_ir.pth

# python environment (conda, venv, module load ...)
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate s2sr
