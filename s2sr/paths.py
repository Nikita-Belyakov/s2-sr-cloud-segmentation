"""Locations of data, model weights and outputs.

Every location can be set with an environment variable (see ``configs/env.example.sh``); the defaults are folders inside the repository.

WORK_DIR      working directory: the folder in which the scripts look for / write ``cloudy_split``, ``cloudy_split_sr``, ``sr_eval``, ``seg_cv`` ...
S2V_ROOT      Sen2VENuS folder with ``npz/`` (patches) and ``splits/`` (``train.csv``, ``valid.csv``, ``test_launch.csv``)
S2V_NPZ       the folder with the ``*.npz`` patches (default ``$S2V_ROOT/npz``)
S2V_SPLITS    the folder with the split csv files (default ``$S2V_ROOT/splits``)
WEIGHTS_DIR   folder with the checkpoints of the SR models (file names below)
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

def _p(name, default):
    return Path(os.environ.get(name, str(default)))


WORK_DIR = _p("WORK_DIR", REPO / "work")
WORK_DIR.mkdir(parents=True, exist_ok=True)
S2V_ROOT = _p("S2V_ROOT", REPO / "data" / "sen2venus")
S2V_NPZ = _p("S2V_NPZ", S2V_ROOT / "npz")
S2V_SPLITS = _p("S2V_SPLITS", S2V_ROOT / "splits")
WEIGHTS_DIR = _p("WEIGHTS_DIR", REPO / "weights")

# checkpoints of the six SR models (each file name can be overridden by the environment variable in the second column)
WEIGHTS = {
    "ediffsr_8ch": _p("EDIFFSR_8CH", WEIGHTS_DIR / "ediffsr_8ch.pth"),          # 8 bands, x2 (10 m -> 5 m) and x4 (20 m -> 5 m), joint
    "ediffsr_rgbnir": _p("EDIFFSR_RGBNIR", WEIGHTS_DIR / "ediffsr_rgbnir.pth"),  # B2, B3, B4, B8, x2
    "ediffsr_ir": _p("EDIFFSR_IR", WEIGHTS_DIR / "ediffsr_ir.pth"),              # B5, B6, B7, B8A, x4
    "esrgan_8ch": _p("ESRGAN_8CH", WEIGHTS_DIR / "esrgan_8ch.pth"),              # {'params_ema': state_dict}
    "esrgan_rgbnir": _p("ESRGAN_RGBNIR", WEIGHTS_DIR / "esrgan_rgbnir.pth"),
    "esrgan_ir": _p("ESRGAN_IR", WEIGHTS_DIR / "esrgan_ir.pth"),
}
