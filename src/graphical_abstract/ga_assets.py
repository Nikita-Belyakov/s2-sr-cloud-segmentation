"""Thumbnails for the graphical abstract: input, bicubic, EDiffSR, ESRGAN (true colour and an IR composite) and the colour mask of one cloudy test patch."""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import sys
from pathlib import Path

import numpy as np
import tifffile as tff
import torch
import torch.nn.functional as F
from PIL import Image

D = WORK_DIR
OUT = D / "ga_assets"
OUT.mkdir(exist_ok=True)
SPLIT = "test"
TIF_IDX = [4, 5, 6, 8, 1, 2, 3, 7]
SRC = {"ediffsr": D / "cloudy_split_sr/ediffsr_x2", "esrgan": D / "cloudy_split_sr/esrgan_x4_to512"}
STEMS = sys.argv[1:] or ["mask84_039_075_15_18"]


def rgb(x, gain=3.0):
    return np.clip(np.stack([x[6], x[5], x[4]], -1) * gain, 0, 1)      # B4, B3, B2


def ir(x, gain=1.6):
    return np.clip(np.stack([x[7], x[6], x[5]], -1) * gain, 0, 1)      # B8, B4, B3


def save(a, name):
    Image.fromarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8)).save(OUT / name)


for stem in STEMS:
    raw = torch.from_numpy(np.clip(tff.imread(D / "cloudy_split" / SPLIT / f"{stem}.tif")[TIF_IDX].astype(np.float32) / 10000, 0, 0.5))
    m256 = np.array(Image.open(D / "cloudy_split" / SPLIT / f"{stem}.png"))
    bic = F.interpolate(raw[None], scale_factor=2, mode="bicubic", align_corners=False)[0].numpy()
    sr = {m: tff.imread(SRC[m] / SPLIT / f"{stem}.tif").astype(np.float32) / 10000 for m in SRC}
    r = raw.numpy()
    # the 10 m patch is shown on the 512 grid with nearest-neighbour so that it looks coarse
    r512 = np.repeat(np.repeat(r, 2, 1), 2, 2)
    save(rgb(r512), f"{stem}_input_rgb.png")
    save(ir(r512), f"{stem}_input_ir.png")
    save(rgb(bic), f"{stem}_bicubic_rgb.png")
    save(ir(bic), f"{stem}_bicubic_ir.png")
    for m in SRC:
        save(rgb(sr[m]), f"{stem}_{m}_rgb.png")
        save(ir(sr[m]), f"{stem}_{m}_ir.png")
    # colour mask on the 512 grid: label 1 = cloud, label 2 = shadow; clear = pale grey
    CLOUD, SHADOW = (0.20, 0.55, 0.95), (0.45, 0.15, 0.55)
    m512 = np.repeat(np.repeat(m256, 2, 0), 2, 1)
    col = np.zeros(m512.shape + (3,), np.float32)
    col[:] = (0.93, 0.93, 0.93)
    col[m512 == 1] = CLOUD
    col[m512 == 2] = SHADOW
    save(col, f"{stem}_mask.png")
    ov = rgb(bic).copy()
    for v, c in ((1, CLOUD), (2, SHADOW)):
        ov[m512 == v] = 0.45 * ov[m512 == v] + 0.55 * np.array(c)
    save(ov, f"{stem}_overlay.png")
    print("saved", stem, np.unique(m256))
