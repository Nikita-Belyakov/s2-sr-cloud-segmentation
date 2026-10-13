"""Distribution of B10 (channel 12 of the cloudy patches, DN = reflectance x 10000) by class and for the small clouds: to choose cirrus thresholds from the data."""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
from pathlib import Path

import numpy as np
import tifffile as tff
from PIL import Image
from scipy import ndimage

D = WORK_DIR
S8 = np.ones((3, 3))
pix = {0: [], 1: [], 2: []}   # raw png values: 0 clear, 1 cloud, 2 shadow
comp_mean = {"small cloud (<50 px)": [], "medium cloud (50-500 px)": [], "large cloud (>=500 px)": []}
patch_frac = {t: [] for t in (10, 20, 30, 50, 100)}
for sp in ("train", "val", "test"):
    for f in sorted((D / "cloudy_split" / sp).glob("*.tif"))[::2]:
        a = tff.imread(f)
        b10 = a[12].astype(np.float32)
        m = np.array(Image.open(f.with_suffix(".png")))
        for c in pix:
            if (m == c).any():
                pix[c].append(b10[m == c][::10])
        for t in patch_frac:
            patch_frac[t].append(float((b10 > t).mean()))
        lab, n = ndimage.label(m == 1, structure=S8)
        if n:
            sz = np.bincount(lab.ravel())[1:]
            mu = ndimage.mean(b10, lab, index=np.arange(1, n + 1))
            for s, v in zip(sz, mu):
                comp_mean["small cloud (<50 px)" if s < 50 else "medium cloud (50-500 px)" if s < 500 else "large cloud (>=500 px)"].append(v)
L = ["# B10 (DN) statistics", "", "## Pixels by class", "", "| class | p50 | p90 | p99 | p99.9 | share > 10 | share > 20 | share > 30 | share > 50 | share > 100 |", "|---|---|---|---|---|---|---|---|---|---|"]
for c, nm in ((0, "clear"), (1, "cloud"), (2, "shadow")):
    x = np.concatenate(pix[c])
    L.append(f"| {nm} | " + " | ".join(f"{np.percentile(x, p):.1f}" for p in (50, 90, 99, 99.9)) + " | " + " | ".join(f"{(x > t).mean():.3f}" for t in (10, 20, 30, 50, 100)) + " |")
L += ["", "## Mean B10 of the connected cloud objects", "", "| object size | objects | p10 | p25 | p50 | p75 | p90 |", "|---|---|---|---|---|---|---|"]
for k, v in comp_mean.items():
    v = np.array(v)
    L.append(f"| {k} | {len(v):,} | " + " | ".join(f"{np.percentile(v, p):.1f}" for p in (10, 25, 50, 75, 90)) + " |")
L += ["", "## Patches: share of patches with a given fraction of pixels above the B10 threshold", "", "| B10 threshold | patches with >= 1% | >= 5% | >= 10% | >= 25% |", "|---|---|---|---|---|"]
for t, v in patch_frac.items():
    v = np.array(v)
    L.append(f"| > {t} DN | " + " | ".join(f"{(v >= q).mean():.3f}" for q in (0.01, 0.05, 0.10, 0.25)) + " |")
(D / "b10_stats.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
