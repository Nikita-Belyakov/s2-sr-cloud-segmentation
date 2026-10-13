"""Brightness of the labelled cloud objects (mean reflectance of B2, B3, B4, no clipping) and the contrast with their surroundings,
to choose a threshold between thin (semi-transparent) and dense clouds."""
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
groups = {"small (<50 px)": [], "medium (50-500 px)": [], "large (>=500 px)": []}
contrast = {k: [] for k in groups}
for sp in ("train", "val", "test"):
    for f in sorted((D / "cloudy_split" / sp).glob("*.tif"))[::2]:
        a = tff.imread(f).astype(np.float32) / 10000
        vis = a[1:4].mean(0)  # B2, B3, B4
        m = np.array(Image.open(f.with_suffix(".png")))
        cloud = m == 1
        lab, n = ndimage.label(cloud, structure=S8)
        if not n:
            continue
        sz = np.bincount(lab.ravel())[1:]
        mu = ndimage.mean(vis, lab, index=np.arange(1, n + 1))
        ring_mask = ndimage.binary_dilation(cloud, iterations=6) & ~ndimage.binary_dilation(cloud, iterations=2) & (m == 0)
        for i in range(n):
            g = "small (<50 px)" if sz[i] < 50 else "medium (50-500 px)" if sz[i] < 500 else "large (>=500 px)"
            groups[g].append(mu[i])
            if sz[i] < 50:
                obj = lab == i + 1
                ring = ndimage.binary_dilation(obj, iterations=6) & ~ndimage.binary_dilation(obj, iterations=2) & (m == 0)
                if ring.sum() >= 10:
                    contrast[g].append(mu[i] - vis[ring].mean())
L = ["# Visible brightness of the labelled cloud objects (mean of B2, B3, B4, reflectance)", "", "| objects | count | p10 | p25 | p50 | p75 | p90 | share < 0.20 | share < 0.25 | share < 0.30 | share < 0.35 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
for k, v in groups.items():
    v = np.array(v)
    L.append(f"| {k} | {len(v):,} | " + " | ".join(f"{np.percentile(v, p):.3f}" for p in (10, 25, 50, 75, 90)) + " | " + " | ".join(f"{(v < t).mean():.2f}" for t in (0.20, 0.25, 0.30, 0.35)) + " |")
c = np.array(contrast["small (<50 px)"])
L += ["", f"Contrast of the small clouds with the clear ring around them (object − ring, reflectance), {len(c)} objects: p10 {np.percentile(c, 10):.3f}, p25 {np.percentile(c, 25):.3f}, p50 {np.percentile(c, 50):.3f}, p75 {np.percentile(c, 75):.3f}, p90 {np.percentile(c, 90):.3f}"]
(D / "thin_stats.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
