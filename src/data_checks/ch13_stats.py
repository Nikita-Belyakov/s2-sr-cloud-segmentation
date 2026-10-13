"""What is channel 13 of the cloudy patches (binary 0/1)? Cross-tabulation with the label masks and with channel 12 (B10)."""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
from pathlib import Path

import numpy as np
import tifffile as tff
from PIL import Image

D = WORK_DIR
tab = np.zeros((3, 2), np.int64)      # raw label (0 clear, 1 cloud, 2 shadow) x channel 13 (0/1)
b10 = {(c, v): [] for c in (0, 1, 2) for v in (0, 1)}
uniq = set()
pf, pl = [], []                       # per patch: share of flagged pixels, share of labelled cloud pixels
for sp in ("train", "val", "test"):
    for f in sorted((D / "cloudy_split" / sp).glob("*.tif"))[::2]:
        a = tff.imread(f)
        m = np.array(Image.open(f.with_suffix(".png")))
        c13 = a[13]
        uniq |= set(np.unique(c13).tolist())
        for c in (0, 1, 2):
            for v in (0, 1):
                s = (m == c) & (c13 == v)
                tab[c, v] += int(s.sum())
                if s.any():
                    b10[(c, v)].append(a[12][s][::10])
        pf.append(float((c13 > 0).mean()))
        pl.append(float((m == 1).mean()))
nm = {0: "clear", 1: "cloud", 2: "shadow"}
L = ["# Channel 13", "", f"distinct values: {sorted(uniq)}", "", "| label | channel 13 = 0 | channel 13 = 1 | share flagged |", "|---|---|---|---|"]
for c in (0, 1, 2):
    L.append(f"| {nm[c]} | {tab[c, 0]:,} | {tab[c, 1]:,} | {tab[c, 1] / tab[c].sum():.3f} |")
L += ["", f"flagged pixels that are labelled cloud: {tab[1, 1] / tab[:, 1].sum():.3f}, clear: {tab[0, 1] / tab[:, 1].sum():.3f}, shadow: {tab[2, 1] / tab[:, 1].sum():.3f}",
      f"labelled cloud pixels that are flagged: {tab[1, 1] / tab[1].sum():.3f}", "",
      f"correlation over patches between the flagged share and the labelled cloud share: {np.corrcoef(pf, pl)[0, 1]:.2f}", "",
      "mean of channel 12 (B10, DN) by label and channel 13:", "", "| label | ch13 = 0 | ch13 = 1 |", "|---|---|---|"]
for c in (0, 1, 2):
    L.append(f"| {nm[c]} | " + " | ".join(f"{np.concatenate(b10[(c, v)]).mean():.2f}" if b10[(c, v)] else "n/a" for v in (0, 1)) + " |")
(D / "ch13_stats.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
