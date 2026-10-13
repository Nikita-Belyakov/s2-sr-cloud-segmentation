"""Scene-level train/val/test split of the unpacked Sentinel-2 cloudy patches (tif + png copied together).

Whole scenes go to one split; scene mask84_040_049 is cut in two halves by patch row i (i <= 14 -> val, i >= 15 -> test).
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import csv
import re
import shutil
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
DST = Path(sys.argv[2])

TRAIN = {"mask84_012_055", "mask84_012_056", "mask84_038_073", "mask84_040_073", "mask84_040_074"}
VAL = {"mask84_019_025"}
TEST = {"mask84_039_075"}
HALF_SCENE, HALF_ROW = "mask84_040_049", 14  # i <= 14 -> val, i > 14 -> test


def split_of(stem):
    m = re.match(r"(mask\d+_\d+_\d+)_(\d+)_(\d+)$", stem)
    scene, i = m.group(1), int(m.group(2))
    if scene in TRAIN:
        return scene, "train"
    if scene in VAL:
        return scene, "val"
    if scene in TEST:
        return scene, "test"
    if scene == HALF_SCENE:
        return scene, "val" if i <= HALF_ROW else "test"
    raise ValueError(stem)


rows = []
for tif in sorted(SRC.glob("*.tif")):
    scene, sp = split_of(tif.stem)
    out = DST / sp
    out.mkdir(parents=True, exist_ok=True)
    for ext in (".tif", ".png"):
        shutil.copy2(SRC / (tif.stem + ext), out / (tif.stem + ext))
    rows.append((tif.stem, scene, sp))

with open(DST / "split_manifest.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["patch", "scene", "split"])
    w.writerows(rows)

for sp in ("train", "val", "test"):
    sel = [r for r in rows if r[2] == sp]
    print(sp, len(sel), "patches,", len({r[1] for r in sel}), "scenes")
print("total", len(rows))
