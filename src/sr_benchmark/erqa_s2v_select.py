"""Choose the Sen2Venus test scenes: from valid.csv (patches the models did not see in training), the scene (site_tile_date) with the most valid patches
for each of 12 different sites, 40 random patches per scene. Output: erqa_s2v/selection.csv"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

S = str(S2V_SPLITS) + "/"
OUT = WORK_DIR / "erqa_s2v"
OUT.mkdir(exist_ok=True)
SITES = ["ARM_C_V2", "SO1_C_V2", "BAMBENW2", "ES-IC3XG", "KUDALIAR", "NARYN_C_", "BENGA_C_", "ATTO_C_V", "ALSACE_C", "ANJI_C_V", "MAD-AMBO", "JAM2018_"]
PER_SCENE = 40


def parse(fn):
    t = fn[:-4].split("_")
    k = next(i for i, x in enumerate(t) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", x))
    return "_".join(t[:k - 1]), t[k - 1], t[k], "_".join(t[:-1])  # site, tile, date, scene


va = pd.read_csv(S + "valid.csv")["filename"].tolist()
rng = np.random.default_rng(0)
rows = []
for site in SITES:
    cnt = Counter(parse(f)[3] for f in va if parse(f)[0] == site)
    if not cnt:
        print("no scene for", site)
        continue
    scene = cnt.most_common(1)[0][0]
    files = sorted(f for f in va if parse(f)[3] == scene)
    pick = sorted(rng.choice(files, min(PER_SCENE, len(files)), replace=False))
    for f in pick:
        s, tile, date, sc = parse(f)
        rows.append({"filename": f, "site": s, "tile": tile, "date": date, "scene": sc})
    print(f"{site}: scene {scene}, {len(files)} valid patches, {len(pick)} chosen")
pd.DataFrame(rows).to_csv(OUT / "selection.csv", index=False)
print(len(rows), "patches in", len({r['scene'] for r in rows}), "scenes")
