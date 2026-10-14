"""Paired differences of the SR inputs with respect to the ORIGINAL input in the scene-level cross-validation (8 runs = 4 folds x 2 seeds):
mean IoU, IoU of clouds / shadows, IoU and recall of small clouds and of thin clouds (outputs of seg_cv, seg_small_eval, seg_thin_eval).
Prints a markdown table; the figures of the form "n+/n" are the numbers of runs with a positive difference (paired t-test, descriptive).
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import WORK_DIR  # noqa: E402
import json

import numpy as np
from scipy import stats

D = WORK_DIR
SETUPS = ["orig", "bicubic", "ediffsr", "esrgan", "orig_ediffsr", "orig_esrgan"]
PAIRS = [("orig_esrgan", "orig"), ("orig_ediffsr", "orig"), ("bicubic", "orig"), ("esrgan", "orig"), ("ediffsr", "orig"),
         ("orig_esrgan", "bicubic"), ("orig_ediffsr", "bicubic")]


def load(sub):
    out = {s: {} for s in SETUPS}
    for f in sorted((D / sub).glob("*/results.json" if sub == "seg_cv" else "*.json")):
        r = json.loads(f.read_text())
        if r.get("setup") in SETUPS and (sub == "seg_cv" or r.get("protocol") == "cv"):
            out[r["setup"]][(r["fold"], r["seed"])] = r
    return out


def iou_rec(cf, c):
    cf = np.array(cf, float)
    tp = cf[c, c]
    fn = cf[c].sum() - tp
    fp = cf[:, c].sum() - tp
    return tp / max(tp + fp + fn, 1), tp / max(tp + fn, 1)


def block(name, M):
    rows = [f"### {name}", "", "| setup - reference | runs | mean difference | std | n+/n | p |", "|---|---|---|---|---|---|"]
    for a, b in PAIRS:
        ids = sorted(set(M[a]) & set(M[b]))
        if len(ids) < 3:
            continue
        x, y = np.array([M[a][i] for i in ids]), np.array([M[b][i] for i in ids])
        d = x - y
        p = stats.ttest_rel(x, y).pvalue
        rows.append(f"| {a} - {b} | {len(ids)} | {d.mean():+.3f} | {d.std(ddof=1):.3f} | {int((d > 0).sum())}/{len(d)} | {p:.3f} |")
    return rows + [""]


cv, sm, th = load("seg_cv"), load("seg_small"), load("seg_thin")
out = ["# Paired differences with respect to the original input (cross-validation, 4 folds x 2 seeds)", ""]
out += block("mean IoU", {s: {k: r["test_256"]["miou"] for k, r in cv[s].items()} for s in SETUPS})
out += block("IoU of clouds", {s: {k: r["test_256"]["iou"][2] for k, r in cv[s].items()} for s in SETUPS})
out += block("IoU of shadows", {s: {k: r["test_256"]["iou"][1] for k, r in cv[s].items()} for s in SETUPS})
for cls, c in (("cloud", 2),):
    out += block("small clouds, IoU", {s: {k: iou_rec(r[cls]["confusion"], c)[0] for k, r in sm[s].items()} for s in SETUPS})
    out += block("small clouds, recall", {s: {k: iou_rec(r[cls]["confusion"], c)[1] for k, r in sm[s].items()} for s in SETUPS})
for g, nm in (("thin", "thin clouds"), ("small_thin", "small thin clouds"), ("mid", "intermediate clouds"), ("dense", "dense clouds")):
    out += block(f"{nm}, IoU", {s: {k: iou_rec(r["groups"][g]["confusion"], 2)[0] for k, r in th[s].items()} for s in SETUPS})
    out += block(f"{nm}, recall", {s: {k: iou_rec(r["groups"][g]["confusion"], 2)[1] for k, r in th[s].items()} for s in SETUPS})
text = "\n".join(out)
(D / "seg_cv_report").mkdir(exist_ok=True)
(D / "seg_cv_report" / "paired_vs_orig.md").write_text(text, encoding="utf8")
print(text)
