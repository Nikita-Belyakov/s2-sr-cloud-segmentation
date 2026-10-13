"""Report for seg_small_eval.py: IoU / F1 / precision / recall / detection rate and confusion matrices of the SMALL clouds (and shadows).
Output: seg_small_report/small_report.md + confusion-matrix figures."""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import json
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = WORK_DIR
OUT = D / "seg_small_report"
OUT.mkdir(exist_ok=True)
SETUPS = ["orig", "bicubic", "ediffsr", "esrgan", "orig_ediffsr", "orig_esrgan"]
LAB = {"orig": "Original 8ch (256)", "bicubic": "Bicubic x2 8ch (512)", "ediffsr": "EDiffSR 8ch (512)",
       "esrgan": "ESRGAN 8ch (512)", "orig_ediffsr": "Orig + EDiffSR 16ch (512)", "orig_esrgan": "Orig + ESRGAN 16ch (512)"}
CL = ["clear", "shadow", "cloud"]
IDX = {"shadow": 1, "cloud": 2}
R = {"fixed": {s: {} for s in SETUPS}, "cv": {s: {} for s in SETUPS}}
for f in sorted((D / "seg_small").glob("*.json")):
    r = json.loads(f.read_text())
    if r["setup"] not in SETUPS:
        continue
    R[r["protocol"]][r["setup"]][(r["fold"], r["seed"])] = r


def met(r, cls):
    c = IDX[cls]
    cf = np.array(r[cls]["confusion"], float)
    tp = cf[c, c]
    fn = cf[c].sum() - tp
    fp = cf[:, c].sum() - tp
    return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "detection": r[cls]["detected"] / max(r[cls]["gt_small_objects"], 1)}


def ms(v, p=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{p}f} ± {v.std(ddof=1):.{p}f}" if len(v) > 1 else f"{v.mean():.{p}f}"


KEYS = ["iou", "f1", "precision", "recall", "detection"]
HDR = "| setup | runs | IoU | F1 | precision | recall | detection rate |\n|---|---|---|---|---|---|---|"


def table(protocol, cls, pooled=False):
    L = [HDR]
    for s in SETUPS:
        runs = R[protocol][s]
        if not runs:
            continue
        if pooled:  # sum the folds of one seed
            vals = []
            for sd in sorted({k[1] for k in runs}):
                ks = [k for k in runs if k[1] == sd]
                if len(ks) < 4:
                    continue
                agg = {cls: {"confusion": sum(np.array(runs[k][cls]["confusion"]) for k in ks).tolist(),
                             "gt_small_objects": sum(runs[k][cls]["gt_small_objects"] for k in ks), "detected": sum(runs[k][cls]["detected"] for k in ks)}}
                vals.append(met(agg, cls))
        else:
            vals = [met(r, cls) for r in runs.values()]
        L.append(f"| {LAB[s]} | {len(vals)} | " + " | ".join(ms([v[k] for v in vals]) for k in KEYS) + " |")
    return L + [""]


def pairs(protocol, cls, title):
    P = [("bicubic", "orig"), ("ediffsr", "bicubic"), ("esrgan", "bicubic"),
         ("orig_ediffsr", "bicubic"), ("orig_esrgan", "bicubic")]
    L = [f"#### {title}", "", "Paired by run; mean ± std of the difference; (runs with a positive difference / runs, paired t-test p).", "",
         "| setup − reference | ΔIoU | ΔF1 | Δprecision | Δrecall | Δdetection rate |", "|---|---|---|---|---|---|"]
    for a, b in P:
        ids = sorted(set(R[protocol][a]) & set(R[protocol][b]))
        if len(ids) < 2:
            continue
        cells = []
        for k in KEYS:
            d = np.array([met(R[protocol][a][i], cls)[k] - met(R[protocol][b][i], cls)[k] for i in ids])
            p = stats.ttest_rel(d, np.zeros_like(d)).pvalue if len(d) > 2 and d.std() > 0 else float("nan")
            cells.append(f"{d.mean():+.3f} ± {d.std(ddof=1):.3f} ({(d > 0).sum()}/{len(d)}, p={p:.2f})")
        L.append(f"| {LAB[a]} − {LAB[b]} | " + " | ".join(cells) + " |")
    return L + [""]


def pooled_conf(protocol, s, cls):
    runs = R[protocol][s]
    return np.mean([sum(np.array(runs[k][cls]["confusion"], float) for k in runs if k[1] == sd) for sd in sorted({k[1] for k in runs})], 0)


def conf_md(protocol, cls):
    L = ["Rows = ground truth, columns = prediction; row-normalised %, then pixel counts (mean over seeds, pooled over all test scenes).", ""]
    for s in SETUPS:
        if not R[protocol][s]:
            continue
        cf = pooled_conf(protocol, s, cls)
        nr = cf / np.maximum(cf.sum(1, keepdims=True), 1)
        L += [f"**{LAB[s]}**", "", "| GT \\ pred | clear % | shadow % | cloud % | | clear px | shadow px | cloud px |", "|---|---|---|---|---|---|---|---|"]
        for i in range(3):
            L.append(f"| {CL[i]} | " + " | ".join(f"{100 * nr[i, j]:.1f}" for j in range(3)) + " | | " + " | ".join(f"{int(round(cf[i, j])):,}" for j in range(3)) + " |")
        L.append("")
    return L


def conf_fig(protocol, cls, name, title):
    have = [s for s in SETUPS if R[protocol][s]]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.2))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, s in zip(axs.ravel(), have):
        ax.axis("on")
        cf = pooled_conf(protocol, s, cls)
        nr = cf / np.maximum(cf.sum(1, keepdims=True), 1)
        ax.imshow(nr, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(CL); ax.set_yticklabels(CL)
        ax.set_xlabel("predicted"); ax.set_ylabel("ground truth"); ax.set_title(LAB[s], fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * nr[i, j]:.1f}%\n{int(round(cf[i, j])):,}", ha="center", va="center", fontsize=8, color="w" if nr[i, j] > .5 else "k")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / name, dpi=110); plt.close(fig)


rep = ["# Small clouds and small shadows: IoU, F1 and confusion matrices", "",
       "**Small** = a connected component (8-connectivity) of fewer than 50 pixels of the 256 grid (1 pixel = 100 m², so < 0.5 ha). "
       "**Region** of a class = pixels that belong to a small component of that class in the ground truth or in the prediction. "
       "Inside the region: TP = ground truth and prediction both say the class, FN = only the ground truth does, FP = only the prediction does; "
       "IoU = TP/(TP+FP+FN), F1 = 2TP/(2TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN). "
       "Detection rate = share of the small ground-truth objects of which at least half of the pixels are predicted correctly. "
       "The confusion matrices count the pixels of the same region by their true and predicted class (row `cloud` = small clouds and their misses; row `clear` = false alarms inside predicted small clouds).", "",
       "Test data: **fixed split** = scene 039_075 and the lower half of 040_049 (1075 patches), 3 seeds; **cross-validation** = all 8 scenes (5342 patches), every scene tested by models that did not see it, 4 folds × 2 seeds.", ""]
for cls in ("cloud", "shadow"):
    rep += [f"## Small {cls}s", ""]
    if any(R["fixed"][s] for s in SETUPS):
        rep += ["### Fixed split (1075 test patches), mean ± std over seeds", ""] + table("fixed", cls) + pairs("fixed", cls, "Paired differences, fixed split")
    if any(R["cv"][s] for s in SETUPS):
        rep += ["### Cross-validation, pooled over all 8 scenes (folds summed per seed, mean ± std over 2 seeds)", ""] + table("cv", cls, pooled=True)
        rep += ["### Cross-validation, mean ± std over runs (fold × seed)", ""] + table("cv", cls) + pairs("cv", cls, "Paired differences, cross-validation (per run)")
    for protocol, label in (("fixed", "fixed split"), ("cv", "cross-validation, all 8 scenes")):
        if any(R[protocol][s] for s in SETUPS):
            rep += [f"### Confusion matrices of the small-{cls} region, {label}", ""] + conf_md(protocol, cls)
            conf_fig(protocol, cls, f"small_{cls}_confusion_{protocol}.png", f"Small {cls}s: confusion matrices of the region of small {cls}s ({label}); row-normalised %, pixel counts")
(OUT / "small_report.md").write_text("\n".join(rep), encoding="utf-8")
print("report written:", OUT / "small_report.md")
