"""Report for seg_thin_eval.py: IoU / F1 / precision / recall / detection rate and confusion matrices of THIN clouds (brightness proxy) and the other brightness groups.
Output: seg_thin_report/thin_report.md + confusion-matrix figures."""
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
OUT = D / "seg_thin_report"
OUT.mkdir(exist_ok=True)
SETUPS = ["orig", "bicubic", "ediffsr", "esrgan", "orig_ediffsr", "orig_esrgan"]
LAB = {"orig": "Original 8ch (256)", "bicubic": "Bicubic x2 8ch (512)", "ediffsr": "EDiffSR 8ch (512)",
       "esrgan": "ESRGAN 8ch (512)", "orig_ediffsr": "Orig + EDiffSR 16ch (512)", "orig_esrgan": "Orig + ESRGAN 16ch (512)"}
GNAME = {"thin": "Thin clouds (brightness < 0.15, any size)", "small_thin": "Small thin clouds (brightness < 0.15, < 50 px)",
         "mid": "Intermediate clouds (brightness 0.15-0.30)", "dense": "Dense clouds (brightness >= 0.30)"}
CL = ["clear", "shadow", "cloud"]
R = {"fixed": {s: {} for s in SETUPS}, "cv": {s: {} for s in SETUPS}}
for f in sorted((D / "seg_thin").glob("*.json")):
    r = json.loads(f.read_text())
    if r["setup"] not in SETUPS:
        continue
    R[r["protocol"]][r["setup"]][(r["fold"], r["seed"])] = r["groups"]
KEYS = ["iou", "f1", "precision", "recall", "detection"]


def met(g):
    cf = np.array(g["confusion"], float)
    tp = cf[2, 2]
    fn = cf[2].sum() - tp
    fp = cf[:, 2].sum() - tp
    return {"iou": tp / max(tp + fp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
            "detection": g["detected"] / max(g["gt_objects"], 1)}


def ms(v, p=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{p}f} ± {v.std(ddof=1):.{p}f}" if len(v) > 1 else f"{v.mean():.{p}f}"


def pooled(runs, grp):
    out = []
    for sd in sorted({k[1] for k in runs}):
        ks = [k for k in runs if k[1] == sd]
        if len(ks) < 4:
            continue
        out.append({"confusion": sum(np.array(runs[k][grp]["confusion"]) for k in ks).tolist(), "gt_objects": sum(runs[k][grp]["gt_objects"] for k in ks),
                    "detected": sum(runs[k][grp]["detected"] for k in ks)})
    return out


def table(protocol, grp, pool=False):
    L = ["| setup | runs | IoU | F1 | precision | recall | detection rate |", "|---|---|---|---|---|---|---|"]
    for s in SETUPS:
        runs = R[protocol][s]
        if not runs:
            continue
        vals = [met(g) for g in (pooled(runs, grp) if pool else [r[grp] for r in runs.values()])]
        L.append(f"| {LAB[s]} | {len(vals)} | " + " | ".join(ms([v[k] for v in vals]) for k in KEYS) + " |")
    return L + [""]


def pairs(protocol, grp):
    P = [("ediffsr", "bicubic"), ("esrgan", "bicubic"), ("orig_ediffsr", "bicubic"), ("orig_esrgan", "bicubic"), ("bicubic", "orig")]
    L = ["Paired by run; mean ± std of the difference; (runs with a positive difference / runs, paired t-test p).", "",
         "| setup − reference | ΔIoU | ΔF1 | Δprecision | Δrecall | Δdetection rate |", "|---|---|---|---|---|---|"]
    for a, b in P:
        ids = sorted(set(R[protocol][a]) & set(R[protocol][b]))
        if len(ids) < 2:
            continue
        cells = []
        for k in KEYS:
            d = np.array([met(R[protocol][a][i][grp])[k] - met(R[protocol][b][i][grp])[k] for i in ids])
            p = stats.ttest_rel(d, np.zeros_like(d)).pvalue if len(d) > 2 and d.std() > 0 else float("nan")
            cells.append(f"{d.mean():+.3f} ± {d.std(ddof=1):.3f} ({(d > 0).sum()}/{len(d)}, p={p:.2f})")
        L.append(f"| {LAB[a]} − {LAB[b]} | " + " | ".join(cells) + " |")
    return L + [""]


def pconf(protocol, s, grp):
    runs = R[protocol][s]
    return np.mean([sum(np.array(runs[k][grp]["confusion"], float) for k in runs if k[1] == sd) for sd in sorted({k[1] for k in runs})], 0)


def conf_md(protocol, grp):
    L = ["Rows = ground truth, columns = prediction; row-normalised %, then pixel counts (mean over seeds, pooled over the test scenes). "
         "The rows `clear` and `shadow` are always 100% `cloud` by construction (they are the false-alarm pixels inside predicted clouds of the group).", ""]
    for s in SETUPS:
        if not R[protocol][s]:
            continue
        cf = pconf(protocol, s, grp)
        nr = cf / np.maximum(cf.sum(1, keepdims=True), 1)
        L += [f"**{LAB[s]}**", "", "| GT \\ pred | clear % | shadow % | cloud % | | clear px | shadow px | cloud px |", "|---|---|---|---|---|---|---|---|"]
        for i in range(3):
            L.append(f"| {CL[i]} | " + " | ".join(f"{100 * nr[i, j]:.1f}" for j in range(3)) + " | | " + " | ".join(f"{int(round(cf[i, j])):,}" for j in range(3)) + " |")
        L.append("")
    return L


def conf_fig(protocol, grp, name, title):
    have = [s for s in SETUPS if R[protocol][s]]
    fig, axs = plt.subplots(2, 4, figsize=(17, 8.2))
    for ax in axs.ravel():
        ax.axis("off")
    for ax, s in zip(axs.ravel(), have):
        ax.axis("on")
        cf = pconf(protocol, s, grp)
        nr = cf / np.maximum(cf.sum(1, keepdims=True), 1)
        ax.imshow(nr, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(CL); ax.set_yticklabels(CL)
        ax.set_xlabel("predicted"); ax.set_ylabel("ground truth"); ax.set_title(LAB[s], fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * nr[i, j]:.1f}%\n{int(round(cf[i, j])):,}", ha="center", va="center", fontsize=8, color="w" if nr[i, j] > .5 else "k")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(OUT / name, dpi=110); plt.close(fig)


rep = ["# Thin (semi-transparent) clouds: IoU, F1 and confusion matrices", "",
       "**Thin-cloud proxy.** The channels of these patches contain no usable cirrus information (channel 12 never exceeds 11 DN, channel 13 is an undocumented binary cloud flag), "
       "so thin clouds are defined by the brightness of the cloud object: the mean reflectance of B2, B3, B4 over the pixels of a connected cloud component (no clipping). "
       "Thin < 0.15, intermediate 0.15-0.30, dense >= 0.30. Almost all small clouds (< 50 px) are dim: median brightness 0.10, contrast with the surroundings about +0.02. "
       "This is a proxy for semi-transparent clouds, it does not identify cirrus.", "",
       "**Region and metrics.** The region of a group = pixels of its objects in the ground truth or in the prediction (an object belongs to a group by its own brightness). "
       "Inside it: TP = both say cloud, FN = only the ground truth, FP = only the prediction; IoU = TP/(TP+FP+FN), F1 = 2TP/(2TP+FP+FN), precision = TP/(TP+FP), recall = TP/(TP+FN); "
       "detection rate = share of the ground-truth objects of the group of which at least half of the pixels are predicted as cloud.", "",
       "**Test data.** Fixed split: scene 039_075 and the lower half of 040_049 (1075 patches), 3 seeds. Cross-validation: all 8 scenes (5342 patches), each scene tested by models that did not see it, 4 folds × 2 seeds.", ""]
for grp in ("thin", "small_thin", "mid", "dense"):
    rep += [f"## {GNAME[grp]}", ""]
    if any(R["fixed"][s] for s in SETUPS):
        n = next(iter(R["fixed"]["orig"].values()))[grp]["gt_objects"] if R["fixed"]["orig"] else 0
        rep += [f"### Fixed split ({n} ground-truth objects in the test folder), mean ± std over 3 seeds", ""] + table("fixed", grp) + pairs("fixed", grp)
    if any(R["cv"][s] for s in SETUPS):
        n = sum(r[grp]["gt_objects"] for k, r in R["cv"]["orig"].items() if k[1] == 0) if R["cv"]["orig"] else 0
        rep += [f"### Cross-validation, pooled over all 8 scenes ({n} ground-truth objects), mean ± std over 2 seeds", ""] + table("cv", grp, pool=True)
        rep += ["### Cross-validation, mean ± std over runs (fold × seed)", ""] + table("cv", grp) + pairs("cv", grp)
    for protocol, label in (("fixed", "fixed split"), ("cv", "cross-validation, all 8 scenes")):
        if any(R[protocol][s] for s in SETUPS):
            rep += [f"### Confusion matrices, {GNAME[grp].lower()}, {label}", ""] + conf_md(protocol, grp)
            conf_fig(protocol, grp, f"{grp}_confusion_{protocol}.png", f"{GNAME[grp]}: confusion matrices of the region, {label}; row-normalised %, pixel counts")
(OUT / "thin_report.md").write_text("\n".join(rep), encoding="utf-8")
print("report written:", OUT / "thin_report.md")
