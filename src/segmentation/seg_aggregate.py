"""Collect seg_runs/*/results.json -> seg_report/seg_report.md (+ csv + figures)."""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = WORK_DIR
OUT = D / "seg_report"
OUT.mkdir(exist_ok=True)
SETUPS = ["orig", "bicubic", "ediffsr", "esrgan", "orig_ediffsr", "orig_esrgan",
          "resid_ediffsr", "ens_bicubic_ediffsr", "ens_bicubic_esrgan", "ens_bicubic_bicubic"]
LAB = {"orig": "Original 8ch (256)", "bicubic": "Bicubic x2 8ch (512)", "ediffsr": "EDiffSR 8ch (512)", "esrgan": "ESRGAN 8ch (512)",
       "orig_ediffsr": "Orig + EDiffSR 16ch (512)", "orig_esrgan": "Orig + ESRGAN 16ch (512)",
       "resid_ediffsr": "Bicubic + (EDiffSR - Bicubic) 16ch (512)",
       "ens_bicubic_ediffsr": "Ensemble: Bicubic + EDiffSR models", "ens_bicubic_esrgan": "Ensemble: Bicubic + ESRGAN models",
       "ens_bicubic_bicubic": "Ensemble: 2 Bicubic models"}
CL = ["clear", "shadow", "cloud"]
R = {}
for s in SETUPS:
    R[s] = {}
    for f in sorted(D.glob(f"seg_runs/{s}_s*/results.json")):
        r = json.loads(f.read_text())
        R[s][r["seed"]] = r
have = [s for s in SETUPS if R[s]]


def ms(v, p=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{p}f} ± {v.std(ddof=1):.{p}f}" if len(v) > 1 else f"{v.mean():.{p}f}"


def vals(s, grid, fn):
    return [fn(r[grid]) for r in R[s].values()]


def main_table(grid, title):
    L = [f"### {title}", "", "| setup | seeds | IoU clear | IoU shadow | IoU cloud | mIoU | F1 clear | F1 shadow | F1 cloud | mF1 | accuracy |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in have:
        row = [LAB[s], str(len(R[s]))]
        row += [ms(vals(s, grid, lambda m, c=c: m["iou"][c])) for c in range(3)] + [ms(vals(s, grid, lambda m: m["miou"]))]
        row += [ms(vals(s, grid, lambda m, c=c: m["f1"][c])) for c in range(3)] + [ms(vals(s, grid, lambda m: m["mf1"]))]
        row += [ms(vals(s, grid, lambda m: m["accuracy"]))]
        L.append("| " + " | ".join(row) + " |")
    return L + [""]


def pr_table(grid, title):
    L = [f"### {title}", "", "| setup | precision clear/shadow/cloud | recall clear/shadow/cloud |", "|---|---|---|"]
    for s in have:
        p = [ms(vals(s, grid, lambda m, c=c: m["precision"][c]), 3) for c in range(3)]
        r = [ms(vals(s, grid, lambda m, c=c: m["recall"][c]), 3) for c in range(3)]
        L.append(f"| {LAB[s]} | {' / '.join(p)} | {' / '.join(r)} |")
    return L + [""]


def conf_mean(s, grid):
    C = np.array([r[grid]["confusion"] for r in R[s].values()], float)
    return C.mean(0), (C / C.sum(2, keepdims=True)).mean(0), (C / C.sum(2, keepdims=True)).std(0, ddof=1) if len(C) > 1 else np.zeros((3, 3))


def conf_tables(grid, title):
    L = [f"### {title}", "", "Rows = ground truth, columns = prediction. Left: row-normalised, % (mean ± std over seeds). "
         "Right: pixel counts (mean over seeds).", ""]
    for s in have:
        cnt, nrm, sd = conf_mean(s, grid)
        L += [f"**{LAB[s]}**", "", "| GT \\ pred | clear % | shadow % | cloud % | | clear px | shadow px | cloud px |", "|---|---|---|---|---|---|---|---|"]
        for i in range(3):
            L.append(f"| {CL[i]} | " + " | ".join(f"{100 * nrm[i, j]:.2f} ± {100 * sd[i, j]:.2f}" for j in range(3)) + " | | "
                     + " | ".join(f"{int(round(cnt[i, j])):,}" for j in range(3)) + " |")
        L.append("")
    return L


def delta_table(grid, title, pairs):
    """pairs: list of (setup, reference); difference setup - reference, paired by seed"""
    L = [f"### {title}", "", "Paired by seed (same seed number); mean ± std of the difference; `n+/n` = seeds with a positive difference.", "",
         "| setup − reference | ΔmIoU | ΔIoU shadow | ΔIoU cloud | ΔF1 shadow | ΔF1 cloud |", "|---|---|---|---|---|---|"]
    for s, ref in pairs:
        if s not in have or ref not in have:
            continue
        seeds = sorted(set(R[s]) & set(R[ref]))
        if not seeds:
            continue
        cells = []
        for fn in (lambda m: m["miou"], lambda m: m["iou"][1], lambda m: m["iou"][2], lambda m: m["f1"][1], lambda m: m["f1"][2]):
            d = np.array([fn(R[s][k][grid]) - fn(R[ref][k][grid]) for k in seeds])
            cells.append(f"{d.mean():+.3f} ± {d.std(ddof=1) if len(d) > 1 else 0:.3f} ({(d > 0).sum()}/{len(d)})")
        L.append(f"| {LAB[s]} − {LAB[ref]} | " + " | ".join(cells) + " |")
    return L + [""]


PAIRS_ORIG = [(s, "orig") for s in SETUPS if s != "orig"]
PAIRS_BICUBIC = [("ediffsr", "bicubic"), ("esrgan", "bicubic"), ("orig_ediffsr", "bicubic"), ("orig_esrgan", "bicubic"),
                  ("bicubic", "orig")]
PAIRS_NEW = [("ens_bicubic_ediffsr", "bicubic"), ("ens_bicubic_ediffsr", "ens_bicubic_bicubic"),
             ("ens_bicubic_esrgan", "bicubic"), ("ens_bicubic_esrgan", "ens_bicubic_bicubic"),
             ("ens_bicubic_bicubic", "bicubic"),
             ("resid_ediffsr", "bicubic"), ("resid_ediffsr", "orig_ediffsr")]


rep = ["# Cloud / cloud-shadow segmentation: DeepLabV3+ (timm-efficientnet-b0) on original vs SR inputs", "",
        "Test split (1075 patches, 2 held-out scenes / half scenes). Classes: clear, shadow, cloud. Mean ± std over seeds. "
        "Model selection: best epoch by validation mean IoU of shadow and cloud.", "",
        f"Runs found: " + ", ".join(f"{s}: {len(R[s])}" for s in SETUPS), ""]
rep += main_table("test_256", "Test metrics on the common 256 grid (SR predictions averaged over 2x2 blocks, original 256 masks) — primary, comparable across setups")
rep += main_table("test_native", "Test metrics on the native grid of each setup (256 for original, 512 for SR / concat setups)")
rep += pr_table("test_256", "Precision and recall per class (common 256 grid)")
rep += delta_table("test_256", "Difference to the original 8-channel baseline (common 256 grid)", PAIRS_ORIG)
rep += delta_table("test_256", "SR inputs versus bicubic interpolation (does the gain come from super-resolution or only from the 512 grid?)", PAIRS_BICUBIC)
rep += delta_table("test_256", "Residual channels and probability ensembles: what does EDiffSR / ESRGAN add to bicubic?", PAIRS_NEW)
rep += conf_tables("test_256", "Confusion matrices (common 256 grid)")
rep += conf_tables("test_native", "Confusion matrices (native grid)")
rep += main_table("val_256", "Validation metrics on the common 256 grid (for reference)")
trained = [s for s in have if not s.startswith("ens_")]
train_time = {s: np.mean([r["train_time_s"] for r in R[s].values()]) / 60 for s in trained}
rep += ["### Training (ensembles are built from the trained models above, no training of their own)", "",
        "| setup | channels | best epoch (per seed) | train time, min |", "|---|---|---|---|"]
for s in trained:
    rep.append(f"| {LAB[s]} | {list(R[s].values())[0]['channels']} | {[R[s][k]['best_epoch'] for k in sorted(R[s])]} | {train_time[s]:.0f} |")
(OUT / "seg_report.md").write_text("\n".join(rep), encoding="utf-8")

with open(OUT / "seg_results_all_runs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["setup", "seed", "grid", "iou_clear", "iou_shadow", "iou_cloud", "miou", "f1_clear", "f1_shadow", "f1_cloud", "mf1", "accuracy", "best_epoch"])
    for s in have:
        for k, r in sorted(R[s].items()):
            for g in ("test_256", "test_native"):
                m = r[g]
                w.writerow([s, k, g, *[f"{v:.5f}" for v in m["iou"]], f"{m['miou']:.5f}", *[f"{v:.5f}" for v in m["f1"]], f"{m['mf1']:.5f}",
                            f"{m['accuracy']:.5f}", r["best_epoch"]])

# figures
col = {"orig": "#1f77b4", "bicubic": "#7f7f7f", "ediffsr": "#d62728", "esrgan": "#2ca02c",
       "orig_ediffsr": "#ff9896", "orig_esrgan": "#98df8a", "resid_ediffsr": "#9467bd", "ens_bicubic_ediffsr": "#e377c2",
       "ens_bicubic_esrgan": "#17becf", "ens_bicubic_bicubic": "#bcbd22"}
n = len(have)
ncol = 4
nrow = (n + ncol - 1) // ncol
fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.3 * nrow))
axs = np.atleast_1d(axs).ravel()
for ax in axs[n:]:
    ax.axis("off")
for ax, s in zip(axs, have):
    _, nrm, _ = conf_mean(s, "test_256")
    ax.imshow(nrm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(CL); ax.set_yticklabels(CL)
    ax.set_xlabel("predicted"); ax.set_ylabel("ground truth"); ax.set_title(LAB[s], fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100 * nrm[i, j]:.1f}%", ha="center", va="center", color="w" if nrm[i, j] > .5 else "k")
fig.suptitle("Row-normalised confusion matrices, test split, common 256 grid (mean over seeds)")
fig.tight_layout(); fig.savefig(OUT / "confusion_matrices.png", dpi=120); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (key, ttl) in zip(axs, (("iou", "IoU"), ("f1", "F1"))):
    for i, s in enumerate(have):
        v = np.array([r["test_256"][key] for r in R[s].values()])
        bw = 0.84 / n
        ax.bar(np.arange(3) + i * bw - bw * (n - 1) / 2, v.mean(0), bw, yerr=v.std(0, ddof=1) if len(v) > 1 else None,
               color=col[s], label=LAB[s], capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(CL); ax.set_title(f"{ttl} per class (test, 256 grid)"); ax.grid(axis="y", alpha=.3)
axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "iou_f1_per_class.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
for s in trained:
    c = np.array([[h["val_fg_iou"] for h in r["history"]] for r in R[s].values()])
    ax.plot(c.mean(0), color=col[s], label=LAB[s])
ax.set_xlabel("epoch"); ax.set_ylabel("val mean IoU (shadow, cloud)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "val_curves.png", dpi=120); plt.close(fig)
print("report written:", OUT / "seg_report.md")
