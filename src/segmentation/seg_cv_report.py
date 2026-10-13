"""Report: (1) boundary / object-size metrics of the trained models of the fixed split (seg_boundary/*.json),
(2) scene-level cross-validation (seg_cv/*/results.json) incl. the same boundary metrics pooled over all 8 scenes.
Output: seg_cv_report/cv_report.md + figures.
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from seg_boundary import BIN_NAMES, bf_prf, biou, pool, size_precision, size_recall  # noqa: E402

D = WORK_DIR
OUT = D / "seg_cv_report"
OUT.mkdir(exist_ok=True)
SETUPS = ["orig", "bicubic", "ediffsr", "esrgan", "orig_ediffsr", "orig_esrgan", "resid_ediffsr"]
LAB = {"orig": "Original 8ch (256)", "bicubic": "Bicubic x2 8ch (512)", "ediffsr": "EDiffSR 8ch (512)", "esrgan": "ESRGAN 8ch (512)", "orig_ediffsr": "Orig + EDiffSR 16ch (512)",
       "orig_esrgan": "Orig + ESRGAN 16ch (512)", "resid_ediffsr": "Bicubic + (EDiffSR - Bicubic) 16ch (512)"}
FG = ["shadow", "cloud"]
CL = ["clear", "shadow", "cloud"]
COL = {"orig": "#1f77b4", "bicubic": "#7f7f7f", "ediffsr": "#d62728", "esrgan": "#2ca02c",
       "orig_ediffsr": "#ff9896", "orig_esrgan": "#98df8a", "resid_ediffsr": "#9467bd"}


def ms(v, p=3):
    v = np.asarray([x for x in v if x == x], float)
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.{p}f} ± {v.std(ddof=1):.{p}f}" if len(v) > 1 else f"{v.mean():.{p}f}"


# ------------------------------------------------------------------ load
B_MAIN = {s: {} for s in SETUPS}
for f in sorted((D / "seg_boundary").glob("*.json")):
    r = json.loads(f.read_text())
    if r["setup"] not in SETUPS:
        continue
    B_MAIN[r["setup"]][r["seed"]] = r["boundary"]
CV = {s: {} for s in SETUPS}
for f in sorted(D.glob("seg_cv/*/results.json")):
    r = json.loads(f.read_text())
    if r["setup"] not in SETUPS:
        continue
    CV[r["setup"]][(r["fold"], r["seed"])] = r
have_main = [s for s in SETUPS if B_MAIN[s]]
have_cv = [s for s in SETUPS if CV[s]]


def bmetrics(raw):
    """raw boundary counts -> flat dict of metrics"""
    p = pool([raw])
    m = {}
    for c in FG:
        for t in ("1", "2"):
            pr, rc, f = bf_prf(p["bf"][c][t])
            m[f"{c}_bf{t}"], m[f"{c}_bf{t}_p"], m[f"{c}_bf{t}_r"] = f, pr, rc
        m[f"{c}_biou2"] = biou(p["biou"][c]["2"])
        rec, det, n = size_recall(p["size_gt"][c])
        prec, npr = size_precision(p["size_pr"][c])
        for k in range(3):
            m[f"{c}_rec{k}"], m[f"{c}_det{k}"], m[f"{c}_prec{k}"] = rec[k], det[k], prec[k]
            m[f"{c}_nobj{k}"], m[f"{c}_npred{k}"] = n[k], npr[k]
    return m


def btables(runs, title, note):
    """runs: setup -> run id -> raw boundary counts"""
    M = {s: {k: bmetrics(v) for k, v in runs[s].items()} for s in SETUPS if runs[s]}
    L = [f"## {title}", "", note, ""]
    L += ["### Boundary F-score (tolerance 1 and 2 px = 10 / 20 m) and boundary IoU (band width 2 px), common 256 grid", "",
          "| setup | runs | shadow BF τ=1 | shadow BF τ=2 | shadow boundary IoU | cloud BF τ=1 | cloud BF τ=2 | cloud boundary IoU |", "|---|---|---|---|---|---|---|---|"]
    for s in M:
        v = lambda key: [m[key] for m in M[s].values()]  # noqa: E731
        L.append(f"| {LAB[s]} | {len(M[s])} | " + " | ".join(ms(v(f"{c}_{k}")) for c in FG for k in ("bf1", "bf2", "biou2")) + " |")
    L += ["", "Boundary precision / recall at τ=2 px (precision: predicted boundary lies on a true boundary; recall: the reverse)", "",
          "| setup | shadow P | shadow R | cloud P | cloud R |", "|---|---|---|---|---|"]
    for s in M:
        v = lambda key: [m[key] for m in M[s].values()]  # noqa: E731
        L.append(f"| {LAB[s]} | " + " | ".join(ms(v(f"{c}_bf2_{k}")) for c in FG for k in ("p", "r")) + " |")
    for c in FG:
        first = next(iter(runs["orig"].values())) if runs["orig"] else None
        n_obj = [int(pool([first])["size_gt"][c][k][0]) for k in range(3)] if first else [0, 0, 0]
        L += ["", f"### {c}: quality by object size (connected components, 256-grid pixels of 100 m²); "
              f"number of GT objects in one test set (one run): " + ", ".join(f"{BIN_NAMES[k]}: {n_obj[k]:,}" for k in range(3)), "",
              "| setup | recall " + " | recall ".join(BIN_NAMES) + " | detection rate " + " | detection rate ".join(BIN_NAMES)
              + " | precision " + " | precision ".join(BIN_NAMES) + " |", "|---|" + "---|" * 9]
        for s in M:
            v = lambda key: [m[key] for m in M[s].values()]  # noqa: E731
            L.append(f"| {LAB[s]} | " + " | ".join(ms(v(f"{c}_{q}{k}")) for q in ("rec", "det", "prec") for k in range(3)) + " |")
    return L + [""], M


def pairs_table(M, pairs, title, keys):
    L = [f"### {title}", "", "Paired by run id (seed, or fold+seed); mean ± std of the difference; `n+/n` = runs with a positive difference; p = paired t-test (descriptive).", "",
         "| setup − reference | " + " | ".join(k[1] for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for s, ref in pairs:
        if s not in M or ref not in M:
            continue
        ids = sorted(set(M[s]) & set(M[ref]))
        if len(ids) < 2:
            continue
        cells = []
        for key, _ in keys:
            d = np.array([M[s][i][key] - M[ref][i][key] for i in ids], float)
            d = d[~np.isnan(d)]
            p = stats.ttest_rel(d, np.zeros_like(d)).pvalue if len(d) > 2 and d.std() > 0 else float("nan")
            cells.append(f"{d.mean():+.3f} ± {d.std(ddof=1):.3f} ({(d > 0).sum()}/{len(d)}, p={p:.2f})")
        L.append(f"| {LAB[s]} − {LAB[ref]} | " + " | ".join(cells) + " |")
    return L + [""]


PAIRS = [("bicubic", "orig"), ("ediffsr", "bicubic"), ("esrgan", "bicubic"),
         ("orig_ediffsr", "bicubic"), ("orig_esrgan", "bicubic"), ("resid_ediffsr", "bicubic")]
DKEYS = [("shadow_bf2", "shadow BF τ=2"), ("cloud_bf2", "cloud BF τ=2"), ("shadow_biou2", "shadow bIoU"), ("cloud_biou2", "cloud bIoU"),
         ("shadow_rec0", "shadow recall <50 px"), ("cloud_rec0", "cloud recall <50 px"), ("cloud_det0", "cloud detection <50 px")]

rep = ["# Boundary / object-size metrics and scene-level cross-validation", ""]

# ------------------------------------------------------------------ part 1: boundary metrics, fixed split
if have_main:
    L, MM = btables(B_MAIN, "Part 1. Boundary and object-size metrics, fixed split (test = 039_075 + half of 040_049), 3 seeds",
                    "Models trained on the fixed train split; metrics on its test folder, computed on the common 256 grid. Mean ± std over seeds.")
    rep += L + pairs_table(MM, PAIRS, "Differences in boundary / small-object metrics (fixed split)", DKEYS)

# ------------------------------------------------------------------ part 2: cross-validation
if have_cv:
    n_runs = {s: len(CV[s]) for s in have_cv}
    rep += ["## Part 2. Scene-level cross-validation (4 folds × 2 seeds)", "",
            "Fold A test: 012_055 + 012_056; B: 040_073 + 040_074; C: 038_073 + 039_075; D: 019_025 + 040_049 (neighbouring scene numbers are kept together). "
            "Validation: one more scene (A: 039_075, B: 038_073, C: 040_049, D: 012_056), training: the remaining 5 scenes. "
            "Every scene is a test scene exactly once. Runs finished: " + ", ".join(f"{LAB[s]}: {n_runs[s]}/8" for s in have_cv), ""]
    # ---- standard metrics
    def runmet(s, fn):
        return {k: fn(r["test_256"]) for k, r in CV[s].items()}
    rep += ["### IoU / F1, test, common 256 grid — mean ± std over runs (fold × seed)", "",
            "| setup | runs | IoU shadow | IoU cloud | mIoU | F1 shadow | F1 cloud | mF1 |", "|---|---|---|---|---|---|---|---|"]
    for s in have_cv:
        v = lambda fn: list(runmet(s, fn).values())  # noqa: E731
        rep.append(f"| {LAB[s]} | {n_runs[s]} | " + " | ".join(ms(v(f)) for f in (
            lambda m: m["iou"][1], lambda m: m["iou"][2], lambda m: m["miou"], lambda m: m["f1"][1], lambda m: m["f1"][2], lambda m: m["mf1"])) + " |")
    rep += ["", "### Pooled over all 8 scenes (confusion matrices of the 4 folds summed, then metrics; mean ± std over seeds)", "",
            "| setup | IoU clear | IoU shadow | IoU cloud | mIoU | F1 shadow | F1 cloud | mF1 |", "|---|---|---|---|---|---|---|---|"]
    POOL = {}
    for s in have_cv:
        seeds = sorted({k[1] for k in CV[s]})
        per_seed = []
        for sd in seeds:
            ks = [k for k in CV[s] if k[1] == sd]
            if len(ks) < 4:
                continue
            conf = sum(np.array(CV[s][k]["test_256"]["confusion"], float) for k in ks)
            tp = np.diag(conf)
            fp, fn = conf.sum(0) - tp, conf.sum(1) - tp
            iou, f1 = tp / (tp + fp + fn), 2 * tp / (2 * tp + fp + fn)
            per_seed.append((iou, f1, conf))
        POOL[s] = per_seed
        if per_seed:
            I = np.array([p[0] for p in per_seed])
            Fm = np.array([p[1] for p in per_seed])
            rep.append(f"| {LAB[s]} | " + " | ".join(ms(x) for x in (I[:, 0], I[:, 1], I[:, 2], I.mean(1), Fm[:, 1], Fm[:, 2], Fm.mean(1))) + " |")
    rep += ["", "### mIoU per fold (mean over seeds)", "", "| setup | A | B | C | D |", "|---|---|---|---|---|"]
    for s in have_cv:
        cells = []
        for fo in "ABCD":
            v = [r["test_256"]["miou"] for k, r in CV[s].items() if k[0] == fo]
            cells.append(f"{np.mean(v):.3f}" if v else "n/a")
        rep.append(f"| {LAB[s]} | " + " | ".join(cells) + " |")
    ex = CV[have_cv[0]]
    rep += ["", "Fold sizes and class shares of the test scenes (pixels of the 256 grid):", "", "| fold | test patches | clear % | shadow % | cloud % |", "|---|---|---|---|---|"]
    for fo in "ABCD":
        ks = [k for k in ex if k[0] == fo]
        if ks:
            conf = np.array(ex[ks[0]]["test_256"]["confusion"], float).sum(1)
            rep.append(f"| {fo} | {ex[ks[0]]['n_test']} | " + " | ".join(f"{100 * conf[i] / conf.sum():.2f}" for i in range(3)) + " |")
    # paired differences of IoU metrics
    SM = {s: {k: {"miou": r["test_256"]["miou"], "iou_shadow": r["test_256"]["iou"][1], "iou_cloud": r["test_256"]["iou"][2],
                  "f1_shadow": r["test_256"]["f1"][1], "f1_cloud": r["test_256"]["f1"][2]} for k, r in CV[s].items()} for s in have_cv}
    rep += [""] + pairs_table(SM, PAIRS,
                              "Paired differences of IoU / F1 (cross-validation)",
                              [("miou", "ΔmIoU"), ("iou_shadow", "ΔIoU shadow"), ("iou_cloud", "ΔIoU cloud"), ("f1_shadow", "ΔF1 shadow"), ("f1_cloud", "ΔF1 cloud")])
    # confusion matrices pooled over folds
    rep += ["### Confusion matrices, pooled over all 8 scenes (row-normalised %, mean over seeds; rows = ground truth)", ""]
    for s in have_cv:
        if not POOL.get(s):
            continue
        conf = np.mean([p[2] for p in POOL[s]], 0)
        nrm = conf / conf.sum(1, keepdims=True)
        rep += [f"**{LAB[s]}**", "", "| GT \\ pred | clear % | shadow % | cloud % | | clear px | shadow px | cloud px |", "|---|---|---|---|---|---|---|---|"]
        for i in range(3):
            rep.append(f"| {CL[i]} | " + " | ".join(f"{100 * nrm[i, j]:.2f}" for j in range(3)) + " | | " + " | ".join(f"{int(round(conf[i, j])):,}" for j in range(3)) + " |")
        rep.append("")
    # boundary metrics, pooled over folds per seed
    CVB = {s: {} for s in SETUPS}
    for s in have_cv:
        for sd in sorted({k[1] for k in CV[s]}):
            ks = [k for k in CV[s] if k[1] == sd]
            if len(ks) == 4:
                CVB[s][sd] = pool([CV[s][k]["test_boundary"] for k in ks])
    if any(CVB[s] for s in have_cv):
        L, MB = btables(CVB, "Part 3. Boundary and object-size metrics, cross-validation pooled over all 8 scenes",
                        "For every seed the boundary counts of the 4 folds are summed (every scene is predicted by a model that has not seen it); mean ± std over seeds.")
        rep += L + pairs_table(MB, PAIRS, "Differences in boundary / small-object metrics (cross-validation, pooled per seed)", DKEYS)
        # also run-level (fold x seed) paired differences
        RB = {s: {k: bmetrics(r["test_boundary"]) for k, r in CV[s].items()} for s in have_cv}
        rep += pairs_table(RB, PAIRS, "Differences in boundary / small-object metrics (cross-validation, per run = fold × seed)", DKEYS)

    # ---- figures
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.8 / len(have_cv)
    for i, s in enumerate(have_cv):
        v = [np.mean([r["test_256"]["miou"] for k, r in CV[s].items() if k[0] == fo] or [np.nan]) for fo in "ABCD"]
        ax.bar(np.arange(4) + i * w - 0.4 + w / 2, v, w, color=COL[s], label=LAB[s])
    ax.set_xticks(range(4)); ax.set_xticklabels([f"fold {f}" for f in "ABCD"]); ax.set_ylabel("test mIoU"); ax.grid(axis="y", alpha=.3)
    ax.legend(fontsize=7, ncol=2); ax.set_title("Scene-level cross-validation: mIoU per fold (mean over seeds)")
    fig.tight_layout(); fig.savefig(OUT / "cv_miou_per_fold.png", dpi=120); plt.close(fig)

if have_main or have_cv:
    src = (MB if have_cv and any(CVB[s] for s in have_cv) else MM)
    fig, axs = plt.subplots(2, 3, figsize=(17, 8))
    for r, c in enumerate(FG):
        for ax, (key, ttl) in zip(axs[r][:1], [(f"{c}_bf2", f"{c}: boundary F-score, τ = 2 px")]):
            for i, s in enumerate(src):
                v = [m[key] for m in src[s].values()]
                ax.bar(i, np.mean(v), color=COL[s], yerr=np.std(v, ddof=1) if len(v) > 1 else None, capsize=3)
            ax.set_xticks(range(len(src))); ax.set_xticklabels([LAB[s].split(" (")[0] for s in src], rotation=60, ha="right", fontsize=7)
            ax.set_title(ttl); ax.grid(axis="y", alpha=.3)
        for ax, q, ttl in ((axs[r][1], "rec", f"{c}: recall by object size"), (axs[r][2], "prec", f"{c}: precision by predicted object size")):
            for s in src:
                v = np.array([[m[f"{c}_{q}{k}"] for k in range(3)] for m in src[s].values()])
                ax.plot(range(3), v.mean(0), marker="o", color=COL[s], label=LAB[s])
            ax.set_xticks(range(3)); ax.set_xticklabels(BIN_NAMES); ax.set_title(ttl); ax.grid(alpha=.3)
    axs[0][1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "boundary_and_size_metrics.png", dpi=110); plt.close(fig)

(OUT / "cv_report.md").write_text("\n".join(rep), encoding="utf-8")
print("report written:", OUT / "cv_report.md")
