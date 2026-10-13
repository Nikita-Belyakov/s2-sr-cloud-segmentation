"""ERQA of EDiffSR / ESRGAN / bicubic against the real Venus 5 m reference on Sen2Venus test scenes (erqa_s2v/selection.csv: 12 scenes x 40 patches of valid.csv).

Reference = Venus 5 m (256x256, 8 bands, /10000, clip [0, 0.5], channel order [B5,B6,B7,B8A,B2,B3,B4,B8] as in Sen2VenusDataset).
Compared images (all 256x256, 5 m): bicubic x2 of the Sentinel-2 input, EDiffSR x2, ESRGAN x4 averaged to x2.
ERQA = F1 of the matching of Canny edges (100/200, tolerance 1 px). Blue = Venus edge that the result misses, red = edge that Venus does not have
(artifact / invented structure), white = edge in both. Composites (BGR for OpenCV): rgb = B4,B3,B2 (gain 3), cir = B8,B4,B3 (gain 2), re = B8A,B6,B5 (gain 2).
The main score is computed without ERQA's global shift compensation (the 1 px edge tolerance stays); the column erqa_v10_gc uses the standard global compensation (+-3 px).
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import cv2  # noqa: E402
import erqa  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr, ttest_rel  # noqa: E402

D = WORK_DIR
OUT = D / "erqa_s2v"
S2V = S2V_NPZ
METHODS = ["bicubic", "ediffsr", "esrgan"]
LAB = {"bicubic": "Bicubic", "ediffsr": "EDiffSR", "esrgan": "ESRGAN"}
COMP = {"rgb": ([4, 5, 6], 3.0), "cir": ([5, 6, 7], 2.0), "re": ([0, 1, 3], 2.0)}  # BGR indices into [B5,B6,B7,B8A,B2,B3,B4,B8], gain
COL = {"bicubic": "#7f7f7f", "ediffsr": "#d62728", "esrgan": "#2ca02c"}
M10 = erqa.ERQA(global_compensation=False, version="1.0")
M11 = erqa.ERQA(global_compensation=False, version="1.1")
MGC = erqa.ERQA(global_compensation=True, version="1.0")
K3 = np.ones((3, 3), np.uint8)
sel = pd.read_csv(OUT / "selection.csv")


def to_u8(x, comp="rgb", gain=None):
    idx, g = COMP[comp]
    a = np.stack([x[i] for i in idx], -1) * (g if gain is None else gain)
    return np.ascontiguousarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def f1_safe(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def parts(img, gt, metric=M10):
    edge = metric.edge_detector(img) // 255
    gedge = metric.edge_detector(gt) // 255
    tp, fn = metric.match_edges(edge, gedge)
    ntp = int(tp.sum())
    return f1_safe(ntp, int(edge.sum()) - ntp, int(fn.sum())), edge, gedge, tp, fn


def vis_of(im, gt, thick=False):
    f1, edge, gedge, tp, fn = parts(im, gt)
    tp, fn = tp.astype(np.uint8), fn.astype(np.uint8)
    fp = (edge.astype(np.uint8) - tp).clip(0, 1)
    if thick:
        tp, fn, fp = (cv2.dilate(x, K3) for x in (tp, fn, fp))
    out = np.zeros(tp.shape + (3,), np.float32)
    out[fn > 0] = (0, 0, 1)
    out[fp > 0] = (1, 0, 0)
    out[tp > 0] = (1, 1, 1)
    return f1, out


def load(fn):
    a = np.load(S2V / fn)
    f1 = torch.clip(torch.from_numpy(a["sentinel_20m"]).float() / 10000, 0, 0.5)
    f2 = torch.clip(torch.from_numpy(a["sentinel_10m"]).float() / 10000, 0, 0.5)
    f1 = F.interpolate(f1[None], size=(128, 128), mode="bicubic", align_corners=False)[0]
    feats = torch.cat([f1, f2], 0)
    gt = np.clip(np.concatenate([a["venus_5m_20m"], a["venus_5m_10m"]], 0).astype(np.float32) / 10000, 0, 0.5)
    res = {"bicubic": F.interpolate(feats[None], scale_factor=2, mode="bicubic", align_corners=False)[0].numpy()}
    for m in ("ediffsr", "esrgan"):
        res[m] = np.load(OUT / m / (fn[:-4] + ".npy"))
    return gt, {m: np.clip(v, 0, 0.5) for m, v in res.items()}


rows = []
sens = {}
for n, r in enumerate(sel.itertuples()):
    gt, res = load(r.filename)
    for m in METHODS:
        diff = res[m] - gt
        mse = float((diff ** 2).mean())
        cos = (res[m] * gt).sum(0) / (np.linalg.norm(res[m], axis=0) * np.linalg.norm(gt, axis=0) + 1e-12)
        extra = {"psnr_db": 10 * np.log10(0.25 / max(mse, 1e-12)), "rmse": mse ** 0.5, "sam_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())}
        for comp, (_, gain) in COMP.items():
            g8, i8 = to_u8(gt, comp), to_u8(res[m], comp)
            f10, edge, gedge, tp, fn = parts(i8, g8)
            f11 = parts(i8, g8, M11)[0]
            ntp, nfp, nfn = int(tp.sum()), int(edge.sum() - tp.sum()), int(fn.sum())
            row = {"patch": r.filename[:-4], "scene": r.scene, "site": r.site, "method": m, "composite": comp, "erqa_v10": f10, "erqa_v11": f11,
                   "precision": ntp / max(ntp + nfp, 1), "recall": ntp / max(ntp + nfn, 1), "ref_edge_px": int(gedge.sum()), "compared_edge_px": int(edge.sum()),
                   "valid": int(gedge.sum() >= 0.001 * gedge.size)}
            if comp == "rgb":
                row.update(extra)
                ig, gg = MGC._global_compensation(i8, g8)
                row["erqa_v10_gc"] = parts(ig, gg)[0]
                for g in (2.0, 4.0):
                    sens.setdefault((m, g), []).append(parts(to_u8(res[m], comp, g), to_u8(gt, comp, g))[0])
                sens.setdefault((m, 3.0), []).append(f10)
            rows.append(row)
    if n % 80 == 0:
        print(n, len(sel), flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "erqa_s2v_patch_scores.csv", index=False)
rgb = df[(df.composite == "rgb") & (df.valid == 1)]
n_all, n_valid = int(((df.composite == "rgb") & (df.method == "bicubic")).sum()), int(((df.composite == "rgb") & (df.method == "bicubic") & (df.valid == 1)).sum())

# ---------------------------------------------------------------- summary tables
L = ["# ERQA against the real Venus 5 m reference, Sen2Venus test scenes", "",
     f"{sel.scene.nunique()} scenes from {sel.site.nunique()} sites, {len(sel)} patches of `valid.csv` (patches the models did not see in training; the Sen2Venus split is by patch, "
     "so neighbouring patches of the same scene were in the training set). Reference = Venus 5 m, compared images at 5 m: bicubic x2 of the Sentinel-2 input, EDiffSR x2, ESRGAN x4 averaged to x2. "
     "ERQA = F1 of the matching of Canny edges (1 = identical edge structure). Precision = share of the result's edge pixels that Venus also has (1 − precision = edges that are not in the reference: "
     "artifacts, invented structure); recall = share of Venus edges that the result reproduces (1 − recall = missing zones). Mean ± std over patches.", "",
     f"Degenerate patches (Venus has fewer than 0.1% edge pixels, true colour): {n_all - n_valid} of {n_all} are excluded from the means; they are flagged in `erqa_s2v_patch_scores.csv`.", "",
     "## All scenes", "", "| composite | method | ERQA v1.0 | ERQA v1.1 | precision | recall | result / reference edge pixels |", "|---|---|---|---|---|---|---|"]
for comp in COMP:
    for m in METHODS:
        s = df[(df.composite == comp) & (df.method == m) & (df.valid == 1)]
        L.append(f"| {comp} | {LAB[m]} | {s.erqa_v10.mean():.3f} ± {s.erqa_v10.std():.3f} | {s.erqa_v11.mean():.3f} ± {s.erqa_v11.std():.3f} | {s.precision.mean():.3f} | {s.recall.mean():.3f} | "
                 f"{s.compared_edge_px.sum() / max(s.ref_edge_px.sum(), 1):.2f} |")
L += ["", "### Pixel-wise fidelity of the same images (all 8 bands, patches as above)", "", "| method | PSNR, dB | RMSE | SAM, deg | ERQA v1.0 with global shift compensation |", "|---|---|---|---|---|"]
for m in METHODS:
    s = rgb[rgb.method == m]
    L.append(f"| {LAB[m]} | {s.psnr_db.mean():.2f} | {s.rmse.mean():.4f} | {s.sam_deg.mean():.2f} | {s.erqa_v10_gc.mean():.3f} |")
piv = rgb.pivot(index="patch", columns="method", values="erqa_v10")
L += ["", "### Paired comparison per patch (true colour, ERQA v1.0)", "", "| comparison | mean difference | patches where the first is better | paired t-test p |", "|---|---|---|---|"]
for a_, b_ in (("ediffsr", "bicubic"), ("esrgan", "bicubic"), ("ediffsr", "esrgan")):
    d = (piv[a_] - piv[b_]).dropna()
    L.append(f"| {LAB[a_]} − {LAB[b_]} | {d.mean():+.4f} ± {d.std():.4f} | {100 * (d > 0).mean():.0f}% ({int((d > 0).sum())}/{len(d)}) | {ttest_rel(piv[a_].loc[d.index], piv[b_].loc[d.index]).pvalue:.3g} |")
L += ["", "## By scene (true colour, ERQA v1.0, mean over valid patches; in brackets: edges missed / edges not in the reference = 1−recall / 1−precision)", "",
      "| site | scene | patches | Bicubic | EDiffSR | ESRGAN | EDiffSR − Bicubic | ESRGAN − Bicubic |", "|---|---|---|---|---|---|---|---|"]
for sc, g in rgb.groupby("scene", sort=False):
    pv = g.pivot(index="patch", columns="method", values="erqa_v10")
    cells = []
    for m in METHODS:
        s = g[g.method == m]
        cells.append(f"{s.erqa_v10.mean():.3f} ({1 - s.recall.mean():.2f} / {1 - s.precision.mean():.2f})")
    L.append(f"| {g.site.iloc[0]} | {sc} | {len(pv)} | " + " | ".join(cells) + f" | {(pv.ediffsr - pv.bicubic).mean():+.3f} | {(pv.esrgan - pv.bicubic).mean():+.3f} |")
L += ["", "## Sensitivity to the brightness stretch (true colour, all patches including degenerate ones)", "", "| method | gain 2 | gain 3 (used) | gain 4 |", "|---|---|---|---|"]
for m in METHODS:
    L.append(f"| {LAB[m]} | " + " | ".join(f"{np.mean(sens[(m, g)]):.3f}" for g in (2.0, 3.0, 4.0)) + " |")
dens = rgb[rgb.method == "bicubic"].set_index("patch").ref_edge_px
L += ["", "## ERQA against the edge richness of the patch (Spearman correlation with the number of Venus edge pixels, true colour)", "", "| method | correlation |", "|---|---|"]
for m in METHODS:
    e = rgb[rgb.method == m].set_index("patch").erqa_v10.loc[dens.index]
    L.append(f"| {LAB[m]} | {spearmanr(e, dens).correlation:+.2f} |")
(OUT / "erqa_s2v_summary.md").write_text("\n".join(L), encoding="utf-8")

# ---------------------------------------------------------------- overview figure
fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
axs[0].boxplot([rgb[rgb.method == m].erqa_v10 for m in METHODS], labels=[LAB[m] for m in METHODS], showfliers=False)
axs[0].set_title("ERQA per patch against Venus 5 m (true colour)"); axs[0].set_ylabel("ERQA v1.0"); axs[0].grid(axis="y", alpha=.3)
scenes = list(rgb.scene.unique())
for k, m in enumerate(METHODS):
    v = [rgb[(rgb.scene == sc) & (rgb.method == m)].erqa_v10.mean() for sc in scenes]
    axs[1].bar(np.arange(len(scenes)) + (k - 1) * 0.27, v, 0.26, color=COL[m], label=LAB[m])
axs[1].set_xticks(range(len(scenes))); axs[1].set_xticklabels([rgb[rgb.scene == sc].site.iloc[0] for sc in scenes], rotation=60, ha="right", fontsize=7)
axs[1].set_title("ERQA by scene"); axs[1].legend(fontsize=8); axs[1].grid(axis="y", alpha=.3)
for m in METHODS:
    s = rgb[rgb.method == m]
    axs[2].scatter(s.recall.mean(), s.precision.mean(), s=120, color=COL[m], label=LAB[m])
    for sc in scenes:
        g = s[s.scene == sc]
        axs[2].scatter(g.recall.mean(), g.precision.mean(), s=18, color=COL[m], alpha=.45)
axs[2].set_xlabel("recall: Venus edges reproduced"); axs[2].set_ylabel("precision: result edges that Venus has"); axs[2].legend(fontsize=8); axs[2].grid(alpha=.3)
axs[2].set_title("edge precision against recall (big: mean, small: per scene)")
fig.tight_layout(); fig.savefig(OUT / "erqa_s2v_overview.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- patch visualisations
def show(x, gain=3.0):
    return np.clip(np.stack([x[6], x[5], x[4]], -1) * gain, 0, 1)  # B4,B3,B2 -> RGB


def clean(axs):
    for a in np.ravel(axs):
        a.set_xticks([]); a.set_yticks([])


rep, worst = [], []
for sc in scenes:
    g = rgb[(rgb.scene == sc) & (rgb.method == "ediffsr")].sort_values("erqa_v10")
    rep.append((sc, g.iloc[len(g) // 2].patch))
allp = rgb.groupby("patch").erqa_v10.mean().sort_values()
worst = [(rgb[rgb.patch == p].scene.iloc[0], p) for p in allp.index[:6]]


def maps_for(patch):
    gt, res = load(patch + ".npz")
    return gt, res, {m: vis_of(to_u8(res[m]), to_u8(gt), thick=True) for m in METHODS}, {m: vis_of(to_u8(res[m]), to_u8(gt))[1] for m in METHODS}


def gallery(items, name, title):
    fig, axs = plt.subplots(len(items), 4, figsize=(16, 4.25 * len(items)))
    for i, (sc, p) in enumerate(items):
        gt, res, th, _ = maps_for(p)
        axs[i, 0].imshow(show(gt), interpolation="nearest"); axs[i, 0].set_title(f"{sc}\n{p.split('_')[-1]}  (Venus 5 m)", fontsize=8.5)
        for k, m in enumerate(METHODS):
            axs[i, k + 1].imshow(th[m][1], interpolation="nearest"); axs[i, k + 1].set_title(f"{LAB[m]}  ERQA {th[m][0]:.3f}", fontsize=10)
    clean(axs)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=3.0); fig.savefig(OUT / name, dpi=80); plt.close(fig)


PO = D / "paper_figs"; PO.mkdir(exist_ok=True)
sel = [rep[i] for i in (0, 2, 4, 5)]
fig, axs = plt.subplots(len(sel), 4, figsize=(5.6, 1.4 * len(sel) + 0.35))
for i, (sc, p) in enumerate(sel):
    gt, res, th, _ = maps_for(p)
    axs[i, 0].imshow(show(gt), interpolation="nearest")
    best = max(METHODS, key=lambda mm: round(th[mm][0], 2))
    for k, m in enumerate(METHODS):
        axs[i, k + 1].imshow(th[m][1], interpolation="nearest")
        axs[i, k + 1].text(0.03, 0.04, f"{th[m][0]:.2f}", transform=axs[i, k + 1].transAxes, fontsize=9,
                           fontweight="bold" if m == best else "normal",
                           bbox=dict(facecolor="w", alpha=.85, pad=1.5, edgecolor="none"))
for k, h in enumerate(["VENµS 5 m", "Bicubic", "EDiffSR", "ESRGAN"]):
    axs[0, k].set_title(h, fontsize=9)
clean(axs)
fig.tight_layout(h_pad=0.4, w_pad=0.3); fig.savefig(PO / "erqa_s2v_gallery.png", dpi=250); plt.close(fig)
import sys; sys.exit(0)

for sc, p in rep + worst[:3]:
    gt, res, th, ex = maps_for(p)
    tag = f"{p}"
    fig, axs = plt.subplots(4, 2, figsize=(9, 18))
    axs[0, 0].imshow(show(gt), interpolation="nearest"); axs[0, 0].set_title("Venus 5 m reference", fontsize=9)
    gedge = np.zeros(gt.shape[1:] + (3,), np.float32)
    gedge[cv2.dilate((M10.edge_detector(to_u8(gt)) // 255).astype(np.uint8), K3) > 0] = 1
    axs[0, 1].imshow(gedge, interpolation="nearest"); axs[0, 1].set_title("Venus edges (Canny 100/200)", fontsize=9)
    for k, m in enumerate(METHODS):
        axs[k + 1, 0].imshow(show(res[m]), interpolation="nearest"); axs[k + 1, 0].set_title(f"{LAB[m]}", fontsize=9)
        axs[k + 1, 1].imshow(th[m][1], interpolation="nearest"); axs[k + 1, 1].set_title(f"{LAB[m]}: ERQA {th[m][0]:.3f} (blue missing, red false, white correct)", fontsize=8.5)
    clean(axs)
    fig.suptitle(f"{sc}  {p}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.2); fig.savefig(OUT / f"s2v_{tag}_full.png", dpi=100); plt.close(fig)
    # zoom: window with most edge differences between the three results and Venus
    diff = sum(((ex[m][..., 0] > .5) ^ (ex[m][..., 2] > .5)).astype(np.float32) for m in METHODS)
    win = 64
    box = cv2.boxFilter(diff, -1, (win, win), normalize=False, borderType=cv2.BORDER_CONSTANT)
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    y0, x0 = int(np.clip(cy - win // 2, 0, 256 - win)), int(np.clip(cx - win // 2, 0, 256 - win))
    sl = (slice(y0, y0 + win), slice(x0, x0 + win))
    fig, axs = plt.subplots(3, 3, figsize=(10, 10.4))
    for k, m in enumerate(METHODS):
        axs[k, 0].imshow(show(gt)[sl], interpolation="nearest"); axs[k, 0].set_title("Venus 5 m" if k == 0 else "", fontsize=9)
        axs[k, 1].imshow(show(res[m])[sl], interpolation="nearest"); axs[k, 1].set_title(LAB[m], fontsize=9)
        axs[k, 2].imshow(ex[m][sl], interpolation="nearest"); axs[k, 2].set_title(f"{LAB[m]} ERQA map (blue missing, red false)", fontsize=9)
    clean(axs)
    fig.suptitle(f"{p}: zoom {win}x{win} px at 5 m (x {x0}-{x0 + win}, y {y0}-{y0 + win})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.4); fig.savefig(OUT / f"s2v_{tag}_zoom.png", dpi=100); plt.close(fig)
print("done")
