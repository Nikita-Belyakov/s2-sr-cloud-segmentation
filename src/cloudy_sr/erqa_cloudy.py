"""ERQA (Edge Restoration Quality Assessment, MSU Video Group) of the SR results at 5 m against the original 10 m Sentinel-2 patches,
test part of the cloudy patches.

ERQA matches the Canny edges (thresholds 100/200) of the compared image to the edges of the reference (tolerance 1 px) and reports the F1.
Visualisation: white = edge present in both, blue = edge of the reference that is missing in the compared image (lost / smoothed detail,
'missing zones'), red = edge of the compared image that the reference does not have (added detail, artifacts, ringing, noise).

There is no 5 m ground truth for these scenes, so the reference is the ORIGINAL 10 m patch put on the 512 grid by bicubic x2 (the very image
EDiffSR receives as its condition). ERQA then shows where the SR output differs structurally from the interpolated source: blue = the SR lost
an edge of the source, red = the SR created an edge that is not in the source (real sub-pixel detail and hallucinated structure cannot be told apart
without a 5 m reference).

Protocols (3-channel uint8 composites of the 8 bands, 1075 patches of cloudy_split/test, methods EDiffSR x2 and ESRGAN x4 averaged to x2):
  sr5m     reference = original 10 m patch, bicubic x2 -> 512x512;     compared = SR result 512x512          (main)
  cons10   reference = original 10 m patch 256x256;                     compared = SR result averaged 2x2 back to 256x256
           (the same edge comparison on the real 10 m data: which source edges the SR keeps, loses or adds at the scale where the truth is known)
Composites (BGR order for OpenCV): rgb = B4,B3,B2 (gain 3), cir = B8,B4,B3 (gain 2), re = B8A,B6,B5 (gain 2); gain maps reflectance to uint8.
The images are aligned by construction, so the global shift compensation of ERQA is off; the 1 px edge shift compensation is on.
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile as tff
import torch
import torch.nn.functional as F
from PIL import Image

import cv2  # noqa: E402
import erqa  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

D = WORK_DIR
OUT = D / "erqa_sr5m"
OUT.mkdir(exist_ok=True)
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
TIF_IDX = [4, 5, 6, 8, 1, 2, 3, 7]  # model order: B5,B6,B7,B8A,B2,B3,B4,B8
COMP = {"rgb": ([4, 5, 6], 3.0), "cir": ([5, 6, 7], 2.0), "re": ([0, 1, 3], 2.0)}  # BGR channel indices (model order) and gain
CN = {0: "clear", 1: "cloud", 2: "shadow"}  # raw png values
METHODS = ["ediffsr", "esrgan"]
PROTO = ["sr5m", "cons10"]
LAB = {"ediffsr": "EDiffSR", "esrgan": "ESRGAN"}
SRC = {"ediffsr": D / "cloudy_split_sr/ediffsr_x2", "esrgan": D / "cloudy_split_sr/esrgan_x4_to512"}

M10 = erqa.ERQA(global_compensation=False, version="1.0")
M11 = erqa.ERQA(global_compensation=False, version="1.1")


def to_u8(x, comp, gain=None):
    idx, g = COMP[comp]
    a = np.stack([x[i] for i in idx], -1) * (g if gain is None else gain)
    return np.ascontiguousarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def f1_safe(tp, fp, fn):
    """F1 of the edge matching; 0 when nothing matches. (erqa.ERQA.f1_matches returns nan = 0/0 when both images have edges but none coincide.)"""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def parts(img, gt, metric=M10):
    edge = metric.edge_detector(img) // 255
    gedge = metric.edge_detector(gt) // 255
    tp, fn = metric.match_edges(edge, gedge)
    ntp = int(tp.sum())
    return f1_safe(ntp, int(edge.sum()) - ntp, int(fn.sum())), edge, gedge, tp, fn


def load(stem):
    raw = torch.from_numpy(np.clip(tff.imread(D / "cloudy_split" / SPLIT / f"{stem}.tif")[TIF_IDX].astype(np.float32) / 10000, 0, 0.5))
    m256 = np.array(Image.open(D / "cloudy_split" / SPLIT / f"{stem}.png"))
    m512 = np.repeat(np.repeat(m256, 2, 0), 2, 1)
    ref512 = F.interpolate(raw[None], scale_factor=2, mode="bicubic", align_corners=False)[0].numpy()
    sr = {m: tff.imread(SRC[m] / SPLIT / f"{stem}.tif").astype(np.float32) / 10000 for m in METHODS}
    cons = {m: F.avg_pool2d(torch.from_numpy(sr[m])[None], 2)[0].numpy() for m in METHODS}
    return raw.numpy(), ref512, m256, m512, sr, cons


stems = sorted(f.stem for f in (D / "cloudy_split" / SPLIT).glob("*.tif"))
rows, pool, sens = [], {}, {}
for n, st in enumerate(stems):
    raw, ref512, m256, m512, sr, cons = load(st)
    cf, sf = float((m256 == 1).mean()), float((m256 == 2).mean())
    for proto in PROTO:
        ref, dct, mask = (ref512, sr, m512) if proto == "sr5m" else (raw, cons, m256)
        for m in METHODS:
            for comp in COMP:
                gt, im = to_u8(ref, comp), to_u8(dct[m], comp)
                f10, edge, gedge, tp, fn = parts(im, gt, M10)
                f11 = parts(im, gt, M11)[0]
                ntp, nfp, nfn = int(tp.sum()), int(edge.sum() - tp.sum()), int(fn.sum())
                rows.append({"patch": st, "protocol": proto, "method": m, "composite": comp, "erqa_v10": f10, "erqa_v11": f11,
                             "precision": ntp / max(ntp + nfp, 1), "recall": ntp / max(ntp + nfn, 1), "ref_edge_px": int(gedge.sum()),
                             "compared_edge_px": int(edge.sum()), "cloud_frac": cf, "shadow_frac": sf,
                             "valid": int(gedge.sum() >= 0.001 * gedge.size)})  # patches whose reference has almost no edges (flat / saturated) are degenerate
                if comp == "rgb":
                    tpb, fpb, fnb = tp.astype(bool), (edge - tp).astype(bool), fn.astype(bool)
                    for c in CN:
                        s = mask == c
                        pool.setdefault((proto, m, c), np.zeros(3))[:] += [int((tpb & s).sum()), int((fpb & s).sum()), int((fnb & s).sum())]
                    if proto == "sr5m":
                        for g in (2.0, 3.0, 4.0):
                            sens.setdefault((m, g), []).append(parts(to_u8(dct[m], comp, g), to_u8(ref, comp, g))[0])
    if n % 200 == 0:
        print(n, len(stems), flush=True)

with open(OUT / "erqa_patch_scores.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def sel(proto, m, comp="rgb", valid=True):
    return [r for r in rows if r["protocol"] == proto and r["method"] == m and r["composite"] == comp and (r["valid"] or not valid)]


n_all = len(sel("sr5m", "ediffsr", "rgb", valid=False))
n_val = len(sel("sr5m", "ediffsr", "rgb"))


# ---------------------------------------------------------------- summary
L = [f"# ERQA of the SR results (5 m) against the original 10 m patches, {SPLIT} part ({len(stems)} patches)", "",
     "ERQA = F1 of the matching of Canny edges of the compared image to the edges of the reference (1 = the edges coincide, 0 = nothing in common). "
     "`v1.0` is the standard version, `v1.1` additionally penalises edges wider than the reference. "
     "Precision = share of the SR edge pixels that exist in the reference (1 − precision = edges added by the SR: new detail, artifacts); "
     "recall = share of the reference edges that the SR keeps (1 − recall = missing zones, smoothed detail). Mean ± std over patches.", "",
     "**Reference.** No 5 m ground truth exists, so for the main protocol (`sr5m`) the reference is the original 10 m patch interpolated to the 512 grid (bicubic ×2). "
     "ERQA therefore measures how the SR output differs structurally from its own interpolated source; it does not prove that the added edges are wrong or right.", "",
     f"**Degenerate patches.** ERQA is undefined when the reference has almost no edges (flat or saturated cloud patches). Patches whose reference has fewer than 0.1% edge pixels "
     f"(true-colour composite, main protocol: {n_all - n_val} of {n_all}) are excluded from the means below (the rows in `erqa_patch_scores.csv` carry a `valid` flag; "
     f"the per-class tables pool all edge pixels). Where the matching finds no common edge at all the score is 0 (the `erqa` package returns nan there).", ""]
for proto, title in (("sr5m", "Main: SR result 5 m (512×512) against the original 10 m patch interpolated to 512×512"),
                     ("cons10", "Additional: SR result averaged back to 10 m (256×256) against the original 10 m patch")):
    L += [f"## {title}", "", "| composite | method | ERQA v1.0 | ERQA v1.1 | precision (SR edges that exist in the source) | recall (source edges kept) | SR / reference edge pixels |", "|---|---|---|---|---|---|---|"]
    for comp in COMP:
        for m in METHODS:
            s = sel(proto, m, comp)
            v = lambda k: np.array([r[k] for r in s])  # noqa: E731
            L.append(f"| {comp} | {LAB[m]} | {v('erqa_v10').mean():.3f} ± {v('erqa_v10').std():.3f} | {v('erqa_v11').mean():.3f} ± {v('erqa_v11').std():.3f} | "
                     f"{v('precision').mean():.3f} | {v('recall').mean():.3f} | {v('compared_edge_px').sum() / max(v('ref_edge_px').sum(), 1):.2f} |")
    L += ["", "### By class (true-colour composite; edge pixels pooled over all patches; the class is the label at the edge pixel)", "",
          "| class | method | F1 | precision | recall | reference edge px |", "|---|---|---|---|---|---|"]
    for c in CN:
        for m in METHODS:
            tp, fp, fn = pool[(proto, m, c)]
            p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            L.append(f"| {CN[c]} | {LAB[m]} | {2 * p * r / max(p + r, 1e-9):.3f} | {p:.3f} | {r:.3f} | {int(tp + fn):,} |")
    L.append("")
L += ["## Sensitivity to the brightness stretch (main protocol, true colour)", "", "| method | gain 2 | gain 3 (used) | gain 4 |", "|---|---|---|---|"]
for m in METHODS:
    L.append(f"| {LAB[m]} | " + " | ".join(f"{np.mean(sens[(m, g)]):.3f}" for g in (2.0, 3.0, 4.0)) + " |")
cfr = np.array([r["cloud_frac"] for r in sel("sr5m", "ediffsr")])
shr = np.array([r["shadow_frac"] for r in sel("sr5m", "ediffsr")])
L += ["", "## ERQA against the cloud and shadow cover of the patch (main protocol, true colour, Spearman correlation)", "", "| method | with cloud fraction | with shadow fraction |", "|---|---|---|"]
for m in METHODS:
    e = np.array([r["erqa_v10"] for r in sel("sr5m", m)])
    L.append(f"| {LAB[m]} | {spearmanr(e, cfr).correlation:+.2f} | {spearmanr(e, shr).correlation:+.2f} |")
(OUT / "erqa_summary.md").write_text("\n".join(L), encoding="utf-8")

# ---------------------------------------------------------------- overview figure
col = {"ediffsr": "#d62728", "esrgan": "#2ca02c"}
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
axs[0].boxplot([[r["erqa_v10"] for r in sel("sr5m", m)] for m in METHODS], labels=[LAB[m] for m in METHODS], showfliers=False)
axs[0].set_title("ERQA per patch (SR 5 m vs original 10 m, true colour)"); axs[0].set_ylabel("ERQA v1.0"); axs[0].grid(axis="y", alpha=.3)
for m in METHODS:
    s = sel("sr5m", m)
    axs[1].scatter([r["cloud_frac"] for r in s], [r["erqa_v10"] for r in s], s=6, alpha=.4, color=col[m], label=LAB[m])
axs[1].set_xlabel("cloud fraction of the patch"); axs[1].set_ylabel("ERQA v1.0"); axs[1].legend(markerscale=3); axs[1].grid(alpha=.3); axs[1].set_title("ERQA against cloud cover")
xs = np.arange(3)
for k, m in enumerate(METHODS):
    p = [pool[("sr5m", m, c)] for c in CN]
    axs[2].bar(xs + (k - 0.5) * 0.36, [t[0] / max(t[0] + t[2], 1) for t in p], 0.17, color=col[m], label=f"{LAB[m]} recall (edges kept)")
    axs[2].bar(xs + (k - 0.5) * 0.36 + 0.18, [t[0] / max(t[0] + t[1], 1) for t in p], 0.17, color=col[m], alpha=.45, hatch="//", label=f"{LAB[m]} precision (edges that exist in the source)")
axs[2].set_xticks(xs); axs[2].set_xticklabels([CN[c] for c in CN]); axs[2].legend(fontsize=6); axs[2].grid(axis="y", alpha=.3); axs[2].set_title("Edges kept (solid) and edges that exist in the source (hatched), by class")
fig.tight_layout(); fig.savefig(OUT / "erqa_overview.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- patch visualisations
tab = {(r["patch"], r["protocol"], r["method"]): r for r in rows if r["composite"] == "rgb"}
pick = []


def add(why, key, flt=lambda r: True):
    for st in sorted([s for s in stems if flt(tab[(s, "sr5m", "ediffsr")]) and tab[(s, "sr5m", "ediffsr")]["valid"] and tab[(s, "sr5m", "esrgan")]["valid"]], key=key, reverse=True):
        if st not in [p[0] for p in pick]:
            pick.append((st, why))
            return


r_ = lambda s: tab[(s, "sr5m", "ediffsr")]  # noqa: E731
add("cloud and shadow together", lambda s: min(r_(s)["cloud_frac"], 3 * r_(s)["shadow_frac"]))
add("most cloud", lambda s: r_(s)["cloud_frac"])
add("most shadow", lambda s: r_(s)["shadow_frac"])
add("richest in edges (clear)", lambda s: r_(s)["ref_edge_px"], flt=lambda r: r["cloud_frac"] + r["shadow_frac"] < 0.01)
add("lowest ERQA of EDiffSR", lambda s: -tab[(s, "sr5m", "ediffsr")]["erqa_v10"])
add("lowest ERQA of ESRGAN", lambda s: -tab[(s, "sr5m", "esrgan")]["erqa_v10"])
okst = [s for s in stems if tab[(s, "sr5m", "ediffsr")]["valid"]]
med = sorted(okst, key=lambda s: tab[(s, "sr5m", "ediffsr")]["erqa_v10"])[len(okst) // 2]
if med not in [p[0] for p in pick]:
    pick.append((med, "median ERQA of EDiffSR"))
with open(OUT / "visualised_patches.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["patch", "why"] + [f"erqa_{m}" for m in METHODS])
    for st, why in pick:
        w.writerow([st, why] + [f"{tab[(st, 'sr5m', m)]['erqa_v10']:.3f}" for m in METHODS])


def show(x, gain=3.0):
    return np.clip(np.stack([x[6], x[5], x[4]], -1) * gain, 0, 1)  # model order B4,B3,B2 -> RGB


def vis_of(im, gt):
    f1, edge, gedge, tp, fn = parts(im, gt)
    # BGR -> RGB: blue = missing, red = added, white = kept; float, otherwise imshow reads the 0/1 integers as 0..255 and shows black
    return f1, M10.generate_visualization(edge, tp, fn)[..., ::-1].astype(np.float32)
# NOTE: the final figures are drawn by erqa_viz.py (thicker edges for the gallery), this script writes the scores and the tables


def contours(ax, mask):
    k = 2 if mask.shape[0] == 512 else 1
    for val, c in ((1, "#ffd400"), (2, "#ff00ff")):
        if (mask == val).any():
            ax.contour((mask == val).astype(float), levels=[0.5], colors=c, linewidths=0.9)


def clean(axs):
    for a in np.ravel(axs):
        a.set_xticks([]); a.set_yticks([])


gallery = []
for st, why in pick:
    raw, ref512, m256, m512, sr, cons = load(st)
    gt = to_u8(ref512, "rgb")
    res = {m: vis_of(to_u8(sr[m], "rgb"), gt) for m in METHODS}
    gallery.append((st, why, ref512, m512, res))
    # whole patch at 5 m
    fig, axs = plt.subplots(3, 2, figsize=(11, 16.6))
    axs[0, 0].imshow(show(ref512), interpolation="nearest"); contours(axs[0, 0], m512)
    axs[0, 0].set_title("original 10 m on the 512 grid (bicubic x2), yellow: cloud, magenta: shadow", fontsize=9)
    axs[0, 1].imshow(cv2.Canny(gt, 100, 200), cmap="gray", interpolation="nearest"); axs[0, 1].set_title("reference edges (Canny 100/200)", fontsize=9)
    for k, m in enumerate(METHODS):
        axs[k + 1, 0].imshow(show(sr[m]), interpolation="nearest"); axs[k + 1, 0].set_title(f"{LAB[m]} result, 5 m", fontsize=9)
        axs[k + 1, 1].imshow(res[m][1], interpolation="nearest")
        axs[k + 1, 1].set_title(f"{LAB[m]}: ERQA {res[m][0]:.3f}   blue = source edge lost, red = edge added by SR, white = kept", fontsize=8)
    clean(axs)
    fig.suptitle(f"{st}: {why}", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"sr5m_{st}_full.png", dpi=100); plt.close(fig)
    # zoom into the window with the most edge differences
    diff = sum(((res[m][1][..., 0] > .5) ^ (res[m][1][..., 2] > .5)).astype(np.float32) for m in METHODS)  # red-only or blue-only pixels
    win = 96
    box = cv2.boxFilter(diff, -1, (win, win), normalize=False, borderType=cv2.BORDER_CONSTANT)
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    y0, x0 = int(np.clip(cy - win // 2, 0, 512 - win)), int(np.clip(cx - win // 2, 0, 512 - win))
    sl = (slice(y0, y0 + win), slice(x0, x0 + win))
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.6))
    for k, m in enumerate(METHODS):
        axs[k, 0].imshow(show(ref512)[sl], interpolation="nearest"); axs[k, 0].set_title("original 10 m (bicubic x2)" if k == 0 else "", fontsize=9)
        axs[k, 1].imshow(show(sr[m])[sl], interpolation="nearest"); axs[k, 1].set_title(f"{LAB[m]} 5 m", fontsize=9)
        axs[k, 2].imshow(res[m][1][sl], interpolation="nearest"); axs[k, 2].set_title(f"{LAB[m]} ERQA map", fontsize=9)
    clean(axs)
    fig.suptitle(f"{st}: zoom {win}×{win} px of the 5 m grid (x {x0}-{x0 + win}, y {y0}-{y0 + win})", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"sr5m_{st}_zoom.png", dpi=100); plt.close(fig)
    # consistency at 10 m
    cres = {m: vis_of(to_u8(cons[m], "rgb"), to_u8(raw, "rgb")) for m in METHODS}
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 8.6))
    for k, m in enumerate(METHODS):
        axs[k, 0].imshow(show(cons[m]), interpolation="nearest"); axs[k, 0].set_title(f"{LAB[m]} 5 m averaged back to 10 m", fontsize=9)
        axs[k, 1].imshow(cres[m][1], interpolation="nearest"); axs[k, 1].set_title(f"ERQA {cres[m][0]:.3f}", fontsize=9)
        axs[k, 2].imshow(show(raw), interpolation="nearest"); contours(axs[k, 2], m256); axs[k, 2].set_title("original 10 m (reference)", fontsize=9)
    clean(axs)
    fig.suptitle(f"{st}: edges at 10 m (blue = source edge lost by the SR, red = edge added by the SR)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"cons10_{st}.png", dpi=100); plt.close(fig)

fig, axs = plt.subplots(len(gallery), 3, figsize=(15, 5.2 * len(gallery)))
for i, (st, why, ref512, m512, res) in enumerate(gallery):
    axs[i, 0].imshow(show(ref512), interpolation="nearest"); contours(axs[i, 0], m512); axs[i, 0].set_title(f"{st}\n{why}", fontsize=8)
    for k, m in enumerate(METHODS):
        axs[i, k + 1].imshow(res[m][1], interpolation="nearest"); axs[i, k + 1].set_title(f"{LAB[m]}  ERQA {res[m][0]:.3f}", fontsize=9)
clean(axs)
fig.suptitle("ERQA maps, SR 5 m vs original 10 m: blue = source edge lost, red = edge added by SR, white = kept; yellow / magenta = cloud / shadow", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "erqa_gallery.png", dpi=80); plt.close(fig)
print("done; selected:", [p[0] for p in pick])
