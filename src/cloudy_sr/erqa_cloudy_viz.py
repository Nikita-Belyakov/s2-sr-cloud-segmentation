"""ERQA maps for the selected test patches (the scores are already in erqa_sr5m/erqa_patch_scores.csv, the patches in visualised_patches.csv).

blue = edge of the source that the SR result lost (missing zone), red = edge created by the SR (added detail / artifact),
white = edge present in both, black = no edge. The reference is the original 10 m patch (bicubic x2 for the 5 m protocol).
Gallery and full-patch figures draw the edges 3 px thick for visibility, the zoom figures show the exact 1 px edges.
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

D = WORK_DIR
OUT = D / "erqa_sr5m"
SPLIT = "test"
TIF_IDX = [4, 5, 6, 8, 1, 2, 3, 7]
COMP = {"rgb": ([4, 5, 6], 3.0)}
METHODS = ["ediffsr", "esrgan"]
LAB = {"ediffsr": "EDiffSR", "esrgan": "ESRGAN"}
SRC = {"ediffsr": D / "cloudy_split_sr/ediffsr_x2", "esrgan": D / "cloudy_split_sr/esrgan_x4_to512"}
M10 = erqa.ERQA(global_compensation=False, version="1.0")
K3 = np.ones((3, 3), np.uint8)


def to_u8(x, comp="rgb", gain=None):
    idx, g = COMP[comp]
    a = np.stack([x[i] for i in idx], -1) * (g if gain is None else gain)
    return np.ascontiguousarray((np.clip(a, 0, 1) * 255).round().astype(np.uint8))


def f1_safe(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def parts(img, gt):
    edge = M10.edge_detector(img) // 255
    gedge = M10.edge_detector(gt) // 255
    tp, fn = M10.match_edges(edge, gedge)
    ntp = int(tp.sum())
    return f1_safe(ntp, int(edge.sum()) - ntp, int(fn.sum())), edge, gedge, tp, fn


def vis_of(im, gt, thick=False):
    f1, edge, gedge, tp, fn = parts(im, gt)
    tp, fn = tp.astype(np.uint8), fn.astype(np.uint8)
    fp = (edge.astype(np.uint8) - tp).clip(0, 1)
    if thick:
        tp, fn, fp = (cv2.dilate(x, K3) for x in (tp, fn, fp))
    out = np.zeros(tp.shape + (3,), np.float32)       # RGB
    out[fn > 0] = (0, 0, 1)                             # blue: missing
    out[fp > 0] = (1, 0, 0)                             # red: added by the SR
    out[tp > 0] = (1, 1, 1)                             # white: kept
    return f1, out


def load(stem):
    raw = torch.from_numpy(np.clip(tff.imread(D / "cloudy_split" / SPLIT / f"{stem}.tif")[TIF_IDX].astype(np.float32) / 10000, 0, 0.5))
    m256 = np.array(Image.open(D / "cloudy_split" / SPLIT / f"{stem}.png"))
    m512 = np.repeat(np.repeat(m256, 2, 0), 2, 1)
    ref512 = F.interpolate(raw[None], scale_factor=2, mode="bicubic", align_corners=False)[0].numpy()
    sr = {m: tff.imread(SRC[m] / SPLIT / f"{stem}.tif").astype(np.float32) / 10000 for m in METHODS}
    cons = {m: F.avg_pool2d(torch.from_numpy(sr[m])[None], 2)[0].numpy() for m in METHODS}
    return raw.numpy(), ref512, m256, m512, sr, cons


def show(x, gain=3.0):
    return np.clip(np.stack([x[6], x[5], x[4]], -1) * gain, 0, 1)  # model order B4,B3,B2 -> RGB


def contours(ax, mask):
    for val, c in ((1, "#ffd400"), (2, "#ff00ff")):
        if (mask == val).any():
            ax.contour((mask == val).astype(float), levels=[0.5], colors=c, linewidths=1.0)


def clean(axs):
    for a in np.ravel(axs):
        a.set_xticks([]); a.set_yticks([])


pick = [(r["patch"], r["why"]) for r in csv.DictReader(open(OUT / "visualised_patches.csv"))]
gallery = []
for st, why in pick:
    raw, ref512, m256, m512, sr, cons = load(st)
    gt = to_u8(ref512)
    res = {m: vis_of(to_u8(sr[m]), gt, thick=True) for m in METHODS}
    exact = {m: vis_of(to_u8(sr[m]), gt)[1] for m in METHODS}
    gallery.append((st, why, ref512, m512, res))
    fig, axs = plt.subplots(3, 2, figsize=(11, 16.8))
    axs[0, 0].imshow(show(ref512), interpolation="nearest"); contours(axs[0, 0], m512)
    axs[0, 0].set_title("original 10 m on the 512 grid (bicubic x2); yellow: cloud, magenta: shadow", fontsize=9)
    ref_edges = np.zeros(gt.shape[:2] + (3,), np.float32)
    ref_edges[cv2.dilate((M10.edge_detector(gt) // 255).astype(np.uint8), K3) > 0] = 1
    axs[0, 1].imshow(ref_edges, interpolation="nearest"); axs[0, 1].set_title("reference edges (Canny 100/200)", fontsize=9)
    for k, m in enumerate(METHODS):
        axs[k + 1, 0].imshow(show(sr[m]), interpolation="nearest"); axs[k + 1, 0].set_title(f"{LAB[m]} result, 5 m", fontsize=9)
        axs[k + 1, 1].imshow(res[m][1], interpolation="nearest")
        axs[k + 1, 1].set_title(f"{LAB[m]}: ERQA {res[m][0]:.3f}   blue: source edge lost, red: edge added, white: kept", fontsize=8.5)
    clean(axs)
    fig.suptitle(f"{st}: {why}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.2); fig.savefig(OUT / f"sr5m_{st}_full.png", dpi=100); plt.close(fig)
    # zoom: the window of the 5 m grid with the most edge differences, exact 1 px edges
    diff = sum(((exact[m][..., 0] > .5) ^ (exact[m][..., 2] > .5)).astype(np.float32) for m in METHODS)
    win = 96
    box = cv2.boxFilter(diff, -1, (win, win), normalize=False, borderType=cv2.BORDER_CONSTANT)
    cy, cx = np.unravel_index(int(np.argmax(box)), box.shape)
    y0, x0 = int(np.clip(cy - win // 2, 0, 512 - win)), int(np.clip(cx - win // 2, 0, 512 - win))
    sl = (slice(y0, y0 + win), slice(x0, x0 + win))
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.8))
    for k, m in enumerate(METHODS):
        axs[k, 0].imshow(show(ref512)[sl], interpolation="nearest"); axs[k, 0].set_title("original 10 m (bicubic x2)" if k == 0 else "", fontsize=9)
        axs[k, 1].imshow(show(sr[m])[sl], interpolation="nearest"); axs[k, 1].set_title(f"{LAB[m]} 5 m", fontsize=9)
        axs[k, 2].imshow(exact[m][sl], interpolation="nearest"); axs[k, 2].set_title(f"{LAB[m]} ERQA map (blue lost, red added)", fontsize=9)
    clean(axs)
    fig.suptitle(f"{st}: zoom {win}x{win} px of the 5 m grid (x {x0}-{x0 + win}, y {y0}-{y0 + win})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.6); fig.savefig(OUT / f"sr5m_{st}_zoom.png", dpi=100); plt.close(fig)
    # the same comparison at 10 m
    cres = {m: vis_of(to_u8(cons[m]), to_u8(raw), thick=True) for m in METHODS}
    fig, axs = plt.subplots(2, 3, figsize=(12.6, 8.8))
    for k, m in enumerate(METHODS):
        axs[k, 0].imshow(show(cons[m]), interpolation="nearest"); axs[k, 0].set_title(f"{LAB[m]} 5 m averaged back to 10 m", fontsize=9)
        axs[k, 1].imshow(cres[m][1], interpolation="nearest"); axs[k, 1].set_title(f"ERQA {cres[m][0]:.3f} (blue: source edge lost, red: added)", fontsize=9)
        axs[k, 2].imshow(show(raw), interpolation="nearest"); contours(axs[k, 2], m256); axs[k, 2].set_title("original 10 m (reference)", fontsize=9)
    clean(axs)
    fig.suptitle(f"{st}: edges at 10 m", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.6); fig.savefig(OUT / f"cons10_{st}.png", dpi=100); plt.close(fig)

fig, axs = plt.subplots(len(gallery), 3, figsize=(15, 5.3 * len(gallery)))
for i, (st, why, ref512, m512, res) in enumerate(gallery):
    axs[i, 0].imshow(show(ref512), interpolation="nearest"); contours(axs[i, 0], m512); axs[i, 0].set_title(f"{st}\n{why}", fontsize=9)
    for k, m in enumerate(METHODS):
        axs[i, k + 1].imshow(res[m][1], interpolation="nearest"); axs[i, k + 1].set_title(f"{LAB[m]}   ERQA {res[m][0]:.3f}", fontsize=10)
clean(axs)
fig.suptitle("ERQA maps, SR 5 m vs original 10 m. blue: source edge lost, red: edge added by the SR, white: kept.  Yellow / magenta contours: cloud / shadow", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=3.0); fig.savefig(OUT / "erqa_gallery.png", dpi=80); plt.close(fig)
print("done")
