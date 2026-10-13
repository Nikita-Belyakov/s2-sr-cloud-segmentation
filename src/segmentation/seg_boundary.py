"""Boundary-aware and object-size-aware segmentation metrics for shadow (1) and cloud (2).

Everything is computed on the common 256 grid and accumulated as raw counts, so that runs / folds can be pooled exactly.

boundary F-score (tolerance tau px): boundary of a class mask = its pixels that touch a pixel of another class (3x3 neighbourhood).
    precision = share of predicted boundary pixels within tau px of a GT boundary pixel, recall = the reverse.
boundary IoU (width d px): IoU of the inner bands of width d along the contours of GT and prediction (Cheng et al., 2021).
size strata (connected components, 8-connectivity; sizes in 256-grid pixels = 100 m2 each):
    bins [<50, 50-500, >=500] px
    recall by GT object size: share of GT pixels of objects of that size that are predicted correctly;
    detection rate: share of GT objects with at least half of their pixels predicted correctly;
    precision by predicted object size: share of predicted pixels of objects of that size that are correct.
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

CLASSES = {1: "shadow", 2: "cloud"}
TAUS = (1, 2)
BIOU_D = (2,)
EDGES = [50, 500]
BIN_NAMES = ["<50 px", "50-500 px", ">=500 px"]


def _erode(m, r):
    """binary erosion of a (B,1,H,W) float mask with a (2r+1)^2 window, replicate padding (no false boundary at the image border)"""
    p = F.pad(m, (r, r, r, r), mode="replicate")
    return 1 - F.max_pool2d(1 - p, 2 * r + 1, 1)


class BoundaryAcc:
    def __init__(self):
        self.bf = {c: {t: np.zeros(4) for t in TAUS} for c in CLASSES}
        self.biou = {c: {d: np.zeros(2) for d in BIOU_D} for c in CLASSES}
        self.sgt = {c: np.zeros((3, 4)) for c in CLASSES}   # per bin: objects, detected objects, gt px, tp px
        self.spr = {c: np.zeros((3, 3)) for c in CLASSES}   # per bin: objects, pred px, tp px

    @torch.no_grad()
    def update(self, pred, gt):
        """pred, gt: (B,H,W) integer tensors with labels 0 clear / 1 shadow / 2 cloud"""
        for c in CLASSES:
            G = (gt == c).float()[:, None]
            P = (pred == c).float()[:, None]
            bG = (G * (1 - _erode(G, 1))) > 0.5
            bP = (P * (1 - _erode(P, 1))) > 0.5
            for t in TAUS:
                dG = F.max_pool2d(bG.float(), 2 * t + 1, 1, t) > 0
                dP = F.max_pool2d(bP.float(), 2 * t + 1, 1, t) > 0
                self.bf[c][t] += np.array([(bP & dG).sum().item(), bP.sum().item(), (bG & dP).sum().item(), bG.sum().item()])
            for d in BIOU_D:
                aG = (G * (1 - _erode(G, d))) > 0.5
                aP = (P * (1 - _erode(P, d))) > 0.5
                self.biou[c][d] += np.array([(aG & aP).sum().item(), (aG | aP).sum().item()])
        pn, gn = pred.cpu().numpy(), gt.cpu().numpy()
        for i in range(len(pn)):
            for c in CLASSES:
                g, p = gn[i] == c, pn[i] == c
                lab, n = ndimage.label(g, structure=np.ones((3, 3)))
                if n:
                    sz = np.bincount(lab.ravel(), minlength=n + 1)[1:]
                    tp = np.bincount(lab.ravel(), weights=p.ravel().astype(np.float64), minlength=n + 1)[1:]
                    b = np.digitize(sz, EDGES)
                    for k in range(3):
                        s = b == k
                        self.sgt[c][k] += [s.sum(), (s & (tp / sz >= 0.5)).sum(), sz[s].sum(), tp[s].sum()]
                lab, n = ndimage.label(p, structure=np.ones((3, 3)))
                if n:
                    sz = np.bincount(lab.ravel(), minlength=n + 1)[1:]
                    tp = np.bincount(lab.ravel(), weights=g.ravel().astype(np.float64), minlength=n + 1)[1:]
                    b = np.digitize(sz, EDGES)
                    for k in range(3):
                        s = b == k
                        self.spr[c][k] += [s.sum(), sz[s].sum(), tp[s].sum()]

    def result(self):
        return {"bf": {CLASSES[c]: {str(t): v.tolist() for t, v in d.items()} for c, d in self.bf.items()},
                "biou": {CLASSES[c]: {str(k): v.tolist() for k, v in d.items()} for c, d in self.biou.items()},
                "size_gt": {CLASSES[c]: v.tolist() for c, v in self.sgt.items()},
                "size_pr": {CLASSES[c]: v.tolist() for c, v in self.spr.items()}}


# ---------------------------------------------------------------- helpers for the reports (raw counts -> metrics)
def pool(raws):
    """element-wise sum of several raw result dicts"""
    out = None
    for r in raws:
        cur = {k: _tonp(r[k]) for k in ("bf", "biou", "size_gt", "size_pr")}
        if out is None:
            out = cur
        else:
            for k in out:
                _add(out[k], cur[k])
    return out


def _tonp(x):
    return {k: _tonp(v) for k, v in x.items()} if isinstance(x, dict) else np.array(x, dtype=np.float64)


def _add(a, b):
    for k in a:
        if isinstance(a[k], dict):
            _add(a[k], b[k])
        else:
            a[k] = a[k] + b[k]


def bf_prf(cnt):
    tp_pr, n_pr, tp_gt, n_gt = cnt
    p = tp_pr / n_pr if n_pr else np.nan
    r = tp_gt / n_gt if n_gt else np.nan
    f = 2 * p * r / (p + r) if (p + r) else np.nan
    return p, r, f


def biou(cnt):
    return cnt[0] / cnt[1] if cnt[1] else np.nan


def size_recall(cnt):      # per bin: gt px recall, detection rate, number of objects
    return cnt[:, 3] / np.maximum(cnt[:, 2], 1), cnt[:, 1] / np.maximum(cnt[:, 0], 1), cnt[:, 0]


def size_precision(cnt):   # per bin: precision of predicted pixels, number of predicted objects
    return cnt[:, 2] / np.maximum(cnt[:, 1], 1), cnt[:, 0]
