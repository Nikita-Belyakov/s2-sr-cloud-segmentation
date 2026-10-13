"""IoU / F1 / confusion matrix restricted to SMALL objects of one class (cloud = 2, shadow = 1) for the already trained segmenters.

Small = a connected component (8-connectivity, 256 grid, 1 px = 10 m x 10 m) with fewer than 50 pixels (< 0.5 ha), found in the ground truth or in the prediction.
Region R of a class c = pixels that belong to a small component of c in the ground truth OR in the prediction.
Inside R: TP = gt == c and pred == c, FN = gt == c and pred != c, FP = gt != c and pred == c; IoU = TP / (TP + FP + FN), F1 = 2TP / (2TP + FP + FN).
The 3 x 3 confusion matrix (rows = ground truth, columns = prediction) is accumulated over the pixels of R.
Detection rate = share of the small ground-truth objects of which at least half of the pixels are predicted as c.

usage: python seg_small_eval.py fixed|cv     -> seg_small/<protocol>_<setup>_<fold>_s<seed>.json
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import json
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from seg_experiment import SegDS, fold_items, n_channels, norm_stats, parse_setup, split_items  # noqa: E402

D = WORK_DIR
OUT = D / "seg_small"
OUT.mkdir(exist_ok=True)
THR = 50
CLS = {1: "shadow", 2: "cloud"}
SETUPS = ["orig", "bicubic", "ediffsr", "esrgan", "orig_ediffsr", "orig_esrgan"]
S8 = np.ones((3, 3))


class SmallAcc:
    def __init__(self):
        self.conf = {c: np.zeros((3, 3), np.int64) for c in CLS}
        self.obj = {c: np.zeros(2, np.int64) for c in CLS}  # small gt objects, detected ones

    def update(self, pred, gt):
        for i in range(len(pred)):
            p, g = pred[i], gt[i]
            for c in CLS:
                lg, ng = ndimage.label(g == c, structure=S8)
                lp, npd = ndimage.label(p == c, structure=S8)
                small_g = np.zeros(ng + 1, bool)
                small_p = np.zeros(npd + 1, bool)
                if ng:
                    sz = np.bincount(lg.ravel(), minlength=ng + 1)
                    small_g[1:] = sz[1:] < THR
                    det = np.bincount(lg.ravel(), weights=(p == c).ravel().astype(np.float64), minlength=ng + 1)
                    sm = small_g.copy()
                    sm[0] = False
                    self.obj[c] += [int(sm.sum()), int((sm[1:] & (det[1:] >= 0.5 * np.maximum(sz[1:], 1))).sum())]
                if npd:
                    szp = np.bincount(lp.ravel(), minlength=npd + 1)
                    small_p[1:] = szp[1:] < THR
                R = small_g[lg] | small_p[lp]
                R &= True
                if R.any():
                    self.conf[c] += np.bincount((g[R].astype(np.int64) * 3 + p[R]).ravel(), minlength=9).reshape(3, 3)

    def result(self):
        return {CLS[c]: {"confusion": self.conf[c].tolist(), "gt_small_objects": int(self.obj[c][0]), "detected": int(self.obj[c][1])} for c in CLS}


@torch.no_grad()
def run_model(setup, ckpt, test_items, norm_items):
    mean, std = norm_stats(setup, norm_items)
    _, sr = parse_setup(setup)
    dl = DataLoader(SegDS(test_items, setup, mean, std), 16, shuffle=False, num_workers=6, pin_memory=True)
    model = smp.DeepLabV3Plus(encoder_name="timm-efficientnet-b0", encoder_weights=None, in_channels=n_channels(setup), classes=3).cuda()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    acc = SmallAcc()
    for x, _, m256 in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            prob = model(x.cuda(non_blocking=True)).float().softmax(1)
        if sr:
            prob = F.avg_pool2d(prob, 2)
        acc.update(prob.argmax(1).cpu().numpy(), m256.numpy())
    return acc.result()


mode = sys.argv[1]
if mode == "fixed":
    for setup in SETUPS:
        for seed in range(3):
            f = OUT / f"fixed_{setup}__s{seed}.json"
            ck = D / "seg_runs" / f"{setup}_s{seed}" / "best.pt"
            if f.exists() or not ck.exists():
                continue
            r = run_model(setup, ck, split_items("test"), None)
            f.write_text(json.dumps({"protocol": "fixed", "setup": setup, "fold": "", "seed": seed, **r}))
            print("fixed", setup, seed, flush=True)
else:
    for fold in "ABCD":
        tr, va, te = fold_items(fold)
        for setup in SETUPS:
            for seed in range(2):
                f = OUT / f"cv_{setup}_f{fold}_s{seed}.json"
                ck = D / "seg_cv" / f"{setup}_f{fold}_s{seed}" / "best.pt"
                if f.exists() or not ck.exists():
                    continue
                r = run_model(setup, ck, te, tr)
                f.write_text(json.dumps({"protocol": "cv", "setup": setup, "fold": fold, "seed": seed, **r}))
                print("cv", fold, setup, seed, flush=True)
