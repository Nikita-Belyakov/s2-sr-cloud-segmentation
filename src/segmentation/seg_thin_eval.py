"""Cloud metrics for THIN (semi-transparent) clouds, defined by the brightness of the cloud object, for the already trained segmenters.

Brightness of an object = mean reflectance of B2, B3, B4 (patch bands 1..3, no clipping) over its pixels. Objects are connected components (8-connectivity, 256 grid) of the cloud class
found in the ground truth or in the prediction. Groups (cloud class only):
  thin        brightness < 0.15            (any size)
  mid         0.15 <= brightness < 0.30    (any size)
  dense       brightness >= 0.30           (any size)
  small_thin  brightness < 0.15 and fewer than 50 pixels
Region of a group = pixels of its objects in the ground truth OR in the prediction. Inside the region TP = both say cloud, FN = only the ground truth, FP = only the prediction;
IoU = TP/(TP+FP+FN), F1 = 2TP/(2TP+FP+FN); 3 x 3 confusion matrix over the region pixels; detection rate = share of the ground-truth objects of the group with >= half of the pixels predicted as cloud.

usage: python seg_thin_eval.py fixed|cv     -> seg_thin/<protocol>_<setup>_<fold>_s<seed>.json
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
import tifffile as tff
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from seg_experiment import SPLIT, SegDS, fold_items, n_channels, norm_stats, parse_setup, split_items  # noqa: E402

D = WORK_DIR
OUT = D / "seg_thin"
OUT.mkdir(exist_ok=True)
SETUPS = ["orig", "bicubic", "ediffsr", "esrgan", "orig_ediffsr", "orig_esrgan"]
GROUPS = {"thin": (0.0, 0.15, None), "mid": (0.15, 0.30, None), "dense": (0.30, 1e9, None), "small_thin": (0.0, 0.15, 50)}
S8 = np.ones((3, 3))


def objects(mask, vis):
    lab, n = ndimage.label(mask, structure=S8)
    if not n:
        return lab, n, np.zeros(0), np.zeros(0)
    idx = np.arange(1, n + 1)
    return lab, n, np.bincount(lab.ravel(), minlength=n + 1)[1:], np.asarray(ndimage.mean(vis, lab, idx))


class ThinAcc:
    def __init__(self):
        self.conf = {k: np.zeros((3, 3), np.int64) for k in GROUPS}
        self.obj = {k: np.zeros(2, np.int64) for k in GROUPS}

    def update(self, pred, gt, vis):
        for i in range(len(pred)):
            p, g, v = pred[i], gt[i], vis[i]
            lg, ng, szg, mg = objects(g == 2, v)
            lp, npd, szp, mp = objects(p == 2, v)
            det = np.bincount(lg.ravel(), weights=(p == 2).ravel().astype(np.float64), minlength=ng + 1)[1:] if ng else np.zeros(0)
            for k, (lo, hi, mx) in GROUPS.items():
                fg = np.zeros(ng + 1, bool)
                fp_ = np.zeros(npd + 1, bool)
                if ng:
                    s = (mg >= lo) & (mg < hi) & ((szg < mx) if mx else True)
                    fg[1:] = s
                    self.obj[k] += [int(s.sum()), int((s & (det >= 0.5 * np.maximum(szg, 1))).sum())]
                if npd:
                    fp_[1:] = (mp >= lo) & (mp < hi) & ((szp < mx) if mx else True)
                R = fg[lg] | fp_[lp]
                if R.any():
                    self.conf[k] += np.bincount((g[R].astype(np.int64) * 3 + p[R]).ravel(), minlength=9).reshape(3, 3)

    def result(self):
        return {k: {"confusion": self.conf[k].tolist(), "gt_objects": int(self.obj[k][0]), "detected": int(self.obj[k][1])} for k in GROUPS}


@torch.no_grad()
def run_model(setup, ckpt, test_items, norm_items):
    mean, std = norm_stats(setup, norm_items)
    _, sr = parse_setup(setup)
    ds = SegDS(test_items, setup, mean, std)
    dl = DataLoader(ds, 16, shuffle=False, num_workers=6, pin_memory=True)
    model = smp.DeepLabV3Plus(encoder_name="timm-efficientnet-b0", encoder_weights=None, in_channels=n_channels(setup), classes=3).cuda()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    acc, start = ThinAcc(), 0
    for x, _, m256 in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            prob = model(x.cuda(non_blocking=True)).float().softmax(1)
        if sr:
            prob = F.avg_pool2d(prob, 2)
        vis = np.stack([tff.imread(SPLIT / sp / f"{st}.tif")[1:4].astype(np.float32).mean(0) / 10000 for sp, st in ds.items[start:start + len(x)]])
        acc.update(prob.argmax(1).cpu().numpy(), m256.numpy(), vis)
        start += len(x)
    return acc.result()


mode = sys.argv[1]
if mode == "fixed":
    for setup in SETUPS:
        for seed in range(3):
            f = OUT / f"fixed_{setup}__s{seed}.json"
            ck = D / "seg_runs" / f"{setup}_s{seed}" / "best.pt"
            if f.exists() or not ck.exists():
                continue
            f.write_text(json.dumps({"protocol": "fixed", "setup": setup, "fold": "", "seed": seed, "groups": run_model(setup, ck, split_items("test"), None)}))
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
                f.write_text(json.dumps({"protocol": "cv", "setup": setup, "fold": fold, "seed": seed, "groups": run_model(setup, ck, te, tr)}))
                print("cv", fold, setup, seed, flush=True)
