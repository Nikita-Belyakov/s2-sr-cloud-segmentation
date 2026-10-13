"""Probability-averaging ensembles of already trained segmentation models (no training).

ens_bicubic_ediffsr : mean of softmax(model trained on bicubic x2) and softmax(model trained on EDiffSR), same seed
ens_bicubic_esrgan  : the same with ESRGAN
ens_bicubic_bicubic : two bicubic models with different seeds (s and (s+1) mod 3): the plain ensembling effect
Each model gets its own input (bicubic / SR); all inputs are 512x512, so probabilities are averaged on the 512 grid.
Writes seg_runs/<name>_s<seed>/results.json in the same layout as seg_experiment.py (val and test, native and common 256 grid).
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from seg_experiment import CLASSES, SegDS, conf_update, metrics, norm_stats  # noqa: E402

D = WORK_DIR
RUNS = D / "seg_runs"
NORM = {s: norm_stats(s) for s in ("bicubic", "ediffsr", "esrgan")}


def load_model(setup, seed):
    m = smp.DeepLabV3Plus(encoder_name="timm-efficientnet-b0", encoder_weights=None, in_channels=8, classes=3)
    m.load_state_dict(torch.load(RUNS / f"{setup}_s{seed}" / "best.pt", map_location="cpu"))
    return m.cuda().eval()


def loader(split, setup):
    ds = SegDS(split, setup, *NORM[setup])
    return DataLoader(ds, 16, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def run_ensemble(members, split):
    """members: list of (setup, seed); returns metrics (native 512 grid, common 256 grid) of the averaged probabilities
    and of the first member alone (sanity check against results.json)"""
    models = [load_model(s, sd) for s, sd in members]
    loaders = [loader(split, s) for s, _ in members]
    cn, c256, c1 = np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64)
    for batches in zip(*loaders):
        probs = []
        for mdl, (x, m, m256) in zip(models, batches):
            with torch.autocast("cuda", dtype=torch.float16):
                probs.append(mdl(x.cuda(non_blocking=True)).float().softmax(1))
        gt, gt256 = batches[0][1].cuda(), batches[0][2].cuda()
        p = torch.stack(probs).mean(0)
        conf_update(cn, p.argmax(1), gt)
        conf_update(c256, F.avg_pool2d(p, 2).argmax(1), gt256)
        conf_update(c1, F.avg_pool2d(probs[0], 2).argmax(1), gt256)
    return metrics(cn), metrics(c256), metrics(c1)


ENS = {"ens_bicubic_ediffsr": lambda s: [("bicubic", s), ("ediffsr", s)],
       "ens_bicubic_esrgan": lambda s: [("bicubic", s), ("esrgan", s)],
       "ens_bicubic_bicubic": lambda s: [("bicubic", s), ("bicubic", (s + 1) % 3)]}

for name, mk in ENS.items():
    for seed in range(3):
        out = RUNS / f"{name}_s{seed}"
        if (out / "results.json").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        members = mk(seed)
        res = {"setup": name, "seed": seed, "channels": "ensemble", "members": members, "pretrained": "imagenet", "best_epoch": -1,
               "train_time_s": 0.0, "history": [], "classes": CLASSES}
        res["val_native"], res["val_256"], _ = run_ensemble(members, "val")
        res["test_native"], res["test_256"], first = run_ensemble(members, "test")
        # sanity: the first member alone must reproduce its stored test metrics
        ref = json.loads((RUNS / f"{members[0][0]}_s{members[0][1]}" / "results.json").read_text())["test_256"]
        res["sanity_first_member_miou"] = [first["miou"], ref["miou"]]
        (out / "results.json").write_text(json.dumps(res))
        t = res["test_256"]
        print(f"{name} seed {seed}: members {members}  ensemble mIoU {t['miou']:.4f} IoU {np.round(t['iou'], 4).tolist()}  "
              f"(first member alone {first['miou']:.4f}, stored {ref['miou']:.4f})", flush=True)
