"""Boundary / object-size metrics for the already trained models of the fixed split (seg_runs/<setup>_s<seed>/best.pt), test folder.
Also checks that every model reproduces its stored test mIoU (sanity check of weights and normalisation).
Output: seg_boundary/<setup>_s<seed>.json
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import json
import sys
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from seg_boundary import BoundaryAcc  # noqa: E402
from seg_experiment import SegDS, evaluate, norm_stats, parse_setup  # noqa: E402

D = WORK_DIR
OUT = D / "seg_boundary"
OUT.mkdir(exist_ok=True)
SETUPS = ["orig", "bicubic", "ediffsr", "esrgan", "orig_ediffsr", "orig_esrgan", "resid_ediffsr"]

for setup in SETUPS:
    mean, std = norm_stats(setup)
    use_orig, sr = parse_setup(setup)
    C = (8 if use_orig else 0) + (8 if sr else 0)
    dl = DataLoader(SegDS("test", setup, mean, std), 16, shuffle=False, num_workers=6, pin_memory=True)
    for seed in range(3):
        f = OUT / f"{setup}_s{seed}.json"
        run = D / "seg_runs" / f"{setup}_s{seed}"
        if f.exists() or not (run / "best.pt").exists():
            continue
        model = smp.DeepLabV3Plus(encoder_name="timm-efficientnet-b0", encoder_weights=None, in_channels=C, classes=3).cuda()
        model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
        acc = BoundaryAcc()
        _, m256 = evaluate(model, dl, sr, extra=acc)
        ref = json.loads((run / "results.json").read_text())["test_256"]["miou"]
        f.write_text(json.dumps({"setup": setup, "seed": seed, "test_256_miou": m256["miou"], "stored_miou": ref, "boundary": acc.result()}))
        print(f"{setup} seed {seed}: mIoU {m256['miou']:.4f} (stored {ref:.4f})", flush=True)
