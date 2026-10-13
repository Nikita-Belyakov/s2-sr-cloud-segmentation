"""EDiffSR (x2) / ESRGAN (x4 averaged to x2) on the selected Sen2Venus patches (erqa_s2v/selection.csv): Sentinel-2 10 m -> 5 m (256x256).

Input as in Sen2VenusDataset (fuse=True, only_IR=False): the 4 bands of the 20 m group are bicubically resized to 128x128 and concatenated in front of the 4 bands of the
10 m group, /10000, clip [0, 0.5]; channel order [B5,B6,B7,B8A,B2,B3,B4,B8]. Output: erqa_s2v/<model>/<patch>.npy, float32 (8, 256, 256) reflectance.
EDiffSR: bicubic x2, (x-0.5)*2, 100-step reverse SDE, back to reflectance.  ESRGAN: net(x/255) (as in training), clamp to [0, 1], 512x512 averaged 2x2 to 256x256.
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

D = WORK_DIR
S2V = S2V_NPZ
OUT = D / "erqa_s2v"


def lr_features(a):
    f1 = torch.clip(torch.from_numpy(a["sentinel_20m"]).float() / 10000, 0, 0.5)
    f2 = torch.clip(torch.from_numpy(a["sentinel_10m"]).float() / 10000, 0, 0.5)
    f1 = F.interpolate(f1[None], size=(128, 128), mode="bicubic", align_corners=False)[0]
    return torch.cat([f1, f2], 0)  # 8 x 128 x 128


p = argparse.ArgumentParser()
p.add_argument("--model", choices=["ediffsr", "esrgan"], required=True)
a = p.parse_args()
files = pd.read_csv(OUT / "selection.csv")["filename"].tolist()
out = OUT / a.model
out.mkdir(exist_ok=True)
torch.manual_seed(0)

if a.model == "ediffsr":
    from s2sr.ediffsr import EDiffSR
    model = EDiffSR(device="cuda", img_channel=8, model_path=str(WEIGHTS["ediffsr_8ch"]))

    def run(x):  # x: B x 8 x 128 x 128
        lq = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        sr = model.predict_batch(((lq - 0.5) * 2.0).cuda())
        return (sr / 2.0 + 0.5).float().cpu()
else:
    from s2sr.esrgan_net import SSR_RRDBNet
    net = SSR_RRDBNet(num_in_ch=8, num_out_ch=8, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    net.load_state_dict(torch.load(str(WEIGHTS["esrgan_8ch"]), map_location="cpu")["params_ema"], strict=True)
    net = net.cuda().eval()

    @torch.no_grad()
    def run(x):
        o = torch.clamp(net(x.cuda() / 255), 0, 1).float().cpu()
        return F.avg_pool2d(o, 2)

B = 8
for i in range(0, len(files), B):
    chunk = files[i:i + B]
    x = torch.stack([lr_features(np.load(S2V / f)) for f in chunk])
    sr = run(x).numpy()
    for f, s in zip(chunk, sr):
        np.save(out / (f[:-4] + ".npy"), s.astype(np.float32))
    print(i + len(chunk), len(files), flush=True)
