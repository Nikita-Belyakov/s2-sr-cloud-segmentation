"""ESRGAN (SSR_RRDBNet, 8 ch, x4, trained on Sen2Venus) inference on the Sentinel-2 cloudy patches.

Input patch: 14-ch tif 256x256 -> 8 channels in the Sen2Venus order [B5,B6,B7,B8A,B2,B3,B4,B8],
/10000, clip [0, 0.5] (as in Sen2VenusDataset). The net takes the LR grid as is (no bicubic pre-upsampling): 256 -> 1024.

Two input/output normalisations exist in the code of the project:
  'train': out = net(x / 255)      (SSRESRGANModel.feed_data divides lr by 255, gt stays in reflectance)
  'test' : out = net(x) / 255      (test_sen2venus.py)
--norm auto picks the one with the higher PSNR on --calibrate Sen2Venus validation samples.
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tff
import torch
import torch.nn.functional as F

from s2sr.esrgan_net import SSR_RRDBNet  # noqa: E402

TIF_IDX = [4, 5, 6, 8, 1, 2, 3, 7]


def load_net(weights):
    net = SSR_RRDBNet(num_in_ch=8, num_out_ch=8, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    sd = torch.load(weights, map_location="cpu")
    net.load_state_dict(sd["params_ema"], strict=True)
    return net.cuda().eval()


@torch.no_grad()
def run(net, x, norm):
    x = x.cuda()[None]
    out = net(x / 255) if norm == "train" else net(x) / 255
    return torch.clamp(out, 0, 1)[0].float().cpu()


def sen2venus_pair(fname):
    a = np.load(S2V_NPZ / fname)
    f1 = torch.from_numpy(a["sentinel_20m"]).float() / 10000
    f2 = torch.from_numpy(a["sentinel_10m"]).float() / 10000
    f1 = F.interpolate(torch.clip(f1, 0, 0.5)[None], size=(128, 128), mode="bicubic", align_corners=False)[0]
    lr = torch.cat([f1, torch.clip(f2, 0, 0.5)], 0)
    t1 = torch.clip(torch.from_numpy(a["venus_5m_20m"]).float() / 10000, 0, 0.5)
    t2 = torch.clip(torch.from_numpy(a["venus_5m_10m"]).float() / 10000, 0, 0.5)
    gt = F.interpolate(torch.cat([t1, t2], 0)[None], size=(512, 512), mode="bicubic", align_corners=False)[0]
    return lr, gt  # 8x128x128, 8x512x512


def psnr(a, b):
    return float(10 * torch.log10(1.0 / F.mse_loss(a, b)))


def calibrate(net, n):
    df = pd.read_csv(S2V_SPLITS / "valid.csv").iloc[:: max(1, 1000 // n)][:n]
    res = {"train": [], "test": [], "bicubic": []}
    for fn in df["filename"]:
        lr, gt = sen2venus_pair(fn)
        for k in ("train", "test"):
            res[k].append(psnr(run(net, lr, k), gt))
        res["bicubic"].append(psnr(torch.clamp(F.interpolate(lr[None], scale_factor=4, mode="bicubic", align_corners=False)[0], 0, 1), gt))
    m = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"calibration on {len(df)} Sen2Venus valid samples, mean PSNR (dB) vs GT: {m}", flush=True)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default="")
    p.add_argument("--list", default="")
    p.add_argument("--out", default="")
    p.add_argument("--split_root", default="", help="folder with train/ val/ test/ subfolders and split_manifest.csv")
    p.add_argument("--out_root", default="", help="results go to <out_root>/<split>/<patch>.tif")
    p.add_argument("--weights", required=True)
    p.add_argument("--norm", choices=["auto", "train", "test"], default="auto")
    p.add_argument("--calibrate", type=int, default=8)
    p.add_argument("--out_size", type=int, choices=[256, 512, 1024], default=1024,
                   help="saved size; the native x4 output is block-averaged down to it (1024 = native for a 256 input)")
    p.add_argument("--dtype", choices=["float32", "uint16"], default="float32")
    p.add_argument("--shard", default="", help="i/N")
    p.add_argument("--skip_existing", action="store_true")
    a = p.parse_args()

    net = load_net(a.weights)
    norm = a.norm
    if norm == "auto":
        m = calibrate(net, a.calibrate)
        norm = "train" if m["train"] >= m["test"] else "test"
    print("using norm:", norm, flush=True)

    out = Path(a.out_root or a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "norm.txt").write_text(norm)
    (out / "band_order.json").write_text(json.dumps(
        {"model_bands": ["B5", "B6", "B7", "B8A", "B2", "B3", "B4", "B8"], "source_tif_idx": TIF_IDX,
         "dtype": a.dtype, "units": "reflectance (float32) or reflectance*10000 (uint16)",
         "net_scale": 4, "saved_size": a.out_size, "norm": norm}, indent=1))
    if a.split_root:
        a.src_dir = a.split_root
        with open(Path(a.split_root) / "split_manifest.csv") as f:
            names = [f"{r['split']}/{r['patch']}.tif" for r in csv.DictReader(f)]
        for sp in {n.split("/")[0] for n in names}:
            (out / sp).mkdir(exist_ok=True)
    else:
        names = sorted(f.name for f in Path(a.src_dir).glob("*.tif"))
        if a.list:
            want = {l.strip() for l in open(a.list) if l.strip()}
            names = [n for n in names if n in want]
    if a.shard:
        si, sn = map(int, a.shard.split("/"))
        names = names[si::sn]
    if a.skip_existing:
        names = [n for n in names if not (out / n).exists()]
    print(len(names), "patches to process", flush=True)
    t0 = time.time()
    for i, n in enumerate(names, 1):
        arr = tff.imread(Path(a.src_dir) / n)
        x = torch.clip(torch.from_numpy(arr[TIF_IDX].astype(np.float32)) / 10000, 0, 0.5)
        sr = run(net, x, norm)
        f = sr.shape[-1] // a.out_size
        if f > 1:
            sr = F.avg_pool2d(sr[None], f)[0]
        sr = sr.numpy()
        if a.dtype == "uint16":
            sr = np.clip(sr * 10000, 0, 65535).round().astype(np.uint16)
        (out / n).parent.mkdir(parents=True, exist_ok=True)
        tff.imwrite(out / n, sr)
        if i % 20 == 0 or i == len(names):
            print(f"{i}/{len(names)} {n} {sr.shape} {(time.time() - t0) / i:.2f}s/patch", flush=True)


if __name__ == "__main__":
    main()
