"""EDiffSR 8-channel (2x) inference on Sentinel-2 cloudy patches stored in a zip.

Input tif: (14, 256, 256) uint16, bands in S2 order
  idx: 0 B1, 1 B2, 2 B3, 3 B4, 4 B5, 5 B6, 6 B7, 7 B8, 8 B8A, 9 B9, 10 B11, 11 B12, 12 B10(?), 13 cloud flag(?)
Model input channel order (as concatenated in Sen2VenusDataset, fuse=True, only_IR=False):
  [sentinel_20m: B5, B6, B7, B8A] + [sentinel_10m: B2, B3, B4, B8]
Preprocessing as in Sen2VenusDataset: /10000, clip [0, 0.5], bicubic x2, then (x-0.5)*2.
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import tifffile as tff
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from s2sr.ediffsr import EDiffSR

MODEL_BANDS = ["B5", "B6", "B7", "B8A", "B2", "B3", "B4", "B8"]
TIF_IDX = [4, 5, 6, 8, 1, 2, 3, 7]


class CloudyZipDataset(Dataset):
    def __init__(self, zip_path, names, scale=2, src_dir=None):
        self.zip_path = zip_path
        self.src_dir = src_dir
        self.names = names
        self.scale = scale
        self.zf = None

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        if self.src_dir:
            arr = tff.imread(Path(self.src_dir) / name)  # (14, H, W); name may be "split/file.tif"
        else:
            if self.zf is None:  # one handle per worker
                self.zf = zipfile.ZipFile(self.zip_path)
            arr = tff.imread(io.BytesIO(self.zf.read(name)))
        x = torch.from_numpy(arr[TIF_IDX].astype(np.float32)) / 10000
        x = torch.clip(x, 0, 0.5)
        x = F.interpolate(x[None], scale_factor=self.scale, mode="bicubic", align_corners=False)[0]
        return x, name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="")
    p.add_argument("--src_dir", default="", help="folder with the unpacked *.tif patches (used instead of --zip)")
    p.add_argument("--list", default="", help="text file with patch names (one per line); overrides --limit")
    p.add_argument("--out", default="")
    p.add_argument("--split_root", default="", help="folder with train/ val/ test/ subfolders (tif + png)")
    p.add_argument("--out_root", default="", help="results go to <out_root>/<split>/<patch>.tif")
    p.add_argument("--weights", required=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all; otherwise evenly spaced subset")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dtype", choices=["float32", "uint16"], default="float32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--shard", default="", help="i/N")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out_root or a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "band_order.json").write_text(json.dumps(
        {"model_bands": MODEL_BANDS, "source_tif_idx": TIF_IDX, "dtype": a.dtype,
         "units": "reflectance (float32) or reflectance*10000 (uint16)", "scale": 2}, indent=1))

    if a.split_root:
        import csv
        a.src_dir = a.split_root
        with open(Path(a.split_root) / "split_manifest.csv") as f:
            names = [f"{r['split']}/{r['patch']}.tif" for r in csv.DictReader(f)]
        for sp in {n.split("/")[0] for n in names}:
            (out / sp).mkdir(exist_ok=True)
    elif a.src_dir:
        names = sorted(f.name for f in Path(a.src_dir).glob("*.tif"))
    else:
        with zipfile.ZipFile(a.zip) as z:
            names = sorted(n for n in z.namelist() if n.endswith(".tif"))
    if a.list:
        want = {Path(l.strip()).name for l in open(a.list) if l.strip()}
        names = [n for n in names if Path(n).name in want]
    elif a.limit and a.limit < len(names):
        names = [names[i] for i in np.linspace(0, len(names) - 1, a.limit).astype(int)]
    if a.shard:  # "i/N": every N-th patch starting at i, for SLURM array jobs
        i, n = map(int, a.shard.split("/"))
        names = names[i::n]
    if a.skip_existing:
        names = [n for n in names if not (out / (n if a.split_root else Path(n).stem + ".tif")).exists()]
    print(f"{len(names)} patches to process", flush=True)

    model = EDiffSR(device="cuda", img_channel=8, model_path=a.weights)
    loader = DataLoader(CloudyZipDataset(a.zip, names, src_dir=a.src_dir), batch_size=a.batch, num_workers=a.workers)

    t0 = time.time()
    done = 0
    for lq, stems in loader:
        lq = (lq - 0.5) * 2.0
        sr = model.predict_batch(lq.cuda())
        sr = (sr / 2.0 + 0.5).float().cpu().numpy()
        for img, name in zip(sr, stems):
            if a.dtype == "uint16":
                img = np.clip(img * 10000, 0, 65535).round().astype(np.uint16)
            dst = out / name if a.split_root else out / f"{Path(name).stem}.tif"
            tff.imwrite(dst, img)
        done += len(stems)
        el = time.time() - t0
        print(f"{done}/{len(names)}  {el:.0f}s  ({el / done:.1f}s/patch)", flush=True)


if __name__ == "__main__":
    main()
