"""Unified evaluation of SR on Sen2VENuS (replaces the separate per-model metric scripts).

All methods are scored on the SAME fixed set of validation patches with the SAME metric code:
  - EDiffSR 8-channel (joint 2x/4x), EDiffSR RGBNir 2x (4 ch), EDiffSR IR 4x (4 ch) and the two 4-channel models stacked back to 8 channels,
  - ESRGAN 8-channel with the input scaled as in training, net(x / 255) (and, as a diagnostic, as in test_sen2venus.py: net(x) / 255),
  - bicubic interpolation (the same resized input that EDiffSR receives).
Input as in Sen2VenusDataset: reflectance/10000 clipped to [0, 0.5]; 20 m bands -> 128 (bicubic) and concatenated with the 10 m bands; 256 x 256 grid at 5 m.
Channel order of the 8-channel tensors: [B5, B6, B7, B8A, B2, B3, B4, B8].
Metrics per image, averaged over the images (data_range = 1 as in the original scripts):
  PSNR, SSIM (torchmetrics, gaussian window 11), RMSE, SAM (degrees; the spectral vector is along the channel axis), ERGAS (ratio 2 for 10 m bands, 4 for 20 m bands),
  LPIPS (AlexNet) and FID (64-dim Inception feature) on the true-colour composite (B4, B3, B2).
Stages (different environments): --stage ediffsr (conda landsat), --stage esrgan (satlas venv), --stage metrics (conda landsat).
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

D = WORK_DIR
S2V = S2V_NPZ
VALID = str(S2V_SPLITS / "valid.csv")
OUT = D / "sr_eval"
OUT.mkdir(exist_ok=True)

p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["ediffsr", "esrgan", "esrgan4", "metrics"], required=True)
p.add_argument("--n", type=int, default=1024)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--tag", default="main")
a = p.parse_args()

dev = "cuda"
torch.manual_seed(0)
np.random.seed(0)
PRED = OUT / f"{a.tag}_pred"
PRED.mkdir(exist_ok=True)

valid = pd.read_csv(VALID)["filename"].tolist()
idx = np.linspace(0, len(valid) - 1, a.n).round().astype(int)
files = [valid[i] for i in idx]
print("stage", a.stage, "patches:", len(files), "of", len(valid), flush=True)


def load(f):
    z = np.load(S2V / f)
    c = lambda k: torch.clip(torch.from_numpy(z[k]).float() / 10000, 0, 0.5)  # noqa: E731
    s20, s10, v20, v10 = c("sentinel_20m"), c("sentinel_10m"), c("venus_5m_20m"), c("venus_5m_10m")
    lr = torch.cat([F.interpolate(s20[None], size=(128, 128), mode="bicubic", align_corners=False)[0], s10], 0)  # 8 x 128 x 128
    gt = torch.cat([v20, v10], 0)                                                                           # 8 x 256 x 256
    return lr, gt


def batches():
    for i in range(0, len(files), a.bs):
        chunk = files[i:i + a.bs]
        lrs, gts = zip(*[load(f) for f in chunk])
        yield i, chunk, torch.stack(lrs), torch.stack(gts)


def store(name, c):
    return np.lib.format.open_memmap(PRED / f"{name}.npy", mode="w+", dtype=np.float32, shape=(len(files), c, 256, 256))


# ============================================================================= stage 1: EDiffSR
if a.stage == "ediffsr":
    from s2sr.ediffsr import EDiffSR

    m8 = EDiffSR(device=dev, img_channel=8, model_path=str(WEIGHTS["ediffsr_8ch"]))
    m_rgbn = EDiffSR(device=dev, img_channel=4, model_path=str(WEIGHTS["ediffsr_rgbnir"]))
    m_ir = EDiffSR(device=dev, img_channel=4, model_path=str(WEIGHTS["ediffsr_ir"]))

    def ediff(model, x):
        sr = model.predict_batch(((x - 0.5) * 2.0).to(dev))
        return (sr / 2.0 + 0.5).float().cpu().clamp(0, 0.5)

    o8, oir, orgbn = store("ediffsr8", 8), store("ediffsr_ir", 4), store("ediffsr_rgbn", 4)
    for i, chunk, lr, gt in batches():
        up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False)
        o8[i:i + len(chunk)] = ediff(m8, up).numpy()
        oir[i:i + len(chunk)] = ediff(m_ir, up[:, :4]).numpy()
        orgbn[i:i + len(chunk)] = ediff(m_rgbn, up[:, 4:]).numpy()
        print(i + len(chunk), len(files), flush=True)
    for m in (o8, oir, orgbn):
        m.flush()
    print("EDIFFSR_DONE")

# ============================================================================= stage 2: ESRGAN
elif a.stage == "esrgan":
    from s2sr.esrgan_net import SSR_RRDBNet

    net = SSR_RRDBNet(num_in_ch=8, num_out_ch=8, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    net.load_state_dict(torch.load(str(WEIGHTS["esrgan_8ch"]), map_location="cpu")["params_ema"], strict=True)
    net = net.to(dev).eval()
    e_tr, e_te = store("esrgan", 8), store("esrgan_testscript", 8)
    with torch.no_grad():
        for i, chunk, lr, gt in batches():
            x = lr.to(dev)
            e_tr[i:i + len(chunk)] = F.avg_pool2d(torch.clamp(net(x / 255), 0, 1), 2).float().cpu().clamp(0, 0.5).numpy()   # training scaling
            e_te[i:i + len(chunk)] = F.avg_pool2d(torch.clamp(net(x) / 255, 0, 1), 2).float().cpu().clamp(0, 0.5).numpy()   # test_sen2venus.py scaling
            print(i + len(chunk), len(files), flush=True)
    e_tr.flush(); e_te.flush()
    print("ESRGAN_DONE")

# ============================================================================= stage 2b: 4-channel ESRGAN models
elif a.stage == "esrgan4":
    from s2sr.esrgan_net import SSR_RRDBNet

    def build(key):
        net = SSR_RRDBNet(num_in_ch=4, num_out_ch=4, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        net.load_state_dict(torch.load(str(WEIGHTS[key]), map_location="cpu")["params_ema"], strict=True)
        return net.to(dev).eval()

    m_ir = build("esrgan_ir")
    m_rgbn = build("esrgan_rgbnir")
    o_ir, o_rgbn = store("esrgan_ir", 4), store("esrgan_rgbn", 4)
    o_ir_t, o_rgbn_t = store("esrgan_ir_testscript", 4), store("esrgan_rgbn_testscript", 4)
    pool = lambda t: F.avg_pool2d(torch.clamp(t, 0, 1), 2).float().cpu().clamp(0, 0.5).numpy()  # noqa: E731
    with torch.no_grad():
        for i, chunk, lr, gt in batches():
            x = lr.to(dev)
            sl = slice(i, i + len(chunk))
            o_ir[sl] = pool(m_ir(x[:, :4] / 255))
            o_rgbn[sl] = pool(m_rgbn(x[:, 4:] / 255))
            o_ir_t[sl] = pool(m_ir(x[:, :4]) / 255)
            o_rgbn_t[sl] = pool(m_rgbn(x[:, 4:]) / 255)
            print(i + len(chunk), len(files), flush=True)
    for m in (o_ir, o_rgbn, o_ir_t, o_rgbn_t):
        m.flush()
    print("ESRGAN4_DONE")

# ============================================================================= stage 3: metrics
else:
    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
    from torchmetrics.image.fid import FrechetInceptionDistance

    RATIO8 = torch.tensor([4, 4, 4, 4, 2, 2, 2, 2], dtype=torch.float32)
    lp = lpips.LPIPS(net="alex", verbose=False).to(dev)
    cpu = lambda t: t.detach().cpu().numpy()  # noqa: E731

    def per_image(pred, gt, ratio, rgb_idx):
        pred, gt, ratio = pred.to(dev), gt.to(dev), ratio.to(dev)
        mse_b = ((pred - gt) ** 2).mean((2, 3))                       # B x C
        mse = ((pred - gt) ** 2).mean((1, 2, 3))
        r = {"psnr": cpu(-10 * torch.log10(mse)),                     # data_range = 1
             "rmse": cpu(mse.sqrt())}
        r["ssim"] = cpu(tm_ssim(pred, gt, data_range=1.0, reduction="none"))
        num = (pred * gt).sum(1)
        den = pred.norm(dim=1) * gt.norm(dim=1) + 1e-12
        r["sam_deg"] = cpu(torch.rad2deg(torch.arccos((num / den).clamp(-1, 1))).mean((1, 2)))
        mu = gt.mean((2, 3))
        r["ergas"] = cpu(100 * (((mse_b.sqrt() / mu) / ratio) ** 2).mean(1).sqrt())
        r["psnr_band"] = cpu(-10 * torch.log10(mse_b))
        r["ssim_band"] = np.stack([cpu(tm_ssim(pred[:, c:c + 1], gt[:, c:c + 1], data_range=1.0, reduction="none")) for c in range(pred.shape[1])], 1)
        rgbs = None
        if rgb_idx is not None:
            pr = (pred[:, rgb_idx] * 2).clamp(0, 1)                   # reflectance 0..0.5 -> display range 0..1
            gr = (gt[:, rgb_idx] * 2).clamp(0, 1)
            with torch.no_grad():
                r["lpips"] = cpu(lp(pr, gr, normalize=True).flatten())
            rgbs = (pr, gr)
        return r, rgbs

    # name -> (stored array, channel slice of the 8-ch GT, rgb indices inside the prediction)
    GROUPS = {
        "ediffsr8": ("ediffsr8", slice(0, 8), [6, 5, 4]),
        "ediffsr_sep": (None, slice(0, 8), [6, 5, 4]),                # ediffsr_ir + ediffsr_rgbn stacked
        "esrgan": ("esrgan", slice(0, 8), [6, 5, 4]),
        "esrgan_testscript": ("esrgan_testscript", slice(0, 8), [6, 5, 4]),
        "bicubic": ("bicubic", slice(0, 8), [6, 5, 4]),
        "ediffsr_rgbn": ("ediffsr_rgbn", slice(4, 8), [2, 1, 0]),
        "bicubic_rgbn": ("bicubic", slice(4, 8), [2, 1, 0]),
        "ediffsr_ir": ("ediffsr_ir", slice(0, 4), None),
        "bicubic_ir": ("bicubic", slice(0, 4), None),
        "esrgan_rgbn": ("esrgan_rgbn", slice(4, 8), [2, 1, 0]),
        "esrgan_ir": ("esrgan_ir", slice(0, 4), None),
        "esrgan_sep": (None, slice(0, 8), [6, 5, 4]),                 # esrgan_ir + esrgan_rgbn stacked
        "esrgan_rgbn_testscript": ("esrgan_rgbn_testscript", slice(4, 8), [2, 1, 0]),
        "esrgan_ir_testscript": ("esrgan_ir_testscript", slice(0, 4), None),
    }
    arrs = {k: np.load(PRED / f"{k}.npy", mmap_mode="r") for k in
            ["ediffsr8", "ediffsr_ir", "ediffsr_rgbn", "esrgan", "esrgan_testscript",
             "esrgan_ir", "esrgan_rgbn", "esrgan_ir_testscript", "esrgan_rgbn_testscript"]}
    acc = {g: {} for g in GROUPS}
    fid = {g: FrechetInceptionDistance(feature=64, normalize=True).to(dev) for g, v in GROUPS.items() if v[2] is not None}
    names = []
    for i, chunk, lr, gt in batches():
        sl_i = slice(i, i + len(chunk))
        up = F.interpolate(lr, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 0.5)
        src = {k: torch.from_numpy(np.asarray(v[sl_i])) for k, v in arrs.items()}
        src["bicubic"] = up
        src["ediffsr_sep"] = torch.cat([src["ediffsr_ir"], src["ediffsr_rgbn"]], 1)
        src["esrgan_sep"] = torch.cat([src["esrgan_ir"], src["esrgan_rgbn"]], 1)
        for g, (key, sl, rgb) in GROUPS.items():
            key = key or g
            pred = src[key] if src[key].shape[1] == sl.stop - sl.start else src[key][:, sl]
            res, rgbs = per_image(pred, gt[:, sl], RATIO8[sl], rgb)
            for k, v in res.items():
                acc[g].setdefault(k, []).append(v)
            if rgbs is not None:
                fid[g].update(rgbs[1], real=True)
                fid[g].update(rgbs[0], real=False)
        names += chunk
        print(i + len(chunk), len(files), flush=True)

    summary = {"n": len(files), "tag": a.tag}
    bands = {}
    for g in GROUPS:
        r = {k: np.concatenate(v) for k, v in acc[g].items()}
        pd.DataFrame({k: v for k, v in r.items() if v.ndim == 1} | {"filename": names}).to_csv(OUT / f"{a.tag}_{g}_per_image.csv", index=False)
        bands[f"{g}_psnr_band"], bands[f"{g}_ssim_band"] = r["psnr_band"], r["ssim_band"]
        s = {k: float(v.mean()) for k, v in r.items() if v.ndim == 1}
        s |= {k + "_sem": float(v.std(ddof=1) / np.sqrt(len(v))) for k, v in r.items() if v.ndim == 1}
        if g in fid:
            s["fid64"] = float(fid[g].compute().item())
        summary[g] = s
    np.savez(OUT / f"{a.tag}_bands.npz", **bands)
    json.dump(summary, open(OUT / f"{a.tag}_summary.json", "w"), indent=1)
    print(json.dumps({g: {k: round(v, 4) for k, v in summary[g].items() if not k.endswith("_sem")} for g in GROUPS}, indent=1))
    print("ALLDONE")
