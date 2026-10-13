"""Cloud / cloud-shadow segmentation: DeepLabV3+ (timm-efficientnet-b0, smp) on different inputs.

setups (all channels in the Sen2Venus order [B5,B6,B7,B8A,B2,B3,B4,B8]):
  orig          8 ch, original 256x256 (clipped to [0, 0.5], as the SR models saw it)
  ediffsr       8 ch, EDiffSR result 512x512
  esrgan        8 ch, ESRGAN result 512x512
  orig_ediffsr  16 ch 512x512: [original bicubic x2 (8), EDiffSR (8)]
  orig_esrgan   16 ch 512x512: [original bicubic x2 (8), ESRGAN (8)]
  bicubic       8 ch, original bicubic x2 512x512 (interpolation baseline)
  resid_ediffsr 16 ch, [bicubic x2, EDiffSR - bicubic]
labels: 0 clear, 1 shadow, 2 cloud  (png values 1 = cloud, 2 = shadow are remapped as in S2_USA_DS)
Metrics are computed from confusion matrices on the test set, on the native grid of each setup (256 or 512)
and on the common 256 grid (SR probabilities averaged over 2x2 blocks, compared with the original 256 mask).

Two ways to define the data:
  default       the fixed train / val / test folders of cloudy_split (results in seg_runs/)
  --fold A..D   scene-level cross-validation: the test scenes of the fold are held out, one more scene is used for validation,
                the remaining 5 scenes for training (results in seg_cv/); the test result also contains boundary / object-size metrics
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import tifffile as tff
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

D = WORK_DIR
SPLIT = D / "cloudy_split"
SR = {"ediffsr": D / "cloudy_split_sr/ediffsr_x2", "esrgan": D / "cloudy_split_sr/esrgan_x4_to512"}
TIF_IDX = [4, 5, 6, 8, 1, 2, 3, 7]
CLASSES = ["clear", "shadow", "cloud"]
SETUPS = ["orig", "ediffsr", "esrgan", "orig_ediffsr", "orig_esrgan", "bicubic", "resid_ediffsr"]
RESID = {"resid_ediffsr": "ediffsr"}
# 4 folds, test = 2 scenes; neighbouring scene numbers are kept together (012_055/056, 040_073/074, 038_073+039_075, 019_025+040_049)
FOLDS = {"A": {"test": ["mask84_012_055", "mask84_012_056"], "val": ["mask84_039_075"]},
         "B": {"test": ["mask84_040_073", "mask84_040_074"], "val": ["mask84_038_073"]},
         "C": {"test": ["mask84_038_073", "mask84_039_075"], "val": ["mask84_040_049"]},
         "D": {"test": ["mask84_019_025", "mask84_040_049"], "val": ["mask84_012_056"]}}


def n_channels(setup):
    use_orig, sr = parse_setup(setup)
    return (8 if use_orig else 0) + (8 if sr else 0)


def parse_setup(setup):
    if setup in RESID:
        return True, RESID[setup]
    use_orig = setup.startswith("orig")
    sr = None if setup == "orig" else setup.split("_")[-1]
    return use_orig, sr


def split_items(split):
    return [(split, f.stem) for f in sorted((SPLIT / split).glob("*.tif"))]


def scene_of(stem):
    return stem.rsplit("_", 2)[0]


def fold_items(fold):
    """(train, val, test) item lists of a cross-validation fold; an item is (original split folder, patch name)"""
    allit = [it for sp in ("train", "val", "test") for it in split_items(sp)]
    t, v = FOLDS[fold]["test"], FOLDS[fold]["val"]
    return ([it for it in allit if scene_of(it[1]) not in t + v], [it for it in allit if scene_of(it[1]) in v],
            [it for it in allit if scene_of(it[1]) in t])


def read_orig(split, stem):
    raw = tff.imread(SPLIT / split / f"{stem}.tif")[TIF_IDX].astype(np.float32) / 10000
    return np.clip(raw, 0, 0.5)


def read_mask(split, stem):
    m = np.array(Image.open(SPLIT / split / f"{stem}.png"))
    out = np.zeros_like(m, dtype=np.uint8)
    out[m == 1] = 2  # cloud
    out[m == 2] = 1  # shadow
    return out


class SegDS(Dataset):
    """src: a split name ('train' / 'val' / 'test') or a list of (split folder, patch name) items"""

    def __init__(self, src, setup, mean, std, aug=False, limit=0):
        self.setup, self.aug = setup, aug
        self.use_orig, self.sr = parse_setup(setup)
        items = split_items(src) if isinstance(src, str) else list(src)
        if limit and limit < len(items):
            items = [items[i] for i in np.linspace(0, len(items) - 1, limit).astype(int)]
        self.items = items
        self.mean = np.asarray(mean, np.float32).reshape(-1, 1, 1)  # full-length per-channel vectors, see norm_stats()
        self.std = np.asarray(std, np.float32).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sp, s = self.items[i]
        m256 = read_mask(sp, s)
        parts = []
        if self.use_orig or self.sr == "bicubic":
            raw = read_orig(sp, s)
            if self.sr:
                raw = F.interpolate(torch.from_numpy(raw)[None], scale_factor=2, mode="bicubic", align_corners=False)[0].numpy()
        if self.use_orig:
            parts.append(raw)
        if self.sr == "bicubic":
            parts.append(raw)
        elif self.sr:
            sr_arr = tff.imread(SR[self.sr] / sp / f"{s}.tif").astype(np.float32) / 10000
            parts.append(sr_arr - raw if self.setup in RESID else sr_arr)
        m = np.repeat(np.repeat(m256, 2, 0), 2, 1) if self.sr else m256
        x = (np.concatenate(parts, 0) - self.mean) / self.std
        if self.aug:
            k = random.randint(0, 3)
            x, m = np.rot90(x, k, (1, 2)), np.rot90(m, k)
            if random.random() < 0.5:
                x, m = x[:, :, ::-1], m[:, ::-1]
            if random.random() < 0.5:
                x, m = x[:, ::-1], m[::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(m)).long(), torch.from_numpy(m256).long()


def band_stats(items=None):
    """per-band mean/std of the clipped original on every 8th training patch (identical for all setups)"""
    items = (items if items is not None else split_items("train"))[::8]
    s = np.zeros(8)
    ss = np.zeros(8)
    n = 0
    for sp, st in items:
        x = read_orig(sp, st).reshape(8, -1).astype(np.float64)
        s += x.sum(1)
        ss += (x * x).sum(1)
        n += x.shape[1]
    mean = s / n
    return mean, np.sqrt(ss / n - mean ** 2)


def norm_stats(setup, items=None):
    """full-length per-channel mean/std vectors for the setup; the SR - bicubic residual channels get their own statistics"""
    mean, std = band_stats(items)
    if setup in RESID:
        s, ss, n = np.zeros(8), np.zeros(8), 0
        for sp, st in (items if items is not None else split_items("train"))[::8]:
            raw = F.interpolate(torch.from_numpy(read_orig(sp, st))[None], scale_factor=2, mode="bicubic", align_corners=False)[0].numpy()
            d = (tff.imread(SR[RESID[setup]] / sp / f"{st}.tif").astype(np.float32) / 10000 - raw).reshape(8, -1).astype(np.float64)
            s += d.sum(1)
            ss += (d * d).sum(1)
            n += d.shape[1]
        dm = s / n
        return np.concatenate([mean, dm]), np.concatenate([std, np.sqrt(ss / n - dm ** 2)])
    use_orig, sr = parse_setup(setup)
    C = (8 if use_orig else 0) + (8 if sr else 0)
    return np.tile(mean, C // 8), np.tile(std, C // 8)


def conf_update(conf, pred, gt):
    conf += torch.bincount((gt * 3 + pred).view(-1), minlength=9).view(3, 3).cpu().numpy()


def metrics(conf):
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return {"iou": iou.tolist(), "f1": f1.tolist(), "precision": (tp / np.maximum(tp + fp, 1)).tolist(),
            "recall": (tp / np.maximum(tp + fn, 1)).tolist(), "miou": float(iou.mean()), "mf1": float(f1.mean()),
            "accuracy": float(tp.sum() / conf.sum()), "fg_iou": float(iou[1:].mean()), "confusion": conf.astype(np.int64).tolist()}


@torch.no_grad()
def evaluate(model, loader, sr_setup, extra=None):
    """extra: optional seg_boundary.BoundaryAcc, updated with the 256-grid predictions"""
    model.eval()
    cn, c256 = np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64)
    for x, m, m256 in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x.cuda(non_blocking=True))
        prob = logits.float().softmax(1)
        conf_update(cn, prob.argmax(1), m.cuda())
        if sr_setup:
            prob = F.avg_pool2d(prob, 2)
        pred256 = prob.argmax(1)
        conf_update(c256, pred256, m256.cuda())
        if extra is not None:
            extra.update(pred256, m256.cuda())
    return metrics(cn), metrics(c256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setup", choices=SETUPS, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold", choices=list(FOLDS), default="", help="scene-level cross-validation fold; empty = fixed train/val/test folders")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--pretrained", choices=["imagenet", "none"], default="imagenet")
    p.add_argument("--limit_train", type=int, default=0)
    p.add_argument("--limit_val", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()

    out_root = Path(a.out or (D / ("seg_cv" if a.fold else "seg_runs")))
    run = out_root / (f"{a.setup}_f{a.fold}_s{a.seed}" if a.fold else f"{a.setup}_s{a.seed}")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "results.json").exists():
        print("already done:", run)
        return
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True

    if a.fold:
        tr_items, va_items, te_items = fold_items(a.fold)
    else:
        tr_items, va_items, te_items = "train", "val", "test"
    mean, std = norm_stats(a.setup, None if not a.fold else tr_items)
    use_orig, sr = parse_setup(a.setup)
    C = n_channels(a.setup)
    tr = SegDS(tr_items, a.setup, mean, std, aug=True, limit=a.limit_train)
    va = SegDS(va_items, a.setup, mean, std, limit=a.limit_val)
    te = SegDS(te_items, a.setup, mean, std)
    wk = lambda w: random.seed(torch.initial_seed() % 2 ** 31 + w)  # noqa: E731
    dl_tr = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True, persistent_workers=True, worker_init_fn=wk)
    dl_va = DataLoader(va, a.batch, num_workers=a.workers, pin_memory=True)
    dl_te = DataLoader(te, a.batch, num_workers=a.workers, pin_memory=True)
    print(f"setup {a.setup} fold {a.fold or '-'} seed {a.seed}: {C} channels, train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    model = smp.DeepLabV3Plus(encoder_name="timm-efficientnet-b0", encoder_weights=None if a.pretrained == "none" else "imagenet",
                              in_channels=C, classes=3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.cuda.amp.GradScaler()
    ce = nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss("multiclass", from_logits=True)

    hist, best, best_ep = [], -1, -1
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tl, n = 0.0, 0
        for x, m, _ in dl_tr:
            x, m = x.cuda(non_blocking=True), m.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            loss = ce(out.float(), m) + dice(out.float(), m)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(x)
            n += len(x)
        sched.step()
        vn, _ = evaluate(model, dl_va, sr)
        score = vn["fg_iou"]  # mean IoU of shadow and cloud, model selection on val
        hist.append({"epoch": ep, "train_loss": tl / n, "val_miou": vn["miou"], "val_fg_iou": score, "val_iou": vn["iou"]})
        if score > best:
            best, best_ep = score, ep
            torch.save(model.state_dict(), run / "best.pt")
        print(f"ep {ep:02d} loss {tl / n:.4f} val mIoU {vn['miou']:.4f} fg-IoU {score:.4f} IoU {np.round(vn['iou'], 3).tolist()}  best@{best_ep}  {time.time() - t0:.0f}s", flush=True)

    model.load_state_dict(torch.load(run / "best.pt"))
    res = {"setup": a.setup, "seed": a.seed, "fold": a.fold, "channels": C, "pretrained": a.pretrained, "epochs": a.epochs, "best_epoch": best_ep,
           "train_time_s": time.time() - t0, "history": hist, "classes": CLASSES,
           "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    res["val_native"], res["val_256"] = evaluate(model, dl_va, sr)
    from seg_boundary import BoundaryAcc
    bacc = BoundaryAcc()
    res["test_native"], res["test_256"] = evaluate(model, dl_te, sr, extra=bacc)
    res["test_boundary"] = bacc.result()
    (run / "results.json").write_text(json.dumps(res))
    t = res["test_256"]
    print(f"TEST (256 grid) mIoU {t['miou']:.4f} IoU {np.round(t['iou'], 4).tolist()} F1 {np.round(t['f1'], 4).tolist()}", flush=True)


if __name__ == "__main__":
    main()
