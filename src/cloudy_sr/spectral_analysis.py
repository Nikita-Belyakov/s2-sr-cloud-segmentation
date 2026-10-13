"""Spectral / correlation analysis of the original Sentinel-2 cloudy patches vs EDiffSR and ESRGAN super-resolution.

Channels are analysed in wavelength order: B2, B3, B4, B5, B6, B7, B8, B8A
(the model channel order is [B5,B6,B7,B8A,B2,B3,B4,B8]; reordered right after loading).
Classes (raw png values): 0 clear, 1 cloud, 2 shadow.

Sources
  orig_raw : original 256x256, reflectance without clipping
  orig     : original 256x256 clipped to [0, 0.5] (= what both SR models were given)
  ediffsr  : EDiffSR x2 result, 512x512
  esrgan   : ESRGAN x4 result averaged to 512x512
Spectral distortion of a model is measured on the 256 grid: SR is averaged 2x2 (consistency check) and compared with `orig`.
Segmentation proxy: a pixel-wise gradient-boosting classifier (no spatial context) trained on one source and tested on another.
"""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile as tff
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = WORK_DIR
OUT = D / "spectral_analysis"
OUT.mkdir(exist_ok=True)
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2  # every STEP-th patch of every split

TIF_IDX = [4, 5, 6, 8, 1, 2, 3, 7]        # model order inside the 14-ch tif
WL = [4, 5, 6, 0, 1, 2, 7, 3]              # model order -> wavelength order
SRC_IDX = [TIF_IDX[i] for i in WL]
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]
CN = ["clear", "cloud", "shadow"]
SOURCES = ["orig_raw", "orig", "ediffsr", "esrgan"]
MODELS = ["ediffsr", "esrgan"]
COL = {"orig_raw": "#888888", "orig": "#1f77b4", "ediffsr": "#d62728", "esrgan": "#2ca02c"}
LAB = {"orig_raw": "original (unclipped)", "orig": "original (clipped 0.5)", "ediffsr": "EDiffSR x2", "esrgan": "ESRGAN x4->512"}
ROOT = {"ediffsr": D / "cloudy_split_sr/ediffsr_x2", "esrgan": D / "cloudy_split_sr/esrgan_x4_to512"}
NB, BMAX = 200, 1.0
NSAM, SAMMAX = 300, 30.0


class Acc:
    def __init__(self):
        self.hist = {s: np.zeros((3, 8, NB), np.int64) for s in SOURCES}
        self.n = {s: np.zeros(3) for s in SOURCES}
        self.s = {s: np.zeros((3, 8)) for s in SOURCES}
        self.ss = {s: np.zeros((3, 8, 8)) for s in SOURCES}
        z = lambda *sh: np.zeros(sh)  # noqa: E731
        self.c = {m: {k: z(3, 8) for k in ("sd", "sad", "sdd", "sx", "sy", "sxx", "syy", "sxy")} | {"n": z(3), "sam": np.zeros((3, NSAM)), "samsum": z(3)}
                  for m in MODELS}

    def add_source(self, s, x, m):
        x = x.reshape(8, -1)
        m = m.reshape(-1)
        for c in range(3):
            sel = m == c
            if not sel.any():
                continue
            xs = x[:, sel].astype(np.float64)
            self.n[s][c] += xs.shape[1]
            self.s[s][c] += xs.sum(1)
            self.ss[s][c] += xs @ xs.T
            idx = np.clip((xs / BMAX * NB).astype(np.int64), 0, NB - 1)
            for ch in range(8):
                self.hist[s][c, ch] += np.bincount(idx[ch], minlength=NB)

    def add_consistency(self, mdl, x, y, m):
        x = x.reshape(8, -1).astype(np.float64)
        y = y.reshape(8, -1).astype(np.float64)
        m = m.reshape(-1)
        a = self.c[mdl]
        for c in range(3):
            sel = m == c
            if not sel.any():
                continue
            xs, ys = x[:, sel], y[:, sel]
            d = ys - xs
            a["n"][c] += xs.shape[1]
            a["sd"][c] += d.sum(1)
            a["sad"][c] += np.abs(d).sum(1)
            a["sdd"][c] += (d * d).sum(1)
            a["sx"][c] += xs.sum(1)
            a["sy"][c] += ys.sum(1)
            a["sxx"][c] += (xs * xs).sum(1)
            a["syy"][c] += (ys * ys).sum(1)
            a["sxy"][c] += (xs * ys).sum(1)
            cos = (xs * ys).sum(0) / (np.sqrt((xs * xs).sum(0)) * np.sqrt((ys * ys).sum(0)) + 1e-12)
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            a["samsum"][c] += ang.sum()
            a["sam"][c] += np.bincount(np.clip((ang / SAMMAX * NSAM).astype(np.int64), 0, NSAM - 1), minlength=NSAM)


def load(split, stem):
    raw = tff.imread(D / "cloudy_split" / split / f"{stem}.tif")[SRC_IDX].astype(np.float32) / 10000
    m256 = np.array(Image.open(D / "cloudy_split" / split / f"{stem}.png"))
    m512 = np.array(Image.open(D / "cloudy_split_masks512" / split / f"{stem}.png"))
    sr = {k: tff.imread(ROOT[k] / split / f"{stem}.tif")[WL].astype(np.float32) / 10000 for k in MODELS}
    return raw, np.clip(raw, 0, 0.5), m256, m512, sr


def down2(x):
    return x.reshape(8, 256, 2, 256, 2).mean((2, 4))


# ---------------------------------------------------------------- pass 1: statistics + pixel samples
t0 = time.time()
acc = {"all": Acc(), "test": Acc()}
rng = np.random.default_rng(0)
S_TRAIN, S_TEST = 600, 800
samp = {"train": {s: ([], []) for s in SOURCES}, "test": {s: ([], []) for s in SOURCES}}
n_patch = {"train": 0, "val": 0, "test": 0}
for split in ("train", "val", "test"):
    stems = sorted(f.stem for f in (D / "cloudy_split" / split).glob("*.tif"))[::STEP]
    for i, stem in enumerate(stems):
        raw, orig, m256, m512, sr = load(split, stem)
        views = {"orig_raw": (raw, m256), "orig": (orig, m256), "ediffsr": (sr["ediffsr"], m512), "esrgan": (sr["esrgan"], m512)}
        for g in ("all", "test") if split == "test" else ("all",):
            for s, (x, m) in views.items():
                acc[g].add_source(s, x, m)
            for mdl in MODELS:
                acc[g].add_consistency(mdl, orig, down2(sr[mdl]), m256)
        if split in samp:
            k = S_TRAIN if split == "train" else S_TEST
            for s, (x, m) in views.items():
                idx = rng.integers(0, m.size, k)
                samp[split][s][0].append(x.reshape(8, -1)[:, idx].T)
                samp[split][s][1].append(m.reshape(-1)[idx])
        n_patch[split] += 1
        if i % 100 == 0:
            print(f"{split} {i}/{len(stems)}  {time.time() - t0:.0f}s", flush=True)
print("patches used:", n_patch, f"({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- derived metrics
def mean_cov(a, s, c):
    n = a.n[s][c]
    mu = a.s[s][c] / n
    cov = a.ss[s][c] / n - np.outer(mu, mu)
    return mu, cov, n


def corr_of(cov):
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def corr_all_classes(a, s):
    n = a.n[s].sum()
    mu = a.s[s].sum(0) / n
    cov = a.ss[s].sum(0) / n - np.outer(mu, mu)
    return mu, cov


def consistency_metrics(a, mdl):
    c = a.c[mdl]
    n = c["n"][:, None]
    mx, my = c["sx"] / n, c["sy"] / n
    bias = c["sd"] / n
    rmse = np.sqrt(c["sdd"] / n)
    mae = c["sad"] / n
    vx = c["sxx"] / n - mx ** 2
    vy = c["syy"] / n - my ** 2
    r = (c["sxy"] / n - mx * my) / np.sqrt(vx * vy)
    return {"mean_orig": mx, "bias": bias, "rel_bias_pct": 100 * bias / mx, "rmse": rmse, "rel_rmse_pct": 100 * rmse / mx,
            "mae": mae, "pearson": r, "sam_mean": c["samsum"] / c["n"], "sam_hist": c["sam"], "n": c["n"]}


M = {g: {mdl: consistency_metrics(acc[g], mdl) for mdl in MODELS} for g in acc}


def sam_pct(h, p):
    cdf = np.cumsum(h) / h.sum()
    return float(np.searchsorted(cdf, p) * SAMMAX / NSAM)


# text report
rep = []
for g in ("all", "test"):
    rep.append(f"\n===== spectral consistency, group '{g}' (SR averaged 2x2 vs clipped original, 256 grid) =====")
    for mdl in MODELS:
        m = M[g][mdl]
        rep.append(f"\n-- {LAB[mdl]}")
        rep.append("class   pixels      SAM mean/p95 (deg) | rel. bias % per band " + " ".join(f"{b:>6}" for b in BANDS))
        for c in range(3):
            rep.append(f"{CN[c]:7s} {int(m['n'][c]):9d}   {m['sam_mean'][c]:5.2f} / {sam_pct(m['sam_hist'][c], 0.95):5.2f}     | "
                       + " " * 20 + " ".join(f"{v:6.2f}" for v in m["rel_bias_pct"][c]))
        rep.append("        rel. RMSE % per band: ")
        for c in range(3):
            rep.append(f"  {CN[c]:7s} " + " ".join(f"{v:6.2f}" for v in m["rel_rmse_pct"][c]))
        rep.append("        Pearson r(SR_down, orig) per band: ")
        for c in range(3):
            rep.append(f"  {CN[c]:7s} " + " ".join(f"{v:6.4f}" for v in m["pearson"][c]))

# csv
with open(OUT / "spectral_metrics.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "model", "class", "band", "mean_orig", "bias", "rel_bias_pct", "rmse", "rel_rmse_pct", "mae", "pearson", "sam_mean_deg"])
    for g in M:
        for mdl in MODELS:
            m = M[g][mdl]
            for c in range(3):
                for b in range(8):
                    w.writerow([g, mdl, CN[c], BANDS[b], f"{m['mean_orig'][c, b]:.5f}", f"{m['bias'][c, b]:.5f}", f"{m['rel_bias_pct'][c, b]:.3f}",
                                f"{m['rmse'][c, b]:.5f}", f"{m['rel_rmse_pct'][c, b]:.3f}", f"{m['mae'][c, b]:.5f}", f"{m['pearson'][c, b]:.5f}",
                                f"{m['sam_mean'][c]:.4f}"])

# ---------------------------------------------------------------- figures
A = acc["all"]
edges = np.linspace(0, BMAX, NB + 1)
ctr = (edges[:-1] + edges[1:]) / 2
bw = edges[1] - edges[0]


def dens(h):
    return h / max(h.sum(), 1) / bw


# 1) histograms, all pixels
fig, axs = plt.subplots(2, 4, figsize=(18, 8))
for ch in range(8):
    ax = axs.flat[ch]
    for s in SOURCES:
        ax.plot(ctr, dens(A.hist[s][:, ch].sum(0)), color=COL[s], lw=1.3, ls="--" if s == "orig_raw" else "-", label=LAB[s])
    ax.set_yscale("log"); ax.set_xlim(0, 0.8); ax.set_title(BANDS[ch]); ax.set_xlabel("reflectance"); ax.grid(alpha=.3)
axs.flat[0].legend(fontsize=8)
fig.suptitle("Pixel value distributions per band (all pixels, log density)")
fig.tight_layout(); fig.savefig(OUT / "1_hist_all_pixels.png", dpi=110); plt.close(fig)

# 2) histograms by class
fig, axs = plt.subplots(3, 8, figsize=(30, 10))
for c in range(3):
    for ch in range(8):
        ax = axs[c, ch]
        for s in SOURCES:
            ax.plot(ctr, dens(A.hist[s][c, ch]), color=COL[s], lw=1.1, ls="--" if s == "orig_raw" else "-", label=LAB[s])
        ax.set_xlim(0, 0.8); ax.grid(alpha=.3)
        if c == 0:
            ax.set_title(BANDS[ch])
        if ch == 0:
            ax.set_ylabel(f"{CN[c]}\ndensity")
axs[0, 0].legend(fontsize=7)
fig.suptitle("Pixel value distributions per band and class (linear density)")
fig.tight_layout(); fig.savefig(OUT / "2_hist_by_class.png", dpi=100); plt.close(fig)

# 3) mean spectra by class
fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), sharey=False)
for c in range(3):
    ax = axs[c]
    for s in SOURCES:
        mu, cov, _ = mean_cov(A, s, c)
        ax.errorbar(range(8), mu, yerr=np.sqrt(np.diag(cov)), color=COL[s], marker="o", ms=4, capsize=2, lw=1.3,
                    ls="--" if s == "orig_raw" else "-", label=LAB[s], alpha=.9)
    ax.set_xticks(range(8)); ax.set_xticklabels(BANDS); ax.set_title(f"{CN[c]} (n={int(A.n['orig'][c]):,} px @256)"); ax.grid(alpha=.3)
axs[0].set_ylabel("reflectance (mean ± std)"); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "3_mean_spectra_by_class.png", dpi=110); plt.close(fig)


# 4/5) correlation heatmaps
def heat(ax, C, title, vmin=-1, vmax=1, cmap="RdBu_r", fmt="{:.2f}"):
    im = ax.imshow(C, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(8)); ax.set_yticks(range(8)); ax.set_xticklabels(BANDS, fontsize=7); ax.set_yticklabels(BANDS, fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, fmt.format(C[i, j]), ha="center", va="center", fontsize=5.5, color="k")
    return im


def corr_row(axs, getter, tag):
    base = corr_of(getter("orig"))
    heat(axs[0], base, f"{tag} original (clipped)")
    for k, s in enumerate(MODELS):
        heat(axs[1 + k], corr_of(getter(s)), f"{tag} {LAB[s]}")
    for k, s in enumerate(MODELS):
        d = corr_of(getter(s)) - base
        heat(axs[3 + k], d, f"Δ corr ({LAB[s]} − orig)", vmin=-.2, vmax=.2)


fig, axs = plt.subplots(1, 5, figsize=(24, 5))
corr_row(axs, lambda s: corr_all_classes(A, s)[1], "all pixels:")
fig.tight_layout(); fig.savefig(OUT / "4_corr_heatmaps_all.png", dpi=110); plt.close(fig)

fig, axs = plt.subplots(3, 5, figsize=(24, 15))
for c in range(3):
    corr_row(axs[c], lambda s, c=c: mean_cov(A, s, c)[1], f"{CN[c]}:")
fig.tight_layout(); fig.savefig(OUT / "5_corr_heatmaps_by_class.png", dpi=100); plt.close(fig)

# 6) distortion heatmaps
fig, axs = plt.subplots(2, 3, figsize=(20, 8))
for r, mdl in enumerate(MODELS):
    m = M["all"][mdl]
    for k, (key, ttl, lim, cmap) in enumerate([("rel_bias_pct", "relative bias, % of orig mean", 12, "RdBu_r"),
                                              ("rel_rmse_pct", "relative RMSE, % of orig mean", None, "viridis"),
                                              ("pearson", "Pearson r (SR averaged 2x2 vs orig)", None, "viridis")]):
        ax = axs[r, k]
        Z = m[key]
        kw = {"vmin": -lim, "vmax": lim} if lim else ({"vmin": Z.min(), "vmax": 1} if key == "pearson" else {})
        im = ax.imshow(Z, cmap=cmap, aspect="auto", **kw)
        ax.set_xticks(range(8)); ax.set_xticklabels(BANDS); ax.set_yticks(range(3)); ax.set_yticklabels(CN)
        ax.set_title(f"{LAB[mdl]}: {ttl}", fontsize=10)
        for i in range(3):
            for j in range(8):
                ax.text(j, i, f"{Z[i, j]:.3f}" if key == "pearson" else f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=9, color="w" if cmap == "viridis" else "k")
        fig.colorbar(im, ax=ax, fraction=.03)
fig.tight_layout(); fig.savefig(OUT / "6_spectral_distortion_heatmaps.png", dpi=110); plt.close(fig)

# 7) SAM distribution
fig, axs = plt.subplots(1, 3, figsize=(17, 4.5))
xs = (np.arange(NSAM) + .5) * SAMMAX / NSAM
for c in range(3):
    for mdl in MODELS:
        h = M["all"][mdl]["sam_hist"][c]
        axs[c].plot(xs, h / h.sum() / (SAMMAX / NSAM), color=COL[mdl], label=f"{LAB[mdl]} (mean {M['all'][mdl]['sam_mean'][c]:.2f}°)")
    axs[c].set_yscale("log"); axs[c].set_xlim(0, 15); axs[c].set_title(f"SAM between SR (2x2 avg) and original, {CN[c]}")
    axs[c].set_xlabel("spectral angle, degrees"); axs[c].grid(alpha=.3); axs[c].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "7_sam_distribution.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- pass 2: pixel classifier (segmentation proxy)
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

Xtr = {s: np.concatenate(samp["train"][s][0]) for s in SOURCES}
ytr = {s: np.concatenate(samp["train"][s][1]) for s in SOURCES}
Xte = {s: np.concatenate(samp["test"][s][0]) for s in SOURCES}
yte = {s: np.concatenate(samp["test"][s][1]) for s in SOURCES}
print("classifier samples: train", len(ytr["orig"]), "test", len(yte["orig"]), "class share test:",
      np.round(np.bincount(yte["orig"], minlength=3) / len(yte["orig"]), 4), flush=True)
clf = {}
for s in SOURCES:
    cnt = np.bincount(ytr[s], minlength=3)
    w = (len(ytr[s]) / (3 * np.maximum(cnt, 1)))[ytr[s]]
    clf[s] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtr[s], ytr[s], sample_weight=w)
    print("trained on", s, flush=True)


def iou(y, p):
    out = []
    for c in range(3):
        i = np.sum((y == c) & (p == c))
        u = np.sum((y == c) | (p == c))
        out.append(i / u if u else np.nan)
    return np.array(out)


R = np.zeros((4, 4, 3))  # train source x test source x class IoU
for i, a in enumerate(SOURCES):
    for j, b in enumerate(SOURCES):
        R[i, j] = iou(yte[b], clf[a].predict(Xte[b]))
rep.append("\n===== pixel classifier (segmentation proxy), IoU on test split; rows = trained on, columns = tested on =====")
for name, Z in (("macro IoU", R.mean(2)), ("cloud IoU", R[:, :, 1]), ("shadow IoU", R[:, :, 2]), ("clear IoU", R[:, :, 0])):
    rep.append(f"\n{name}\n{'':24s}" + " ".join(f"{s:>10s}" for s in SOURCES))
    for i, a in enumerate(SOURCES):
        rep.append(f"{a:24s}" + " ".join(f"{Z[i, j]:10.4f}" for j in range(4)))
with open(OUT / "classifier_iou.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["train_on", "test_on", "iou_clear", "iou_cloud", "iou_shadow", "macro_iou"])
    for i, a in enumerate(SOURCES):
        for j, b in enumerate(SOURCES):
            w.writerow([a, b, *[f"{v:.4f}" for v in R[i, j]], f"{R[i, j].mean():.4f}"])

fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for ax, (ttl, Z) in zip(axs, (("macro IoU", R.mean(2)), ("cloud IoU", R[:, :, 1]), ("shadow IoU", R[:, :, 2]))):
    im = ax.imshow(Z, vmin=max(0, np.nanmin(Z) - .05), vmax=1, cmap="viridis")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(SOURCES, rotation=20); ax.set_yticklabels(SOURCES)
    ax.set_xlabel("tested on"); ax.set_ylabel("trained on"); ax.set_title(f"pixel classifier, {ttl} (test split)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center", color="w", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.04)
fig.tight_layout(); fig.savefig(OUT / "8_classifier_iou.png", dpi=110); plt.close(fig)

(OUT / "report.txt").write_text("\n".join(rep))
print("\n".join(rep))
print(f"done in {time.time() - t0:.0f}s; figures in {OUT}")
