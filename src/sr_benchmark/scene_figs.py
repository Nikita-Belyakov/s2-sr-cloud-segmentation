"""Figures 2 and 3 of the manuscript with ESRGAN added: the same Sen2VENuS validation scenes as in the draft (rows 0, 1 and 3 of valid.csv).
Rows / columns: Sentinel-2 (bicubic), EDiffSR (8-ch), ESRGAN (8-ch, net(x/255)), VENuS.   Stages: ediffsr (landsat env), esrgan (satlas venv), plot (landsat env)."""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

D = WORK_DIR
S2V = S2V_NPZ
OUT = D / "scene_figs"
OUT.mkdir(exist_ok=True)
SCENES = ["ALSACE_C_32ULU_2018-10-22_384.npz",     # valid.csv row 1  (draft: patch 1, brown fields)
          "SO1_C_V2_31TCH_2020-07-19_387.npz",     # row 3            (draft: patch 3, village)
          "SO1_C_V2_31TCJ_2020-04-10_265.npz"]     # row 0            (draft: patch 0, pond; also the scene of the band-wise figure)
p = argparse.ArgumentParser()
p.add_argument("--stage", choices=["ediffsr", "esrgan", "plot"], required=True)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def load(f):
    z = np.load(S2V / f)
    c = lambda k: torch.clip(torch.from_numpy(z[k]).float() / 10000, 0, 0.5)  # noqa: E731
    s20, s10, v20, v10 = c("sentinel_20m"), c("sentinel_10m"), c("venus_5m_20m"), c("venus_5m_10m")
    lr = torch.cat([F.interpolate(s20[None], size=(128, 128), mode="bicubic", align_corners=False)[0], s10], 0)   # [B5,B6,B7,B8A,B2,B3,B4,B8]
    return lr, torch.cat([v20, v10], 0)


LRS = torch.stack([load(f)[0] for f in SCENES])
GTS = torch.stack([load(f)[1] for f in SCENES])

if a.stage == "ediffsr":
    from s2sr.ediffsr import EDiffSR
    m = EDiffSR(device=dev, img_channel=8, model_path=str(WEIGHTS["ediffsr_8ch"]))
    up = F.interpolate(LRS, scale_factor=2, mode="bicubic", align_corners=False)
    sr = m.predict_batch(((up - 0.5) * 2.0).to(dev))
    np.save(OUT / "ediffsr.npy", (sr / 2.0 + 0.5).float().cpu().clamp(0, 0.5).numpy())
    print("EDIFFSR_DONE")
elif a.stage == "esrgan":
    from s2sr.esrgan_net import SSR_RRDBNet
    net = SSR_RRDBNet(num_in_ch=8, num_out_ch=8, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    net.load_state_dict(torch.load(str(WEIGHTS["esrgan_8ch"]), map_location="cpu")["params_ema"], strict=True)
    net = net.to(dev).eval()
    with torch.no_grad():
        o = F.avg_pool2d(torch.clamp(net(LRS.to(dev) / 255), 0, 1), 2).float().cpu().clamp(0, 0.5)
    np.save(OUT / "esrgan.npy", o.numpy())
    print("ESRGAN_DONE")
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ed, es = np.load(OUT / "ediffsr.npy"), np.load(OUT / "esrgan.npy")
    s2 = F.interpolate(LRS, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 0.5).numpy()
    gt = GTS.numpy()
    ROWS = [("Sentinel-2", s2), ("EDiffSR", ed), ("ESRGAN", es), ("VENµS", gt)]
    plt.rcParams.update({"font.size": 8})

    def rgb(x, gain=3.0):
        return np.clip(np.stack([x[6], x[5], x[4]], -1) * gain, 0, 1)

    def ir(x, gain=2.0):
        return np.clip(np.stack([x[2], x[1], x[0]], -1) * gain, 0, 1)    # B7, B6, B5

    # ---- Figure 2: band-wise, the pond scene
    s = 2
    names = ["B2 (blue)", "B3 (green)", "B4 (red)", "B8 (NIR)", "B5", "B6", "B7", "B8A"]
    chan = [4, 5, 6, 7, 0, 1, 2, 3]
    fig, axs = plt.subplots(4, 8, figsize=(5.6, 3.1))
    for c, ch in enumerate(chan):
        lo, hi = np.percentile(gt[s, ch], [1, 99])
        for r, (lab, arr) in enumerate(ROWS):
            ax = axs[r, c]
            ax.imshow(arr[s, ch], cmap="plasma", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(names[c], fontsize=7.5, pad=2)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.995, top=0.93, bottom=0.01, wspace=0.04, hspace=0.04)
    fig.savefig(OUT / "bandwise_scene.png", dpi=300)
    plt.close(fig)

    # ---- Figure 3: RGB (rows 1-3) and IR (rows 4-6) composites of three scenes, 4 columns = Sentinel-2, EDiffSR, ESRGAN, VENuS
    fig, axs = plt.subplots(6, 4, figsize=(5.4, 7.5))
    for blk, fn in enumerate((rgb, ir)):
        for sc in range(3):
            for c in range(4):
                ax = axs[3 * blk + sc, c]
                ax.imshow(fn(ROWS[c][1][sc]), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if sc == 0 and blk == 0:
                    ax.set_title(ROWS[c][0], fontsize=9, pad=3)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.965, bottom=0.005, wspace=0.03, hspace=0.04)
    for blk, ttl in enumerate(("RGB composite (B4, B3, B2)", "IR composite (B7, B6, B5)")):
        y0 = axs[3 * blk + 2, 0].get_position().y0
        y1 = axs[3 * blk, 0].get_position().y1
        fig.text(0.018, (y0 + y1) / 2, ttl, rotation=90, ha="center", va="center", fontsize=9, fontweight="bold")
    fig.savefig(OUT / "composites_scenes.png", dpi=300)
    plt.close(fig)
    print("PLOT_DONE")
