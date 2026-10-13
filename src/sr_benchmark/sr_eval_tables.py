"""LaTeX tables (tab:sr_performance, tab:band_performance) and paired statistics from sr_eval/<tag>_*.{json,csv,npz}."""
import sys as _sys
from pathlib import Path as _P

_sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from s2sr.paths import S2V_NPZ, S2V_SPLITS, WEIGHTS, WORK_DIR  # noqa: E402,F401
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

D = WORK_DIR / "sr_eval"
TAG = sys.argv[1] if len(sys.argv) > 1 else "main"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else D
S = json.load(open(D / f"{TAG}_summary.json"))
N = S["n"]
B = np.load(D / f"{TAG}_bands.npz")
# arithmetic mean of the metrics of the two 4-band models (as in the draft): LPIPS and FID are not defined for the IR bands
for pre in ("ediffsr", "esrgan"):
    S[f"{pre}_mean"] = {k: (S[f"{pre}_rgbn"][k] + S[f"{pre}_ir"][k]) / 2 for k in ("psnr", "ssim", "rmse", "sam_deg", "ergas")}
NL = "\n"


def fmt(v, nd, best):
    s = f"{v:.{nd}f}"
    return (r"\textbf{" + s + "}") if best else s


# (header, key, decimals, higher is better)
COLS = [(r"PSNR$\uparrow$", "psnr", 2, True), (r"SSIM$\uparrow$", "ssim", 4, True), (r"LPIPS$\downarrow$", "lpips", 4, False),
        (r"FID$\downarrow$", "fid64", 4, False), (r"RMSE$\downarrow$", "rmse", 4, False), (r"SAM$\downarrow$", "sam_deg", 2, False),
        (r"ERGAS$\downarrow$", "ergas", 2, False)]

A = [(r"EDiffSR, 8 ch (joint)", "ediffsr8", False),
     (r"EDiffSR, mean 4+4 ch", "ediffsr_mean", False),
     (r"ESRGAN, 8 ch (joint)", "esrgan", False),
     (r"ESRGAN, mean 4+4 ch", "esrgan_mean", False),
     (r"ESRGAN, test-script scaling", "esrgan_testscript", True),
     (r"Bicubic interpolation", "bicubic", False)]
BK = [(r"EDiffSR, RGBNir $2\times$", "ediffsr_rgbn", False), (r"ESRGAN, RGBNir $2\times$", "esrgan_rgbn", False), (r"Bicubic interpolation", "bicubic_rgbn", False)]
CK = [(r"EDiffSR, IR $4\times$", "ediffsr_ir", False), (r"ESRGAN, IR $4\times$", "esrgan_ir", False), (r"Bicubic interpolation", "bicubic_ir", False)]


def block(rows):
    cand = [k for _, k, it in rows if not it and not k.endswith('_mean')]
    best = {}
    for _, key, nd, hib in COLS:
        vals = [S[k][key] for k in cand if key in S[k]]
        best[key] = (max(vals) if hib else min(vals)) if vals else None
    lines = []
    for label, k, it in rows:
        cells = []
        for _, key, nd, hib in COLS:
            v = S[k].get(key)
            if v is None:
                cells.append("---")
            else:
                cells.append(fmt(v, nd, (not it) and (not k.endswith('_mean')) and best[key] is not None and abs(v - best[key]) < 10 ** (-nd) / 2))
        lab = (r"\textit{" + label + "}") if it else label
        lines.append(lab + " & " + " & ".join(cells) + r" \\")
    return lines


def table(rows8, fname, extra=""):
    ncol = len(COLS) + 1
    cap = (f"Super-resolution quality on {N} validation patches of Sen2VEN$\\mu$S, all methods scored by the same code on the same patches "
           r"(PSNR with a unit peak, SSIM with a Gaussian window of 11 pixels, SAM in degrees, ERGAS with ratios 2 and 4 for the $10$\,m and $20$\,m bands; "
           r"LPIPS and FID on the true-colour composite B4, B3, B2). ``Mean 4+4 ch'' is the arithmetic mean of the metric values of the RGBNir and IR models. "
           r"Arrows show how a metric is read ($\uparrow$: larger is better, $\downarrow$: smaller is better); the best value of each block is in bold (the ``mean'' rows are not compared)." + extra)
    lines = [r"\begin{table}[!t]", r"\centering", r"\fontsize{9}{11}\selectfont", r"\setlength{\tabcolsep}{2pt}",
             r"\caption{" + cap + "}", r"\label{tab:sr_performance}",
             r"\begin{tabular}{l" + "c" * len(COLS) + "}", r"\toprule",
             r"\textbf{Model and mode} & " + " & ".join(r"\textbf{" + c[0] + "}" for c in COLS) + r" \\", r"\midrule",
             r"\multicolumn{%d}{l}{\textit{8 bands}} \\" % ncol] + block(rows8) + [r"\midrule",
             r"\multicolumn{%d}{l}{\textit{4 bands of $10$\,m (B2, B3, B4, B8)}} \\" % ncol] + block(BK) + [r"\midrule",
             r"\multicolumn{%d}{l}{\textit{4 bands of $20$\,m (B5, B6, B7, B8A)}} \\" % ncol] + block(CK) + \
            [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / fname).write_text(NL.join(lines) + NL, encoding="utf8")


table([r for r in A if not r[2]], "t_sr_performance.tex")
table(A, "t_sr_performance_diagnostic.tex", " The italic row is the ESRGAN output obtained with the input scaling of the separate test script (diagnostic).")

# ---- band-wise table (order of the draft: B5, B6, B7, B8A, B, G, R, NIR)
names = ["Band 5", "Band 6", "Band 7", "Band 8A", "B", "G", "R", "NIR"]
meth = [("EDiffSR", "ediffsr8"), ("ESRGAN", "esrgan"), ("Bicubic", "bicubic")]
rows = []
for metric, nd, key in ((r"PSNR$\uparrow$", 2, "psnr_band"), (r"SSIM$\uparrow$", 4, "ssim_band")):
    vals = {lab: B[f"{k}_{key}"].mean(0) for lab, k in meth}
    best = np.max(np.stack(list(vals.values())), 0)
    for lab, _ in meth:
        cells = [fmt(vals[lab][c], nd, abs(vals[lab][c] - best[c]) < 10 ** (-nd) / 2) for c in range(8)]
        rows.append(f"{lab} & {metric} & " + " & ".join(cells) + r" \\")
    rows.append(r"\midrule")
rows = rows[:-1]
band = [r"\begin{table}[!t]", r"\centering", r"\fontsize{9}{11}\selectfont", r"\setlength{\tabcolsep}{2pt}",
        r"\caption{Band-wise super-resolution quality on the same " + str(N) + r" validation patches of Sen2VEN$\mu$S (the 8-channel models; the best value of each column and metric is in bold).}",
        r"\label{tab:band_performance}",
        r"\begin{tabular}{llcccccccc}", r"\toprule",
        r"\textbf{Model} & \textbf{Metric} & " + " & ".join(r"\textbf{" + n + "}" for n in names) + r" \\", r"\midrule"] + rows + \
       [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
(OUT / "t_sr_band.tex").write_text(NL.join(band) + NL, encoding="utf8")


# ---- paired statistics
def csv(g):
    return pd.read_csv(D / f"{TAG}_{g}_per_image.csv")


rep = [f"# Unified Sen2VENuS evaluation, N = {N}", ""]
for g in ["ediffsr8", "ediffsr_sep", "esrgan", "esrgan_sep", "esrgan_testscript", "bicubic", "ediffsr_rgbn", "esrgan_rgbn", "esrgan_rgbn_testscript", "bicubic_rgbn",
          "ediffsr_ir", "esrgan_ir", "esrgan_ir_testscript", "bicubic_ir"]:
    s = S[g]
    rep.append(f"{g:20s} " + "  ".join(f"{k}={s[k]:.4f}+-{s[k + '_sem']:.4f}" for k in ["psnr", "ssim", "rmse", "sam_deg", "ergas", "lpips"] if k in s)
               + (f"  fid64={s['fid64']:.4f}" if "fid64" in s else ""))
rep.append("")
for a, b in [("esrgan", "ediffsr8"), ("esrgan", "bicubic"), ("ediffsr8", "bicubic"), ("ediffsr_sep", "ediffsr8"), ("esrgan_sep", "esrgan"),
             ("ediffsr_rgbn", "bicubic_rgbn"), ("ediffsr_ir", "bicubic_ir"), ("esrgan_rgbn", "bicubic_rgbn"), ("esrgan_ir", "bicubic_ir"),
             ("esrgan_rgbn", "ediffsr_rgbn"), ("esrgan_ir", "ediffsr_ir"), ("esrgan_testscript", "bicubic"),
             ("esrgan_rgbn_testscript", "bicubic_rgbn"), ("esrgan_ir_testscript", "bicubic_ir")]:
    da, db = csv(a), csv(b)
    for m in ["psnr", "ssim", "sam_deg", "ergas"]:
        d = da[m].values - db[m].values
        t = stats.ttest_rel(da[m].values, db[m].values)
        wins = (d > 0).mean() if m in ("psnr", "ssim") else (d < 0).mean()
        rep.append(f"{a} - {b} | {m:8s} mean diff {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}  p={t.pvalue:.2g}  better in {100 * wins:.0f}% of patches")
    rep.append("")
(OUT / f"{TAG}_report.txt").write_text(NL.join(rep), encoding="utf8")
print("tables written to", OUT)
