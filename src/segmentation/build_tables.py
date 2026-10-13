"""LaTeX tables 3-10 and A.1 of the paper from the markdown reports of the segmentation experiments
(seg_cv_report/cv_report.md, seg_small_report/small_report.md, seg_thin_report/thin_report.md) and the ERQA summaries
(erqa_s2v/erqa_s2v_summary.md, erqa_sr5m/erqa_summary.md) found in WORK_DIR; the tables are written to WORK_DIR/tables.

The standard deviation is put under the mean (small font); arrows in the headers show how a metric is read; the best values are in bold.
"""
import re
import sys
from pathlib import Path

_sys = sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from s2sr.paths import WORK_DIR  # noqa: E402

ROOT = WORK_DIR
OUT = WORK_DIR / "tables"
OUT.mkdir(exist_ok=True)


def read_lines(rel):
    return (ROOT / rel).read_text(encoding="utf-8").splitlines()


def col(head, name):
    for k, h in enumerate(head):
        if h == name:
            return k
    raise KeyError(name)


def find_table(lines, heading, nth=0, after=0, section=None):
    """first markdown table after the nth heading line that contains `heading` (searched from line index `after`)"""
    hit, seen = -1, 0
    for i in range(after, len(lines)):
        if lines[i].startswith("#") and heading in lines[i]:
            if seen == nth:
                hit = i
                break
            seen += 1
    if hit < 0:
        raise KeyError(heading)
    j = hit + 1
    while j < len(lines) and not lines[j].startswith("|"):
        j += 1
    rows = []
    while j < len(lines) and lines[j].startswith("|"):
        rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
        j += 1
    return rows[0], rows[2:], hit


UP, DOWN = r"$\uparrow$", r"$\downarrow$"
NAME = {
    "Original 8ch (256)": "Original (256)",
    "Bicubic x2 8ch (512)": r"Bicubic $\times2$",
    "EDiffSR 8ch (512)": "EDiffSR",
    "ESRGAN 8ch (512)": "ESRGAN",
    "Orig + EDiffSR 16ch (512)": r"Orig. + EDiffSR (16 ch)",
    "Orig + ESRGAN 16ch (512)": r"Orig. + ESRGAN (16 ch)",
    "Bicubic + (EDiffSR - Bicubic) 16ch (512)": r"Bicubic + (EDiffSR $-$ bicubic) (16 ch)",
    "Ensemble: Bicubic + EDiffSR models": r"Ensemble: bicubic + EDiffSR",
    "Ensemble: Bicubic + ESRGAN models": r"Ensemble: bicubic + ESRGAN",
    "Ensemble: 2 Bicubic models": r"Ensemble: 2 bicubic",
}


def name(s):
    s = s.strip().replace("−", "-")
    return NAME.get(s, s)


def mean_of(c):
    m = re.match(r"^\s*([+\-]?[\d.]+)", c.strip().replace("−", "-"))
    return float(m.group(1)) if m else None


def fmt(cell, bold=False):
    """'0.475 ± 0.182' -> mean over std (small); '+0.009 ± 0.020 (6/8, p=0.26)' -> three-part cell"""
    c = cell.strip().replace("−", "-")
    m = re.match(r"^([+\-]?[\d.]+)\s*±\s*([\d.]+)\s*(\(([^)]*)\))?$", c)
    if not m:
        return c
    main = (r"\mathbf{" + m.group(1) + "}") if bold else m.group(1)
    if m.group(4):
        info = m.group(4).replace("p=", r"$p{=}$")
        return r"\makecell{$" + main + r"\pm" + m.group(2) + r"$\\{\scriptsize(" + info + ")}}"
    return r"\makecell{$" + main + r"$\\{\scriptsize$\pm" + m.group(2) + "$}}"


def render(rows, higher=None, groups=None):
    """rows: [label, raw cell, ...]; the best value of every column (inside every group of rows) is set in bold.
    higher[j]: True if a larger value of column j is better, False if smaller, None: no bold."""
    n = len(rows[0]) - 1
    higher = higher or [True] * n
    groups = groups or [list(range(len(rows)))]
    bold = [[False] * n for _ in rows]
    for g in groups:
        for j in range(n):
            if higher[j] is None:
                continue
            vals = [(mean_of(rows[i][j + 1]), i) for i in g if mean_of(rows[i][j + 1]) is not None]
            if not vals:
                continue
            best = max(v for v, _ in vals) if higher[j] else min(v for v, _ in vals)
            for v, i in vals:
                if abs(v - best) < 1e-9:
                    bold[i][j] = True
    return [[r[0]] + [fmt(r[j + 1], bold[i][j]) for j in range(n)] for i, r in enumerate(rows)]


def head(*cols):
    return " & ".join(r"\makecell{" + c + "}" if r"\\" in c else c for c in cols)


def write(fname, caption, label, header, rows, colspec, extra="", sep="3pt"):
    lines = [r"\begin{table}[!t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{" + sep + "}",
             r"\caption{" + caption + "}", r"\label{" + label + "}", r"\vspace{3pt}",
             r"\begin{tabular}{" + colspec + "}", r"\toprule", header + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if extra:
        lines += [r"\par\smallskip{\footnotesize " + extra + "}"]
    lines += [r"\end{table}"]
    (OUT / fname).write_text("\n".join(lines) + "\n", encoding="utf8")


BOLD_NOTE = r" Arrows show how a metric is read ($\uparrow$: larger is better); the best value of a column is in bold."
RG = r">{\raggedright\arraybackslash}"
L1 = RG + r"p{4.2cm}"

# ------------------------------------------------------------------ cross-validation, pooled
L = read_lines("seg_cv_report/cv_report.md")
h, body, _ = find_table(L, "Pooled over all 8 scenes")
cols = ["IoU clear", "IoU shadow", "IoU cloud", "mIoU", "F1 shadow", "F1 cloud", "mF1"]
idx = [col(h, c) for c in cols]
raw = [[name(r[0])] + [r[k] for k in idx] for r in body]
write("t_cv_pooled.tex", r"Scene-level cross-validation, pooled over all 8 scenes (confusion matrices of the four folds summed, $256\times256$ grid). Mean (above) and standard deviation (below) over 2 seeds." + BOLD_NOTE,
      "tab:cv", head("Input", r"IoU" + UP + r"\\clear", r"IoU" + UP + r"\\shadow", r"IoU" + UP + r"\\cloud", "mIoU" + UP, r"F1" + UP + r"\\shadow", r"F1" + UP + r"\\cloud", "mF1" + UP),
      render(raw), L1 + "ccccccc")

# ------------------------------------------------------------------ boundary metrics and recall by object size
p3 = next(i for i, l in enumerate(L) if l.startswith("## Part 3"))
h, body, _ = find_table(L, "Boundary F-score", after=p3)
hc = find_table(L, "cloud: quality by object size", after=p3)
size = {r[0]: r for r in hc[1]}
cols = ["shadow BF τ=2", "cloud BF τ=2", "shadow boundary IoU", "cloud boundary IoU"]
idx = [col(h, c) for c in cols]
rk = [col(hc[0], c) for c in ("recall <50 px", "recall 50-500 px", "recall >=500 px")]
raw = [[name(r[0])] + [r[k] for k in idx] + [size[r[0]][k] for k in rk] for r in body]
write("t_boundary.tex", r"Boundary quality and recall of clouds by object size, scene-level cross-validation, pooled over all 8 scenes (mean above, standard deviation below, 2 seeds). BF: boundary F-score with a tolerance of 2 pixels (20\,m); bIoU: boundary IoU (band of 2 pixels); object size in pixels of the $256$ grid (1 pixel = 100\,m$^2$)." + BOLD_NOTE,
      "tab:boundary", head("Input", r"shadow" + r"\\BF" + UP, r"cloud" + r"\\BF" + UP, r"shadow" + r"\\bIoU" + UP, r"cloud" + r"\\bIoU" + UP,
                           r"recall" + UP + r"\\$<$50\,px", r"recall" + UP + r"\\50--500\,px", r"recall" + UP + r"\\$\geq$500\,px"),
      render(raw), L1 + "ccccccc", sep="2.4pt")

# ------------------------------------------------------------------ small clouds
S = read_lines("seg_small_report/small_report.md")
fx_h, fx_b, _ = find_table(S, "Fixed split (1075 test patches), mean", nth=0)
cv_h, cv_b, _ = find_table(S, "Cross-validation, pooled over all 8 scenes", nth=0)
cv = {r[0]: r for r in cv_b}
cols = ["IoU", "F1", "precision", "recall"]
raw = [[name(r[0])] + [r[col(fx_h, c)] for c in cols] + [cv[r[0]][col(cv_h, c)] for c in cols] for r in fx_b]
hdr = (r"Input & \multicolumn{4}{c}{Fixed split} & \multicolumn{4}{c}{Cross-validation} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9} & IoU" + UP + " & F1" + UP +
       " & prec." + UP + " & recall" + UP + " & IoU" + UP + " & F1" + UP + " & prec." + UP + " & recall" + UP)
write("t_small.tex", r"Small clouds (connected components of fewer than 50 pixels, i.e. below 0.5\,ha) in the region of small clouds of the ground truth or of the prediction: IoU, F1, precision and recall of the class cloud inside the region. "
      r"Fixed split (3 seeds) and scene-level cross-validation (pooled over 8 scenes, 2 seeds); mean above, standard deviation below." + BOLD_NOTE,
      "tab:small", hdr, render(raw), RG + r"p{3.1cm}cccccccc", sep="2.2pt")

# ------------------------------------------------------------------ thin clouds
T = read_lines("seg_thin_report/thin_report.md")
blocks = {}
for key, sec in (("thin", "## Thin clouds"), ("small_thin", "## Small thin clouds"), ("mid", "## Intermediate clouds"), ("dense", "## Dense clouds")):
    start = next(i for i, l in enumerate(T) if l.startswith(sec))
    fh, fb, _ = find_table(T, "Fixed split", after=start)
    ch, cb, _ = find_table(T, "Cross-validation, pooled", after=start)
    blocks[key] = (fh, {r[0]: r for r in fb}, ch, {r[0]: r for r in cb}, [r[0] for r in fb])
setups = blocks["thin"][4]
raw = []
for s in setups:
    cells = [name(s)]
    for key in ("thin", "small_thin"):
        fh, fb, ch, cb, _ = blocks[key]
        cells += [fb[s][col(fh, "IoU")], fb[s][col(fh, "F1")], cb[s][col(ch, "IoU")], cb[s][col(ch, "F1")]]
    raw.append(cells)
hdr = (r"Input & \multicolumn{4}{c}{Thin clouds ($<0.15$)} & \multicolumn{4}{c}{Small and thin ($<50$\,px)} \\ \cmidrule(lr){2-5}\cmidrule(lr){6-9} "
       r"& \multicolumn{2}{c}{Fixed} & \multicolumn{2}{c}{CV} & \multicolumn{2}{c}{Fixed} & \multicolumn{2}{c}{CV} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9} "
       r"& IoU" + UP + " & F1" + UP + " & IoU" + UP + " & F1" + UP + " & IoU" + UP + " & F1" + UP + " & IoU" + UP + " & F1" + UP)
write("t_thin.tex", r"Thin clouds: cloud objects of mean visible reflectance (B2, B3, B4) below 0.15, of any size and of fewer than 50 pixels. IoU and F1 of the class cloud in the region of these objects "
      r"(mean above, standard deviation below; fixed split 3 seeds, cross-validation 2 seeds)." + BOLD_NOTE,
      "tab:thin", hdr, render(raw), RG + r"p{3.1cm}cccccccc", sep="2.2pt")

# ------------------------------------------------------------------ brightness groups
raw, groups, labels = [], [], []
for key, nm in (("thin", r"Thin ($<0.15$)"), ("small_thin", "Small and thin"), ("mid", "Intermediate (0.15--0.30)"), ("dense", r"Dense ($\geq0.30$)")):
    fh, fb, ch, cb, _ = blocks[key]
    g = []
    for j, (s_, lab) in enumerate((("Bicubic x2 8ch (512)", r"Bicubic $\times2$"), ("Orig + EDiffSR 16ch (512)", r"Orig. + EDiffSR (16 ch)"),
                                   ("Orig + ESRGAN 16ch (512)", r"Orig. + ESRGAN (16 ch)"))):
        g.append(len(raw))
        raw.append([lab, fb[s_][col(fh, "IoU")], fb[s_][col(fh, "F1")], cb[s_][col(ch, "IoU")], cb[s_][col(ch, "F1")]])
        labels.append(nm if j == 0 else "")
    groups.append(g)
rows = render(raw, groups=groups)
final = [[labels[i]] + r for i, r in enumerate(rows)]
write("t_brightness.tex", "Bicubic input and the concatenations of the original with the SR result, by brightness of the cloud object: IoU and F1 of the class cloud in the region of the objects of the group "
      "(mean above, standard deviation below)." + BOLD_NOTE.replace("of a column", "of a column inside a group"),
      "tab:brightness", head("Cloud objects", "Input", r"IoU" + UP + r"\\(fixed)", r"F1" + UP + r"\\(fixed)", r"IoU" + UP + r"\\(CV)", r"F1" + UP + r"\\(CV)"), final,
      RG + r"p{3cm}" + RG + r"p{3.8cm}cccc")


# ------------------------------------------------------------------ paired differences, small / thin groups
def collect(lines, names):
    out, sec, proto = [], None, None
    for l in lines:
        if l.startswith("## "):
            sec = next((v for k, v in names if l.startswith(k)), None)
            proto = None
        elif l.startswith("###"):
            if "ixed" in l:
                proto = "Fixed"
            elif "per run" in l or "mean ± std over runs" in l:
                proto = "CV"
            else:
                proto = None
        elif sec and proto and l.startswith("| Orig + ") and " − Bicubic x2" in l:
            out.append((sec, proto, [c.strip() for c in l.strip().strip("|").split("|")]))
    return out


got = collect(S, [("## Small clouds", r"Small clouds ($<50$\,px)")]) + collect(T, [("## Thin clouds", "Thin clouds"), ("## Small thin clouds", "Small and thin clouds"),
                                                                                  ("## Intermediate clouds", "Intermediate clouds"), ("## Dense clouds", "Dense clouds")])
secs = []
for sec, proto, c in got:
    if not secs or secs[-1][0] != sec:
        secs.append((sec, []))
    secs[-1][1].append((proto, "EDiffSR" if "EDiffSR" in c[0] else "ESRGAN", c))


def pairs_table(fname, label, caption, selection):
    raw, meta, groups = [], [], []
    for sec, items in selection:
        g = []
        for k, (proto, sr, c) in enumerate(items):
            g.append(len(raw))
            raw.append([sec, c[1], c[3], c[4]])
            meta.append([sec if k == 0 else "", proto, sr])
        groups.append(g)
    rows = render(raw, groups=groups)
    final = [[meta[i][0], meta[i][1], meta[i][2]] + r[1:] for i, r in enumerate(rows)]
    write(fname, caption, label, head("Cloud objects", "Protocol", "SR", r"$\Delta$IoU" + UP, r"$\Delta$precision" + UP, r"$\Delta$recall" + UP), final, RG + r"p{2.5cm}llccc", sep="2pt")


cap = (r"Orig.\,+\,SR (16 channels) minus bicubic input (8 channels), paired by run (Fixed: 3 seeds; CV: 8 runs = 4 folds $\times$ 2 seeds): "
       r"mean $\pm$ std of the difference, in brackets the number of runs with a positive difference and the paired $t$-test $p$-value (descriptive). "
       r"A positive difference favours the SR input ($\uparrow$); the largest difference of a column inside a group is in bold.")
pairs_table("t_pairs_a.tex", "tab:pairs", cap + r" Small and thin clouds.", secs[:3])
pairs_table("t_pairs_b.tex", "tab:pairsb", r"Same as Table~\ref{tab:pairs} for intermediate and dense clouds.", secs[3:])
(OUT / "t_pairs.tex").write_text(r"\input{tables/t_pairs_a}" + "\n" + r"\input{tables/t_pairs_b}" + "\n", encoding="utf8")


# ------------------------------------------------------------------ ERQA tables (the standard deviation stays on the line)
def inline(raw_, higher):
    n = len(raw_[0]) - 1
    best = []
    for j in range(n):
        vals = [mean_of(r[j + 1]) for r in raw_]
        best.append(max(vals) if higher[j] else min(vals))
    out = []
    for r in raw_:
        cells = []
        for j in range(n):
            c = r[j + 1].replace("±", r"\pm").replace("−", "-")
            m = re.match(r"^([+\-]?[\d.]+)(.*)$", c.strip())
            main, rest = m.group(1), m.group(2)
            if abs(float(main) - best[j]) < 1e-9:
                main = r"\mathbf{" + main + "}"
            cells.append("$" + main + rest.replace(" ", "") + "$")
        out.append([r[0]] + cells)
    return out


E = read_lines("erqa_s2v/erqa_s2v_summary.md")
h1, b1, i1 = find_table(E, "All scenes")
rgb = [r for r in b1 if r[0] == "rgb"]
h2, b2, _ = find_table(E, "Pixel-wise fidelity", after=i1)
fid = {r[0]: r for r in b2}
raw = []
for r in rgb:
    f = fid[r[1]]
    raw.append([r[1], r[col(h1, "ERQA v1.0")], r[col(h1, "precision")], r[col(h1, "recall")], f[col(h2, "PSNR, dB")], f[col(h2, "SAM, deg")]])
write("t_erqa_s2v.tex", r"ERQA of the 5\,m results against the real \venus{} 5\,m reference on Sen2VEN$\mu$S (12 scenes, 480 patches of the validation list; true-colour composite, 431 patches with a non-degenerate reference). "
      r"Precision: share of the edge pixels of the result that \venus{} also has; recall: share of the \venus{} edges that the result reproduces. "
      r"Arrows show how a metric is read; the best value of a column is in bold.",
      "tab:erqas2v", head("Method", "ERQA" + UP, "precision" + UP, "recall" + UP, "PSNR, dB" + UP, "SAM, deg" + DOWN), inline(raw, [True, True, True, True, False]), "lccccc")

E2 = read_lines("erqa_sr5m/erqa_summary.md")
h, b, i0 = find_table(E2, "Main: SR result 5 m")
rgbc = {r[1]: r for r in b if r[0] == "rgb"}
hc2, bc, _ = find_table(E2, "By class", after=i0)
byc = {}
for r in bc:
    byc.setdefault(r[1], {})[r[0]] = r
raw = []
for m in ("EDiffSR", "ESRGAN"):
    r = rgbc[m]
    raw.append([m, r[col(h, "ERQA v1.0")], r[col(h, "precision (SR edges that exist in the source)")], r[col(h, "recall (source edges kept)")]]
               + [byc[m][c][col(hc2, "F1")] for c in ("clear", "cloud", "shadow")] + [byc[m]["cloud"][col(hc2, "recall")]])
write("t_erqa_cloudy.tex", r"ERQA of the 5\,m results on the cloudy test patches (1075 patches, true colour) against the original 10\,m patch interpolated to the 5\,m grid; the last four columns are the edge F1 by the class of the pixel at the edge and the edge recall on clouds. "
      r"Arrows show how a metric is read; the best value of a column is in bold.",
      "tab:erqacloudy", head("Method", "ERQA" + UP, "precision" + UP, "recall" + UP, "F1 clear" + UP, "F1 cloud" + UP, "F1 shadow" + UP, r"recall cloud" + UP),
      inline(raw, [True] * 7), "lccccccc", sep="1.2pt")
print("ok")
