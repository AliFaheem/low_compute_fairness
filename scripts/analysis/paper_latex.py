"""Write the paper's tables (LaTeX), figure data (pgfplots .dat) and raw result CSVs into the Overleaf project.

    uv run python scripts/analysis/paper_tables.py        # computes results/analysis/ (answers cache + CSVs)
    uv run python scripts/analysis/paper_latex.py          # -> paper/bias_icse_responsible_ai/{tables,figdata,results}

Uses the same answers cache, metrics and template bootstrap as paper_tables.py (docs/analysis_plan.md), so the
numbers match. Tables use booktabs; figures are drawn by pgfplots in main.tex from figdata/*.dat; the category
heatmap is a colortbl table (tables/f3_categories.tex). Colours follow the dataviz reference palette: blue / orange
/ aqua for en / ur / roman_ur, single-hue blue for levels, blue-grey-red diverging for changes.
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_tables as pt  # noqa: E402

PAPER = "paper/bias_icse_responsible_ai"
LANG_TEX = {"en": "English", "ur": "Urdu", "ur_abc": "Urdu (A/B/C)", "roman_ur": "Roman Urdu"}
VAR_TEX = {"bf16": "BF16", "int8": "INT8", "nf4": "NF4", "awq": "AWQ", "wanda25": "Wanda 25\\%",
           "wanda50": "Wanda 50\\%", "wanda25mix": "Wanda 25\\% (mix)", "wanda50mix": "Wanda 50\\% (mix)"}
CAT_TEX = {"Age": "Age", "Disability_status": "Disability", "ENG-linguistic(language_formality_biases)": "Lang.\\ formality",
           "Gender_identity": "Gender", "Physical_appearance": "Appearance", "Regional": "Regional",
           "Religion": "Religion", "SES": "SES"}
MAIN_VARIANTS = ["int8", "nf4", "awq", "wanda25", "wanda50"]


def num(x, nd=1):
    v = 100 * x
    s = f"{v:.{nd}f}"
    return s.replace("-", "$-$") if not s.startswith("-0.0") or nd != 1 else s.replace("-", "")


def sig(d, lo, hi, nd=1):
    """Value; grey when the 95% CI includes 0."""
    s = num(d, nd)
    return s if (lo > 0 or hi < 0) else f"\\textcolor{{black!45}}{{{s}}}"


def counts(ds):
    up = sum(lo > 0 for _, lo, _ in ds)
    down = sum(hi < 0 for _, _, hi in ds)
    return f"$^{{{up}\\uparrow{down}\\downarrow}}$"


class Ctx:
    def __init__(self, n_boot):
        pt.common.load_env()
        self.items = pt.Items()
        self.runs = pt.load_all(self.items)
        self.stats = {k: pt.per_template(self.items, r["answer"]) for k, r in self.runs.items()}
        self.boot = pt.Bootstrap(len(self.items.templates), n_boot)

    def key(self, m, v, l, s=""):
        return pt.run_key(m, v, l, s)

    def has(self, m, v, l, s=""):
        return self.key(m, v, l, s) in self.runs

    def point(self, k):
        return self.boot.point(self.stats[k])

    def value(self, k, metric):
        p, s = self.point(k)[metric], self.boot.samples(self.stats[k])[metric]
        lo, hi = pt.ci(s)
        return p, lo, hi

    def delta(self, k, base, metric):
        d = self.point(k)[metric] - self.point(base)[metric]
        lo, hi = pt.ci(self.boot.samples(self.stats[k])[metric] - self.boot.samples(self.stats[base])[metric])
        return d, lo, hi

    def bias_flips(self, k, base):
        a, b, sc = self.runs[k]["answer"], self.runs[base]["answer"], self.items.scorable
        ia, ib = (a == self.items.biased) & sc, (b == self.items.biased) & sc
        n = sc.sum()
        return float((ia & ~ib).sum() / n), float((~ia & ib).sum() / n)


def table_benchmark(c):
    metrics = ["acc_dis", "acc_amb", "dir_amb", "s_amb", "s_dis", "gap"]
    out = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
           r"Model & Lang. & Acc$_\text{dis}$ & Unk$_\text{amb}$ & Dir$_\text{amb}$ & $s_\text{amb}$ & $s_\text{dis}$ & Gap$_\text{dis}$ \\",
           r"\midrule"]
    for m in pt.MODELS:
        for j, l in enumerate(pt.LANGS):
            k = c.key(m, "bf16", l)
            cells = []
            for metric in metrics:
                p, lo, hi = c.value(k, metric)
                cells.append(sig(p, lo, hi) if metric in ("dir_amb", "s_amb", "s_dis", "gap") else num(p))
            name = pt.MODEL_NAMES[m] if j == 0 else ""
            out.append(f"{name} & {LANG_TEX[l]} & " + " & ".join(cells) + r" \\")
        out.append(r"\addlinespace[2pt]")
    out[-1] = r"\midrule"
    for l, ref in (("ur", "en"), ("ur_abc", "en"), ("roman_ur", "en"), ("ur_abc", "ur")):
        cells = []
        for metric in metrics:
            ds = [c.delta(c.key(m, "bf16", l), c.key(m, "bf16", ref), metric) for m in pt.MODELS]
            cells.append(num(float(np.mean([d for d, _, _ in ds]))) + counts(ds))
        out.append(f"\\multicolumn{{2}}{{@{{}}l}}{{{LANG_TEX[l]} $-$ {LANG_TEX[ref]}}} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def table_compression(c):
    metrics = ["acc_dis", "acc_amb", "dir_amb", "s_amb", "s_dis", "gap"]
    out = [r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
           r" & $\Delta$Acc$_\text{dis}$ & $\Delta$Unk$_\text{amb}$ & $\Delta$Dir$_\text{amb}$ & $\Delta s_\text{amb}$ & $\Delta s_\text{dis}$ & $\Delta$Gap$_\text{dis}$ & Into & Out of \\",
           r"\midrule"]
    for l in pt.LANGS:
        pts = [c.point(c.key(m, "bf16", l)) for m in pt.MODELS]
        out.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{LANG_TEX[l]}}}}} \\\\")
        out.append("BF16 (abs.) & " + " & ".join(num(float(np.mean([p[x] for p in pts]))) for x in metrics)
                   + r" & -- & -- \\")
        for v in MAIN_VARIANTS:
            pairs = [(c.key(m, v, l), c.key(m, "bf16", l)) for m in pt.MODELS if c.has(m, v, l)]
            cells = []
            for metric in metrics:
                ds = [c.delta(k, b, metric) for k, b in pairs]
                cells.append(num(float(np.mean([d for d, _, _ in ds]))) + counts(ds))
            fl = [c.bias_flips(k, b) for k, b in pairs]
            cells += [num(float(np.mean([a for a, _ in fl]))), num(float(np.mean([b for _, b in fl])))]
            star = "$^\\ast$" if v == "awq" else ""
            out.append(f"{VAR_TEX[v]}{star} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule" if l != pt.LANGS[-1] else r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def table_calibration(c):
    metrics = ["acc_dis", "s_amb", "s_dis"]
    out = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
           r"Lang. & Sparsity & $\Delta$Acc$_\text{dis}$ & $\Delta s_\text{amb}$ & $\Delta s_\text{dis}$ & $\Delta$Flips \\",
           r"\midrule"]
    for l in pt.LANGS:
        for j, pct in enumerate(("25", "50")):
            ms = [m for m in pt.MODELS if c.has(m, f"wanda{pct}mix", l) and c.has(m, f"wanda{pct}", l)]
            cells = []
            for metric in metrics:
                ds = [c.delta(c.key(m, f"wanda{pct}mix", l), c.key(m, f"wanda{pct}", l), metric) for m in ms]
                cells.append(num(float(np.mean([d for d, _, _ in ds]))) + counts(ds))
            fl = []
            for m in ms:
                base = c.runs[c.key(m, "bf16", l)]["answer"]
                fl.append((c.runs[c.key(m, f"wanda{pct}mix", l)]["answer"] != base).mean()
                          - (c.runs[c.key(m, f"wanda{pct}", l)]["answer"] != base).mean())
            cells.append(num(float(np.mean(fl))))
            out.append(f"{LANG_TEX[l] if j == 0 else ''} & {pct}\\% & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule" if l != pt.LANGS[-1] else r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def mix(color, w):
    """Blend a hex colour with the neutral grey (#f0efec) by weight w in [0, 1]; returns an xcolor rgb spec."""
    neutral = np.array([0xF0, 0xEF, 0xEC]) / 255
    col = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    rgb = (1 - w) * neutral + w * col
    return ",".join(f"{x:.3f}" for x in rgb)


def heat_seq(v, vmax=35):
    return f"\\cellcolor[rgb]{{{mix('#2a78d6', min(max(v, 0) / vmax, 1) * 0.75)}}}"


def heat_div(v, vmax=20):
    w = min(abs(v) / vmax, 1) * 0.75
    return f"\\cellcolor[rgb]{{{mix('#e34948' if v > 0 else '#2a78d6', w)}}}"


def table_categories(c):
    """Figure 3 as a heatmap table: per condition, BF16 s_amb and mean Δ s_amb under 4-bit and 50% pruning.
    INT8 and 25% pruning move little and are left to the appendix CSV (fig3b_categories.csv) to keep LNCS width."""
    items = c.items
    cats = sorted(set(items.category))
    groups = {"4-bit": ["nf4", "awq"], "P50": ["wanda50", "wanda50mix"]}
    short = {"en": "English", "ur": "Urdu", "ur_abc": "Urdu (A/B/C)", "roman_ur": "Roman Urdu"}
    col = "|".join(["ccc"] * len(pt.LANGS))
    out = [r"\setlength{\tabcolsep}{2.5pt}", f"\\begin{{tabular}}{{@{{}}l|{col}@{{}}}}", r"\toprule",
           " & " + " & ".join(f"\\multicolumn{{3}}{{c{'|' if k < len(pt.LANGS) - 1 else ''}}}{{{short[l]}}}"
                              for k, l in enumerate(pt.LANGS)) + r" \\",
           r"Category & " + " & ".join([r"BF16 & $\Delta$4-bit & $\Delta$P50"] * len(pt.LANGS)) + r" \\", r"\midrule"]
    for cat in cats:
        sub = sorted({items.template[i] for i in np.where(items.category == cat)[0]})
        b = pt.Bootstrap(len(items.templates), 1, seed=1, subset=sub)
        cells = []
        for l in pt.LANGS:
            v = 100 * float(np.mean([b.point(c.stats[c.key(m, "bf16", l)])["s_amb"] for m in pt.MODELS]))
            cells.append(f"{heat_seq(v)}{v:.1f}".replace("-", "$-$"))
            for g, vs in groups.items():
                ds = [b.point(c.stats[c.key(m, v, l)])["s_amb"] - b.point(c.stats[c.key(m, "bf16", l)])["s_amb"]
                      for m in pt.MODELS for v in vs if c.has(m, v, l)]
                d = 100 * float(np.mean(ds))
                cells.append(f"{heat_div(d)}{d:.1f}".replace("-", "$-$"))
        out.append(f"{CAT_TEX[cat]} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def figdata(c, d):
    os.makedirs(d, exist_ok=True)
    # Figure 1a: into / out of the biased answer per variant and language; noise floor per language
    with open(os.path.join(d, "f1a_bias_flips.dat"), "w") as f:
        f.write("lang variant into out\n")
        for l in pt.LANGS:
            for v in MAIN_VARIANTS:
                fl = [c.bias_flips(c.key(m, v, l), c.key(m, "bf16", l)) for m in pt.MODELS if c.has(m, v, l)]
                f.write(f"{l} {v} {100 * np.mean([a for a, _ in fl]):.2f} {100 * np.mean([b for _, b in fl]):.2f}\n")
    with open(os.path.join(d, "f1a_noise.dat"), "w") as f:
        f.write("lang into out\n")
        for l in pt.LANGS:
            fl = [c.bias_flips(c.key(m, "bf16", l, "rerun"), c.key(m, "bf16", l)) for m in pt.MODELS
                  if c.has(m, "bf16", l, "rerun")]
            f.write(f"{l} {100 * np.mean([a for a, _ in fl]):.2f} {100 * np.mean([b for _, b in fl]):.2f}\n")
    for l in pt.LANGS:  # one file per language for the grouped bars
        with open(os.path.join(d, f"f1a_{l}.dat"), "w") as f:
            f.write("variant into out\n")
            for v in MAIN_VARIANTS:
                fl = [c.bias_flips(c.key(m, v, l), c.key(m, "bf16", l)) for m in pt.MODELS if c.has(m, v, l)]
                f.write(f"{v} {100 * np.mean([a for a, _ in fl]):.2f} {100 * np.mean([b for _, b in fl]):.2f}\n")
    # Figure 1b: flip rate by BF16 confidence bin, pooled over models and compressed variants
    edges = [0.33, 0.5, 0.7, 0.9, 1.01]
    with open(os.path.join(d, "f1b_confidence.dat"), "w") as f:
        f.write("bin " + " ".join(pt.LANGS) + "\n")
        res = {}
        for l in pt.LANGS:
            fl, cnt = np.zeros(4), np.zeros(4)
            for m in pt.MODELS:
                base = c.runs[c.key(m, "bf16", l)]
                bins = np.digitize(base["conf"], edges) - 1
                for v in pt.COMPRESSED:
                    if c.has(m, v, l):
                        flip = c.runs[c.key(m, v, l)]["answer"] != base["answer"]
                        for bb in range(4):
                            fl[bb] += flip[bins == bb].sum()
                            cnt[bb] += (bins == bb).sum()
            res[l] = 100 * fl / np.maximum(cnt, 1)
        for bb in range(4):
            f.write(f"{bb} " + " ".join(f"{res[l][bb]:.2f}" for l in pt.LANGS) + "\n")
    # Figure 2: accuracy vs s_amb for every BF16 and compressed run, one file per language
    for l in pt.LANGS:
        with open(os.path.join(d, f"f2_{l}.dat"), "w") as f:
            f.write("acc samb bf16 model variant\n")
            for m in pt.MODELS:
                for v in pt.VARIANTS:
                    if c.has(m, v, l):
                        p = c.point(c.key(m, v, l))
                        f.write(f"{100 * p['acc_dis']:.2f} {100 * p['s_amb']:.2f} {int(v == 'bf16')} {m} {v}\n")
    arr = []
    for l in pt.LANGS:
        for m in pt.MODELS:
            for v in pt.VARIANTS:
                if c.has(m, v, l):
                    p = c.point(c.key(m, v, l))
                    arr.append((p["acc_dis"], p["s_amb"]))
    a = np.array(arr)
    return float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), len(a)


AXIS = ("axis line style={draw=cAxis}, tick style={draw=cAxis}, major grid style={draw=cGrid}, "
        "tick label style={font=\\scriptsize, text=black!75}, label style={font=\\scriptsize}, "
        "legend style={font=\\scriptsize, draw=none, fill=none}")
VAR_SHORT = {"int8": "INT8", "nf4": "NF4", "awq": "AWQ", "wanda25": "W25", "wanda50": "W50"}
# Urdu with Latin labels shares Urdu's hue (same language), hollow markers and dashed lines tell them apart;
# a fourth hue would fail the all-pairs colour-vision check.
LANG_STYLE = {"en": ("cEn", "*", "solid"), "ur": ("cUr", "square*", "solid"), "ur_abc": ("cUr", "square", "densely dashed"),
              "roman_ur": ("cRo", "triangle*", "solid")}


def fig_bias_flips(c):
    """Figure 1a: per language, % of items newly given / no longer given the biased answer; dashed = noise floor."""
    out = [r"\begin{tikzpicture}",
           r"\begin{groupplot}[group style={group size=2 by 2, horizontal sep=0.35cm, vertical sep=0.9cm, y descriptions at=edge left, x descriptions at=edge bottom},"
           r" width=0.56\textwidth, height=3.6cm, ybar=0.6pt, bar width=5pt, ymin=0, ymax=30, ymajorgrids,"
           r" symbolic x coords={" + ",".join(MAIN_VARIANTS) + r"}, xtick=data,"
           r" xticklabels={" + ",".join(VAR_SHORT[v] for v in MAIN_VARIANTS) + r"}, enlarge x limits=0.15,"
           r" ylabel={\% of items}, " + AXIS + "]"]
    for j, l in enumerate(pt.LANGS):
        nf = [c.bias_flips(c.key(m, "bf16", l, "rerun"), c.key(m, "bf16", l)) for m in pt.MODELS
              if c.has(m, "bf16", l, "rerun")]
        noise = 100 * float(np.mean([a + b for a, b in nf]) / 2)
        legend = r", legend to name=f1leg, legend columns=3" if j == 0 else ""
        out.append(f"\\nextgroupplot[title={{\\scriptsize {LANG_TEX[l]}}}{legend}]")
        for series, idx, col in (("into", 0, "cUr"), ("out", 1, "cEn")):
            coords = []
            for v in MAIN_VARIANTS:
                fl = [c.bias_flips(c.key(m, v, l), c.key(m, "bf16", l)) for m in pt.MODELS if c.has(m, v, l)]
                coords.append(f"({v},{100 * np.mean([x[idx] for x in fl]):.2f})")
            out.append(f"\\addplot[fill={col}, draw=none] coordinates {{{' '.join(coords)}}};")
        y = f"{{axis cs:{MAIN_VARIANTS[0]},{noise:.2f}}}"  # (A |- B): x of A, y of B
        out.append(f"\\draw[cMuted, densely dashed, thick] ({{rel axis cs:0,0}} |- {y}) -- ({{rel axis cs:1,0}} |- {y});")
        if j == 0:
            out += [r"\addlegendentry{newly given the biased answer}", r"\addlegendentry{no longer given it}",
                    r"\addlegendimage{cMuted, densely dashed, thick}\addlegendentry{noise floor (BF16 rerun)}"]
    out += [r"\end{groupplot}", r"\node[anchor=north] at ([yshift=-0.75cm]$(group c1r2.south)!0.5!(group c2r2.south)$) {\pgfplotslegendfromname{f1leg}};",
            r"\end{tikzpicture}"]
    return "\n".join(out)


def fig_confidence(c):
    """Figure 1b: flip rate by BF16 confidence bin, one line per language."""
    edges = [0.33, 0.5, 0.7, 0.9, 1.01]
    labels = ["0.33--0.5", "0.5--0.7", "0.7--0.9", "0.9--1"]
    out = [r"\begin{tikzpicture}",
           r"\begin{axis}[width=0.5\textwidth, height=4.2cm, ymin=0, ymax=40, ymajorgrids, xtick={0,1,2,3},"
           r" xticklabels={" + ",".join(labels) + r"}, xlabel={BF16 confidence}, ylabel={\% answers flipped},"
           r" legend pos=north east, " + AXIS + "]"]
    for l in pt.LANGS:
        fl, cnt = np.zeros(4), np.zeros(4)
        for m in pt.MODELS:
            base = c.runs[c.key(m, "bf16", l)]
            bins = np.digitize(base["conf"], edges) - 1
            for v in pt.COMPRESSED:
                if c.has(m, v, l):
                    flip = c.runs[c.key(m, v, l)]["answer"] != base["answer"]
                    for b in range(4):
                        fl[b] += flip[bins == b].sum()
                        cnt[b] += (bins == b).sum()
        col, mark, dash = LANG_STYLE[l]
        coords = " ".join(f"({b},{100 * fl[b] / max(cnt[b], 1):.2f})" for b in range(4))
        out.append(f"\\addplot[{col}, {dash}, line width=1pt, mark={mark}, mark size=2.2pt, mark options={{solid}}] "
                   f"coordinates {{{coords}}};")
        out.append(f"\\addlegendentry{{{LANG_TEX[l]}}}")
    out += [r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(out)


def fig_scatter(c):
    """Figure 2: s_amb vs disambiguated accuracy for every BF16 and compressed run (BF16 runs ringed)."""
    out = [r"\begin{tikzpicture}",
           r"\begin{axis}[width=0.62\textwidth, height=5.2cm, xmin=15, xmax=95, ymin=-5, ymax=40, grid=major,"
           r" xlabel={Accuracy, disambiguated (\%)}, ylabel={$s_\text{amb}$ ($\times100$)}, legend pos=north west, "
           + AXIS + "]"]
    for l in pt.LANGS:
        col, mark, _ = LANG_STYLE[l]
        hollow = not mark.endswith("*")
        comp, bf = [], []
        for m in pt.MODELS:
            for v in pt.VARIANTS:
                if c.has(m, v, l):
                    p = c.point(c.key(m, v, l))
                    (bf if v == "bf16" else comp).append(f"({100 * p['acc_dis']:.2f},{100 * p['s_amb']:.2f})")
        out.append(f"\\addplot[only marks, mark={mark}, mark size=1.6pt, {col}, line width=0.6pt, fill opacity=0.75, "
                   f"draw opacity=0.9] coordinates {{{' '.join(comp)}}};")
        out.append(f"\\addlegendentry{{{LANG_TEX[l]}}}")
        fill = "white" if hollow else col
        out.append(f"\\addplot[only marks, mark={mark.rstrip('*')}*, mark size=3pt, fill={fill}, draw=black, line width=0.6pt, "
                   f"forget plot] coordinates {{{' '.join(bf)}}};")
    out += [r"\end{axis}", r"\end{tikzpicture}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--paper", default=PAPER)
    args = ap.parse_args()
    c = Ctx(args.boot)
    tdir = os.path.join(args.paper, "tables")
    os.makedirs(tdir, exist_ok=True)
    for name, fn in [("t1_benchmark", table_benchmark), ("t2_compression", table_compression),
                     ("t3_calibration", table_calibration), ("f3_categories", table_categories)]:
        open(os.path.join(tdir, f"{name}.tex"), "w", encoding="utf-8").write(fn(c) + "\n")
        print(f"  {name}.tex")
    fdir = os.path.join(args.paper, "figures")
    os.makedirs(fdir, exist_ok=True)
    for name, fn in [("f1a_bias_flips", fig_bias_flips), ("f1b_confidence", fig_confidence), ("f2_scatter", fig_scatter)]:
        open(os.path.join(fdir, f"{name}.tex"), "w", encoding="utf-8").write(fn(c) + "\n")
        print(f"  figures/{name}.tex")
    r, n = figdata(c, os.path.join(args.paper, "figdata"))
    open(os.path.join(tdir, "f2_r.tex"), "w").write(f"{r:.2f}%\n")
    print(f"  figdata/*.dat (Figure 2: r = {r:.2f} over {n} runs)")
    # Raw result tables (all CSVs from paper_tables.py) and the findings notes, for reference while writing
    rdir = os.path.join(args.paper, "results")
    os.makedirs(rdir, exist_ok=True)
    for p in glob.glob(os.path.join(pt.OUT, "tables", "*.csv")) + [os.path.join(pt.OUT, "tables.md"),
                                                                     os.path.join(pt.OUT, "findings.md")]:
        if os.path.exists(p):
            shutil.copy(p, rdir)
    print(f"-> {args.paper}/tables, figdata, results")


if __name__ == "__main__":
    main()
