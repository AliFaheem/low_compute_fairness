"""Write the paper's tables (LaTeX, booktabs) into main.tex of the Overleaf project.

    uv run python scripts/analysis/paper_tables.py        # computes results/analysis/ (answers cache + CSVs)
    uv run python scripts/analysis/paper_latex.py          # -> paper/icse_overleaf/main.tex (tables)
    uv run --with matplotlib python scripts/analysis/paper_figures.py   # -> paper/icse_overleaf/fig/*.pdf

Uses the same answers cache, metrics and template bootstrap as paper_tables.py (docs/analysis_plan.md), so the
numbers match. Tables and Figure 2's correlation (\\corrR) are spliced into main.tex between "% <auto:NAME>" /
"% </auto:NAME>" marker lines; the rest of main.tex is left untouched. Figures are PDFs drawn by paper_figures.py.
Raw result CSVs and the figure data (.dat) stay in results/analysis/, not in the paper.
"""
import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_tables as pt  # noqa: E402

PAPER = "paper/icse_overleaf"
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


# Preparation time per model, measured from the queue logs (experiments.md): Wanda pruning 13 min (RTX 4080 Super)
# to 21 min (RTX 3090); AWQ quantization incl. perplexity check about 19-21 min (RTX 3090 / 3090 Ti).
PREP = {"bf16": "--", "int8": "on load", "nf4": "on load", "awq": "19--21 min", "wanda25": "13--21 min",
        "wanda50": "13--21 min"}
SETTING = {"bf16": "16-bit", "int8": "8-bit", "nf4": "4-bit", "awq": "4-bit", "wanda25": "25\\% sparse",
           "wanda50": "50\\% sparse"}


def run_meta(m, v, l):
    import json
    p = os.path.join(pt.common.RUNS_DIR, pt.run_key(m, v, l), "meta.json")
    if not os.path.exists(p):
        return None
    meta = json.load(open(p))
    ss = meta.get("sessions", [])
    rows = sum(s.get("rows_after", 0) - s.get("rows_before", 0) for s in ss)
    secs = sum(s.get("seconds", 0) for s in ss)
    return {"gpu": meta.get("gpu"), "weights": meta.get("memory_footprint_gb"),
            "peak": max([s.get("peak_gpu_memory_gb") or 0 for s in ss] or [0]), "speed": rows / secs if secs else None}


def table_cost(c):
    """What compression saves (memory, speed) next to what it costs (accuracy, newly stereotyped items).
    Speed: prompts scored per second relative to the same model's BF16 run, only where both ran on the same GPU
    model; mean over the four conditions and the models with such a pair."""
    out = [r"\begin{tabular}{@{}llrrrlrrrr@{}}", r"\toprule",
           r" & & \multicolumn{4}{c}{What it saves} & \multicolumn{4}{c}{What it costs} \\",
           r"\cmidrule(lr){3-6}\cmidrule(l){7-10}",
           r"Variant & Weights & Mem.\ (GB) & Peak (GB) & Speed & Prep.\ & $\Delta$Acc$_\text{dis}$ en & "
           r"$\Delta$Acc$_\text{dis}$ ur & Into en & Into ur \\", r"\midrule"]
    for v in ["bf16"] + MAIN_VARIANTS:
        metas = [run_meta(m, v, "en") for m in pt.MODELS]
        metas = [x for x in metas if x and x["weights"]]
        w = np.mean([x["weights"] for x in metas])
        peak = np.mean([x["peak"] for x in metas])
        ratios = []
        for m in pt.MODELS:
            for l in pt.LANGS:
                a, b = run_meta(m, v, l), run_meta(m, "bf16", l)
                if a and b and a["speed"] and b["speed"] and a["gpu"] == b["gpu"]:
                    ratios.append(a["speed"] / b["speed"])
        speed = f"{np.mean(ratios):.2f}$\\times$" if ratios else "--"
        if v == "bf16":
            cost = ["--"] * 4
        else:
            cost = []
            for l in ("en", "ur"):
                ds = [c.delta(c.key(m, v, l), c.key(m, "bf16", l), "acc_dis")[0] for m in pt.MODELS if c.has(m, v, l)]
                cost.append(num(float(np.mean(ds))))
            for l in ("en", "ur"):
                fl = [c.bias_flips(c.key(m, v, l), c.key(m, "bf16", l))[0] for m in pt.MODELS if c.has(m, v, l)]
                cost.append(num(float(np.mean(fl))))
        star = "$^\\ast$" if v == "awq" else ""
        out.append(f"{VAR_TEX[v]}{star} & {SETTING[v]} & {w:.1f} & {peak:.1f} & {speed} & {PREP[v]} & "
                   + " & ".join(cost) + r" \\")
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


def splice(tex, name, body):
    """Replace the lines between "% <auto:name>" and "% </auto:name>" in tex with body."""
    pat = re.compile(rf"(^[ \t]*% <auto:{name}>[ \t]*\n).*?(^[ \t]*% </auto:{name}>)", re.S | re.M)
    if not pat.search(tex):
        sys.exit(f"main.tex has no '% <auto:{name}>' ... '% </auto:{name}>' markers")
    return pat.sub(lambda m: m.group(1) + body.rstrip("\n") + "\n" + m.group(2), tex, count=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--paper", default=PAPER)
    args = ap.parse_args()
    c = Ctx(args.boot)
    main_tex = os.path.join(args.paper, "main.tex")
    tex = open(main_tex, encoding="utf-8").read()
    # Figures are PDFs in fig/ (paper_figures.py); only the tables and Figure 2's r are spliced into main.tex.
    for name, fn in [("t1_benchmark", table_benchmark), ("t2_compression", table_compression),
                     ("t3_calibration", table_calibration), ("t0_cost", table_cost)]:
        tex = splice(tex, name, fn(c))
        print(f"  main.tex: {name}")
    r, n = figdata(c, os.path.join(pt.OUT, "figdata"))
    tex = splice(tex, "f2_r", f"\\newcommand{{\\corrR}}{{{r:.2f}}}")
    print(f"  {pt.OUT}/figdata/*.dat (Figure 2: r = {r:.2f} over {n} runs)")
    if tex != open(main_tex, encoding="utf-8").read():
        open(main_tex, "w", encoding="utf-8").write(tex)
    print(f"-> {args.paper}/main.tex")


if __name__ == "__main__":
    main()
