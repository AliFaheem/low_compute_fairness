"""Draw the paper's figures as vector PDFs into the paper's fig/ folder (matplotlib, kept out of the project env).

    uv run python scripts/analysis/paper_tables.py                                   # answers cache + CSVs
    uv run --with matplotlib python scripts/analysis/paper_figures.py               # -> paper/icse_overleaf/fig/*.pdf
    uv run --with matplotlib python scripts/analysis/paper_figures.py --png DIR     # also PNG previews in DIR

Same metrics and runs as paper_tables.py / paper_latex.py. Sized for the LNCS text block (12.2 cm = 4.8 in).
Colours: dataviz reference palette (validated for colour-vision deficiency): blue / orange / aqua for English /
Urdu / Roman Urdu; Urdu with Latin A/B/C labels reuses Urdu's orange with hollow markers and dashed lines (a fourth
hue fails the all-pairs check). Levels use a single-hue violet ramp (so blue means only "less biased"), changes the blue-grey-red diverging pair.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_latex as pl  # noqa: E402
import paper_tables as pt  # noqa: E402

PAPER = "paper/icse_overleaf"
WIDTH = 4.8  # inches, LNCS text width
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#f0efec"
STYLE = {"en": dict(color=BLUE, marker="o", ls="-", fill=True), "ur": dict(color=ORANGE, marker="s", ls="-", fill=True),
         "ur_abc": dict(color=ORANGE, marker="s", ls="--", fill=False), "roman_ur": dict(color=AQUA, marker="^", ls="-", fill=True)}
NAME = {"en": "English", "ur": "Urdu", "ur_abc": "Urdu (A/B/C)", "roman_ur": "Roman Urdu"}
VSHORT = {"int8": "INT8", "nf4": "NF4", "awq": "AWQ", "wanda25": "W25", "wanda50": "W50"}
CAT = {"Age": "Age", "Disability_status": "Disability", "ENG-linguistic(language_formality_biases)": "Lang. formality",
       "Gender_identity": "Gender", "Physical_appearance": "Appearance", "Regional": "Regional",
       "Religion": "Religion", "SES": "SES"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    "legend.frameon": False, "lines.linewidth": 1.2})


def fig_churn(c, path):
    """Figure 1: (a) bias churn per condition vs noise floor; (b) flips by BF16 confidence."""
    fig = plt.figure(figsize=(WIDTH, 3.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.7, wspace=0.12)
    x = np.arange(len(pl.MAIN_VARIANTS))
    w = 0.38
    axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
    for j, (ax, l) in enumerate(zip(axes, pt.LANGS)):
        into, out = [], []
        for v in pl.MAIN_VARIANTS:
            fl = [c.bias_flips(c.key(m, v, l), c.key(m, "bf16", l)) for m in pt.MODELS if c.has(m, v, l)]
            into.append(100 * np.mean([a for a, _ in fl]))
            out.append(100 * np.mean([b for _, b in fl]))
        nf = [c.bias_flips(c.key(m, "bf16", l, "rerun"), c.key(m, "bf16", l)) for m in pt.MODELS
              if c.has(m, "bf16", l, "rerun")]
        noise = 100 * float(np.mean([a + b for a, b in nf]) / 2)
        ax.bar(x - w / 2 - 0.02, into, w, color=ORANGE, label="newly given the biased answer")
        ax.bar(x + w / 2 + 0.02, out, w, color=BLUE, label="no longer given it")
        ax.axhline(noise, color=MUTED, ls=(0, (3, 2)), lw=1, label="noise floor (BF16 rerun)")
        ax.set_title(NAME[l], pad=3)
        ax.set_xticks(x, [VSHORT[v] for v in pl.MAIN_VARIANTS], rotation=90)
        ax.set_ylim(0, 35)
        ax.grid(axis="x", visible=False)
        if j:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("% of items")
    axes[0].text(-0.42, 1.12, "(a)", transform=axes[0].transAxes, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.505))

    ax = fig.add_subplot(gs[1, :])
    edges = [0.33, 0.5, 0.7, 0.9, 1.01]
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
        s = STYLE[l]
        ax.plot(range(4), 100 * fl / np.maximum(cnt, 1), color=s["color"], ls=s["ls"], marker=s["marker"],
                markersize=4, markerfacecolor=s["color"] if s["fill"] else "white", markeredgewidth=1, label=NAME[l])
    ax.set_xticks(range(4), ["0.33–0.50", "0.50–0.70", "0.70–0.90", "0.90–1.00"])
    ax.set_xlabel("BF16 confidence (probability of the BF16 answer)")
    ax.set_ylabel("% of answers flipped")
    ax.set_ylim(0, 40)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2)
    ax.text(-0.095, 1.06, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    return fig


def fig_scatter(c, path):
    """Figure 2: ambiguous-context bias vs disambiguated accuracy, every run; BF16 runs large with a dark ring."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.7))
    pts = []
    for l in pt.LANGS:
        s = STYLE[l]
        for bf in (False, True):
            xs, ys = [], []
            for m in pt.MODELS:
                for v in pt.VARIANTS:
                    if c.has(m, v, l) and (v == "bf16") == bf:
                        p = c.point(c.key(m, v, l))
                        xs.append(100 * p["acc_dis"])
                        ys.append(100 * p["s_amb"])
                        pts.append((p["acc_dis"], p["s_amb"]))
            face = s["color"] if s["fill"] else "white"
            if bf:
                ax.scatter(xs, ys, s=34, marker=s["marker"], facecolor=face, edgecolor=INK, linewidth=0.8, zorder=3)
            else:
                ax.scatter(xs, ys, s=13, marker=s["marker"], facecolor=face, edgecolor=s["color"], linewidth=0.8,
                           alpha=0.85, zorder=2, label=NAME[l])
    ax.scatter([], [], s=34, marker="o", facecolor="white", edgecolor=INK, linewidth=0.8, label="BF16 (uncompressed)")
    a = np.array(pts)
    r = np.corrcoef(a[:, 0], a[:, 1])[0, 1]
    ax.set_xlabel("Accuracy on disambiguated items (%)")
    ax.set_ylabel(r"Ambiguous-context bias $s_{\mathrm{amb}}$ (×100)")
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
    ax.set_xlim(15, 95)
    ax.legend(loc="upper left", ncol=1)
    ax.text(0.99, 0.04, f"Pearson r = {r:.2f} ({len(a)} runs)", transform=ax.transAxes, ha="right", color=INK2)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    return fig, r, len(a)


def fig_categories(c, path):
    """Figure 3: per category, BF16 s_amb (sequential) and its change under 4-bit and 50% pruning (diverging)."""
    items = c.items
    cats = sorted(set(items.category))
    groups = {"4-bit": ["nf4", "awq"], "P50": ["wanda50", "wanda50mix"]}
    base = np.zeros((len(cats), len(pt.LANGS)))
    diff = np.zeros((len(cats), 2 * len(pt.LANGS)))
    for i, cat in enumerate(cats):
        sub = sorted({items.template[t] for t in np.where(items.category == cat)[0]})
        b = pt.Bootstrap(len(items.templates), 1, seed=1, subset=sub)
        for j, l in enumerate(pt.LANGS):
            ref = {m: b.point(c.stats[c.key(m, "bf16", l)])["s_amb"] for m in pt.MODELS}
            base[i, j] = 100 * np.mean(list(ref.values()))
            for g, (gname, vs) in enumerate(groups.items()):
                ds = [b.point(c.stats[c.key(m, v, l)])["s_amb"] - ref[m] for m in pt.MODELS for v in vs if c.has(m, v, l)]
                diff[i, 2 * j + g] = 100 * np.mean(ds)
    # Levels in violet, so blue keeps one meaning (less biased) in the diverging panel beside it.
    seq = LinearSegmentedColormap.from_list("seq", ["#f3f1fb", "#b3a9ee", "#6250d6", "#3b2f8f"])
    div = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, RED])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(WIDTH, 2.35), gridspec_kw={"width_ratios": [4, 8], "wspace": 0.05})
    for ax, data, cmap, vmin, vmax, cols in (
            (a1, base, seq, 0, 40, [NAME[l] for l in pt.LANGS]),
            (a2, diff, div, -20, 20, [f"{NAME[l]}\n{g}" for l in pt.LANGS for g in groups])):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)), cols, rotation=90)
        ax.set_yticks(range(len(cats)), [CAT[x] for x in cats])
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for (i, j), v in np.ndenumerate(data):
            frac = (v - vmin) / (vmax - vmin)
            dark = frac > 0.72 if cmap is seq else abs(v) > 0.6 * vmax
            ax.text(j, i, f"{v:.1f}".replace("-", "−"), ha="center", va="center", fontsize=5.3,
                    color="white" if dark else INK)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cats)), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
    a2.tick_params(labelleft=False)
    for k in range(1, len(pt.LANGS)):
        a2.axvline(2 * k - 0.5, color="white", lw=3)
    a1.set_title(r"BF16 $s_{\mathrm{amb}}$", pad=3)
    a2.set_title(r"Change in $s_{\mathrm{amb}}$ under compression (red = more biased)", pad=3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=PAPER)
    ap.add_argument("--png", help="also write PNG previews into this directory")
    args = ap.parse_args()
    c = pl.Ctx(1)  # point estimates only
    d = os.path.join(args.paper, "fig")
    os.makedirs(d, exist_ok=True)
    out = {}
    out["fig1_bias_churn"] = fig_churn(c, os.path.join(d, "fig1_bias_churn.pdf"))
    f2, r, n = fig_scatter(c, os.path.join(d, "fig2_bias_vs_accuracy.pdf"))
    out["fig2_bias_vs_accuracy"] = f2
    out["fig3_categories"] = fig_categories(c, os.path.join(d, "fig3_categories.pdf"))
    if args.png:
        os.makedirs(args.png, exist_ok=True)
        for name, fig in out.items():
            fig.savefig(os.path.join(args.png, f"{name}.png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    print(f"-> {d}/fig1_bias_churn.pdf, fig2_bias_vs_accuracy.pdf (r = {r:.2f}, {n} runs), fig3_categories.pdf")


if __name__ == "__main__":
    main()
