"""Pick illustrative PakBBQ examples by fixed rules and splice them into the paper's main.tex.

    uv run python scripts/analysis/paper_tables.py     # answers cache (results/analysis/answers.npz)
    uv run python scripts/analysis/paper_examples.py   # -> paper/icse_overleaf/main.tex, markers % <auto:ex_*>

Nothing is hand-picked: each block states its rule, and ties go to the shortest item (then lowest id).
- ex_churn (main paper): two items that change their bias status under compression, in opposite directions.
- ex_categories (appendix): the shortest ambiguous, negative item of each category, English and Roman Urdu.
- ex_format (appendix): Gemma's ten most likely next tokens for one Urdu prompt (answer-format failure).
- ex_labels (appendix): one item whose per-order answers under Urdu letters follow screen position.
Urdu script is not typeset (the LNCS template compiles with pdfLaTeX); Urdu tokens are named in transliteration.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_latex as pl  # noqa: E402
import paper_tables as pt  # noqa: E402

COND = ["en", "ur", "ur_abc", "roman_ur"]
COND_TEX = {"en": "English", "ur": "Urdu", "ur_abc": "Urdu (A/B/C)", "roman_ur": "Roman Urdu"}
VARS = ["bf16", "int8", "nf4", "awq", "wanda25", "wanda50"]
VAR_TEX = {"bf16": "BF16", "int8": "INT8", "nf4": "NF4", "awq": "AWQ", "wanda25": "W25", "wanda50": "W50"}
URDU_LETTERS = {"ا": r"\textit{alif}", "ب": r"\textit{be}", "پ": r"\textit{pe}"}


def tex(s):
    """Escape plain text for LaTeX."""
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(ch, ch) for ch in s)


def token_tex(t):
    t = t.strip("▁") if t.strip() else t
    if t.strip() in URDU_LETTERS:
        return URDU_LETTERS[t.strip()]
    if any("؀" <= ch <= "ۿ" for ch in t):
        return r"\textit{[Urdu text]}"
    if t in ("\n", "\n\n", "\n\n\n") or not t.strip():
        return {"\n": r"\textit{newline}", "\n\n": r"\textit{2 newlines}", "\n\n\n": r"\textit{3 newlines}"}.get(
            t, r"\textit{space}")
    return r"\texttt{" + tex(t.strip()) + "}"


class Data:
    def __init__(self):
        self.c = pl.Ctx(1)
        self.items = self.c.items
        self.text = {l: {it["id"]: it for it in pt.common.load_items(l)} for l in ("en", "roman_ur")}

    def ans(self, m, v, l):
        k = self.c.key(m, v, l)
        return self.c.runs[k]["answer"] if k in self.c.runs else None

    def length(self, i):
        it = self.text["en"][int(self.items.ids[i])]
        return len(it["context"]) + len(it["question"])

    def kind(self, i, a):
        """Short label for answer index a of item i: S (stereotyped group), O (other group), U (Unknown)."""
        if a == self.items.unknown[i]:
            return "U"
        return "S" if a == self.items.biased[i] else "O"


def item_block(d, i, roman=True):
    en = d.text["en"][int(d.items.ids[i])]
    ro = d.text["roman_ur"][int(d.items.ids[i])]
    opts = " / ".join(tex(en[f"ans{k}"]) + (r"$^\mathrm{S}$" if k == d.items.biased[i] else "") for k in range(3))
    rows = [rf"\textbf{{Context:}} {tex(en['context'])} \textbf{{Q:}} {tex(en['question'])} "
            rf"\textbf{{Options:}} {opts}"]
    if roman:
        ropts = " / ".join(tex(ro[f"ans{k}"]) for k in range(3))
        rows.append(rf"\textit{{Roman Urdu:}} {tex(ro['context'])} {tex(ro['question'])} ({ropts})")
    return rows


def ex_churn(d):
    """Rule: over models, ambiguous scorable items whose BF16 answer is the same in all four conditions -- Unknown
    for example 1, the stereotyped group for example 2 -- and which, under compression, move into the biased answer
    (example 1) / out of it to Unknown (example 2) in the most (variant, condition) cells; ties -> shortest."""
    it = d.items
    cand = np.where(it.ambig & it.scorable)[0]
    best = {"into": None, "out": None}
    for m in pt.MODELS:
        bf = {l: d.ans(m, "bf16", l) for l in COND}
        same = np.all([bf[l][cand] == bf["en"][cand] for l in COND], axis=0)
        for direction in ("into", "out"):
            score = np.zeros(len(cand))
            for v in VARS[1:]:
                for l in COND:
                    a = d.ans(m, v, l)
                    if a is None:
                        continue
                    ia, ib = a[cand] == it.biased[cand], bf[l][cand] == it.biased[cand]
                    score += (ia & ~ib) if direction == "into" else (~ia & ib & (a[cand] == it.unknown[cand]))
            start = it.unknown[cand] if direction == "into" else it.biased[cand]  # BF16: Unknown / stereotyped
            score = np.where(same & (bf["en"][cand] == start), score, -1)
            for j in np.argsort(-score, kind="stable")[:50]:
                key = (score[j], -d.length(cand[j]), -int(it.ids[cand[j]]))
                if best[direction] is None or key > best[direction][0]:
                    best[direction] = (key, m, cand[j])
    out = [r"\begin{tabular}{@{}l" + "c" * len(VARS) + "@{}}", r"\toprule",
           "Condition & " + " & ".join(VAR_TEX[v] for v in VARS) + r" \\", r"\midrule"]
    caption_items = []
    for n, direction in enumerate(("into", "out"), 1):
        _, m, i = best[direction]
        caption_items.append((n, m, i, direction))
        out.append(rf"\multicolumn{{{len(VARS) + 1}}}{{@{{}}p{{0.97\textwidth}}}}{{\textbf{{Example {n}}} "
                   rf"({pl.CAT_TEX[it.category[i]]}, {pt.MODEL_NAMES[m]}). " + " ".join(item_block(d, i, roman=False))
                   + r"} \\")
        out.append(r"\addlinespace[2pt]")
        for l in COND:
            cells = []
            for v in VARS:
                a = d.ans(m, v, l)
                cells.append("--" if a is None else d.kind(i, a[i]))
            cells = [rf"\textbf{{{x}}}" if x == "S" else x for x in cells]
            out.append(f"{COND_TEX[l]} & " + " & ".join(cells) + r" \\")
        out.append(r"\midrule" if n == 1 else r"\bottomrule")
    out.append(r"\end{tabular}")
    return "\n".join(out)


def ex_categories(d):
    """Rule: per category, the shortest ambiguous negative item that is scorable for bias."""
    it = d.items
    out = [r"\begin{tabular}{@{}p{0.16\textwidth}p{0.81\textwidth}@{}}", r"\toprule",
           r"Category & Item (English; Roman Urdu below). $^\mathrm{S}$ = stereotype-consistent answer; correct = Unknown \\",
           r"\midrule"]
    for cat in sorted(set(it.category)):
        idx = np.where((it.category == cat) & it.ambig & it.neg & it.scorable)[0]
        i = min(idx, key=lambda j: (d.length(j), int(it.ids[j])))
        rows = item_block(d, i)
        out.append(f"{pl.CAT_TEX[cat]} & " + rows[0] + r" \\")
        out.append(" & " + rows[1] + r" \\")
        out.append(r"\addlinespace[3pt]")
    out[-1] = r"\bottomrule"
    out.append(r"\end{tabular}")
    return "\n".join(out)


def ex_format(d):
    """Rule: the Gemma-4-E4B BF16 Urdu prompt (option order 0) with the highest probability on end-of-text among
    items whose answer letter is still the argmax over labels; shows its 10 most likely next tokens."""
    path = os.path.join(pt.common.RUNS_DIR, d.c.key("gemma-4-e4b", "bf16", "ur"), "preds.jsonl")
    best = None
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["order"] != 0 or "top" not in r:
            continue
        tok0 = r["top"][0]
        if "eos" in tok0[1] or tok0[1].strip() == "":
            key = (tok0[2], -r["mass"])
            if best is None or key > best[0]:
                best = (key, r)
    r = best[1]
    i = d.items.index[r["id"]]
    rows = " & ".join(token_tex(t) for _, t, _ in r["top"][:5]) + r" \\" + "\n" + " & ".join(
        f"{100 * np.exp(lp):.1f}\\%" for _, _, lp in r["top"][:5]) + r" \\"
    rows2 = " & ".join(token_tex(t) for _, t, _ in r["top"][5:10]) + r" \\" + "\n" + " & ".join(
        f"{100 * np.exp(lp):.1f}\\%" for _, _, lp in r["top"][5:10]) + r" \\"
    out = [r"\begin{tabular}{@{}ccccc@{}}", r"\toprule",
           rf"\multicolumn{{5}}{{@{{}}p{{0.95\textwidth}}}}{{Item ({pl.CAT_TEX[d.items.category[i]]}): "
           + " ".join(item_block(d, i, roman=False)) + rf" Probability on the three labels: {100 * r['mass']:.1f}\%.}} \\",
           r"\midrule", rows, r"\midrule", rows2, r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def ex_labels(d):
    """Rule: for Llama-3.1-8B BF16, an item whose per-order Urdu-letter answers sit in the same screen position in
    all three orders (so three different options) while the A/B/C answers are one option in all three orders;
    shortest such item."""
    it = d.items
    k_ur, k_abc = d.c.key("llama-3.1-8b", "bf16", "ur"), d.c.key("llama-3.1-8b", "bf16", "ur_abc")
    s_ur, s_abc = d.c.runs[k_ur]["slot"], d.c.runs[k_abc]["slot"]
    opt_ur = (s_ur + np.arange(3)[None, :]) % 3  # original option chosen in each order
    opt_abc = (s_abc + np.arange(3)[None, :]) % 3
    ok = (s_ur == s_ur[:, :1]).all(axis=1) & (opt_abc == opt_abc[:, :1]).all(axis=1)
    i = min(np.where(ok)[0], key=lambda j: (d.length(j), int(it.ids[j])))
    en = d.text["en"][int(it.ids[i])]
    pos = ["1st", "2nd", "3rd"]
    letters_ur, letters_abc = [r"\textit{alif}", r"\textit{be}", r"\textit{pe}"], ["A", "B", "C"]
    out = [r"\begin{tabular}{@{}lll@{}}", r"\toprule",
           rf"\multicolumn{{3}}{{@{{}}p{{0.95\textwidth}}}}{{Item ({pl.CAT_TEX[it.category[i]]}): "
           + " ".join(item_block(d, i, roman=False)) + r"} \\", r"\midrule",
           r"Option order & Urdu letters: answer (screen position) & Latin letters: answer (screen position) \\",
           r"\midrule"]
    for o in range(3):
        a_ur, a_abc = opt_ur[i, o], opt_abc[i, o]
        out.append(f"Order {o + 1} & {tex(en[f'ans{a_ur}'])} ({letters_ur[s_ur[i, o]]}, {pos[s_ur[i, o]]}) & "
                   f"{tex(en[f'ans{a_abc}'])} ({letters_abc[s_abc[i, o]]}, {pos[s_abc[i, o]]}) \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", default=pl.PAPER)
    args = ap.parse_args()
    d = Data()
    path = os.path.join(args.paper, "main.tex")
    tex_src = open(path, encoding="utf-8").read()
    for name, fn in [("ex_churn", ex_churn), ("ex_categories", ex_categories), ("ex_format", ex_format),
                     ("ex_labels", ex_labels)]:
        tex_src = pl.splice(tex_src, name, fn(d))
        print(f"  main.tex: {name}")
    open(path, "w", encoding="utf-8").write(tex_src)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
