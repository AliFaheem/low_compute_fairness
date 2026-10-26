# Analysis plan: what goes in the paper

Revised 26 Oct 2026 after the first pass over Exp 1. Target: an 8-page ACL-style long paper. The main body has
**3 tables and 3 figures**; everything else goes to the appendix. All numbers come from one script
(`scripts/analysis/paper_tables.py`). It follows the conventions of the BBQ line of work (Parrish et al. 2022;
KoBBQ, MBBQ, PakBBQ) and of the 2026 compression-and-bias papers in `docs/related_work.md`.

## The story the main body tells

The first pass (`results/analysis/findings.md`) supports three claims. Each one gets one main exhibit:

1. **Compression changes many biased answers, though aggregate scores barely move.** Flip rates are far above the
   noise floor even at 8 bits, and largest in Urdu and Roman Urdu. Those extra flips are near-ties, so they reflect
   the models' lower confidence in those languages.
2. **An apparent drop in bias usually means the model got worse.** Bias change tracks accuracy change. Where
   accuracy survives, bias can rise. Bias must therefore be reported next to accuracy.
3. **The harm is uneven across social categories, and calibration language changes ability more than bias.**

The Urdu-adapted models (Qalb, Alif) and the Urdu letter-label artifact are short sections, with details in the
appendix.

## Main body

| Exhibit | Content | Claim |
|---|---|---|
| **Table 1. Setup** | One row per variant: method, bits or sparsity, calibration data, memory (GB), perplexity change in en / ur / roman_ur (mean over models). Model list and data size in the caption. | context |
| **Table 2. Main results** | Rows: BF16, int8, NF4, AWQ, Wanda 25%, Wanda 50% (C4 calibration). Columns: for each language, flip %, Δ acc. (disamb.), Δ s_amb, Δ s_dis, averaged over models. The BF16 row shows absolute values. Each Δ cell also counts the models with a significant rise or fall (e.g. `−4.2 (↑1 ↓3)`). Full-width table. | 1, 2 |
| **Table 3. Calibration language** | Wanda 25% and 50%: *mixed minus C4*, per language: Δ perplexity, Δ acc. (disamb.), Δ s_amb, Δ s_dis, Δ flip %, averaged over models, with counts of models whose paired CI excludes 0. It does not repeat the absolute Wanda rows of Table 2. | 3 |
| **Figure 1. Flips** | (a) Flip rate by variant and language, with the BF16-rerun noise floor marked. (b) Flip rate by BF16 confidence bin per language, which shows the extra Urdu flips are near-ties. | 1 |
| **Figure 2. Bias vs ability** | One point per compressed run (model × variant × language): x = Δ acc. (disamb.), y = Δ s_amb, coloured by compression type. r = 0.83. | 2 |
| **Figure 3. Categories** | Heatmap: category × compression group (8-bit, 4-bit, prune 25%, prune 50%), mean Δ s_amb, one panel per language. | 3 |

## Appendix

| Table | Content |
|---|---|
| A1 Data | category × items, templates, ambiguous / disambiguated, negative / non-negative, scorable |
| A2 BF16 baselines | model × language: acc. ambig., acc. disamb., Unknown (disamb.), s_amb, s_dis with CIs, order consistency |
| A3 Full results | model × variant × language: Δ acc. ambig., Δ acc. disamb., Δ s_amb, Δ s_dis (CIs), flip %, McNemar p (Holm) |
| A4 Flip directions | flips into the biased answer, out of it, into Unknown, out of Unknown; noise floor rows |
| A5 Categories per model | the per-model numbers behind Figure 3 (opposite shifts cancel in the mean) |
| A6 Calibration per model | the per-model numbers behind Table 3 |
| A7 Language adaptation | Llama-3.1-8B vs Qalb vs Alif × variant × language |
| A8 Urdu letter labels | `ur` vs `ur_abc`: accuracy, order consistency, choice share per position |
| A9 Abstention | correlation of bias change with accuracy change and with Unknown-rate change |
| A10 Answer format | BF16 label mass and how often the most likely next token is not a label (end-of-text, newline), from the stored top-10 tokens. Gemma and Alif often fail the format; we discuss this as an instruction-following failure, not rescored. |

## How each number is computed

- **Answer:** each item is scored in 3 cyclic option orders. The answer is the option with the highest probability averaged over orders, each order's 3 label probabilities first renormalised.
- **Accuracy**, separately for ambiguous and disambiguated contexts. In ambiguous contexts the correct answer is "Unknown".
- **Unknown rate (over-abstention):** the share of *disambiguated* items answered "Unknown". On ambiguous items, the Unknown rate equals acc. ambig.
- **Bias scores (BBQ):**
  - A *biased* answer is the stereotyped group for a negative question, and the other group for a non-negative question.
  - `s_dis = 2 · biased / non-Unknown − 1` (disambiguated).
  - `s_amb = (1 − acc. ambig.) · (2 · biased / non-Unknown − 1)` (ambiguous).
  - Both are ×100; 0 means no bias.
  - 6,407 of the 6,432 items can be scored (`data/our_data/bias_targets.json`).
- **Flip rate:** the share of items whose answer differs from the same model's BF16 answer. Noise floor: BF16 vs a BF16 rerun with another batch size (Qwen, Gemma).
- **Uncertainty:** 95% cluster bootstrap over the 199 templates (2,000 resamples), paired against BF16. A model counts as a significant rise or fall when its paired CI excludes 0. McNemar tests on paired correctness, Holm-corrected, in A3.
- **Near-chance rule:** a model–language pair with disambiguated accuracy below 40% is marked †, and its bias scores are not interpreted. No pair falls below this at present; the lowest is Llama in Roman Urdu at 40.7.
- **Averages over models** in Tables 2–3 use the models that have the variant. AWQ has 4 models, because Gemma AWQ failed (tool limitation).

## Still needed

- Perplexity (Tables 1 and 3): `logs/wanda_meta/` and `logs/awq_meta/` are on the stopped machines.
- Figures: a plotting library. matplotlib is not installed; the script writes each figure's data as CSV, so the figures can be drawn in matplotlib or pgfplots.
- Decision: the main Urdu condition (`ur` or `ur_abc`). Gemma's format failures: decided (user, 26 Oct). Keep letter scoring and report the failure (Table A10) as a finding.
