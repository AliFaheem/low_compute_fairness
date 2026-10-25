# Experiment log

Running record of what was run, where, what came out, and what was decided. **Read this first in a new session, and append to it after every run or decision.** Newest entries go at the bottom of the Log. Plan and protocol live in `CLAUDE.md` (section "Experiment plan"); this file is the history.

## Current state

- **Phase:** 3, full Exp 1 runs for Qwen3-8B and Gemma-4-E4B (letter scoring), one queue: tmux `exp1`, log `logs/exp1.log`, script `scripts/eval/queues/exp1_queue.sh`.
  - Order: Qwen bf16 → Gemma bf16 → Qwen nf4, int8 → Gemma nf4, int8 → Qwen wanda50, wanda25 → Gemma wanda50, wanda25.
  - Started 25 Oct 01:42; estimated 17–21 h in total.
- **To check progress in a new session:**
  - `ssh vast 'grep "=== \|complete:" /root/low_compute_fairness/logs/exp1.log'` lists steps, exit codes and finished runs;
  - `uv run python scripts/eval/status.py` shows the grid.
  - If the instance restarted, re-run the queue: it skips finished runs.
- **Results:** private HF dataset `alifaheem94/pakbbq-results`. Pilot runs are under `smoke/` (any `--limit` run); full runs will go under `runs/`.
- **Watch progress:** on Vast, `tail -f logs/<session>.log` or `tmux attach -t <session>`; from anywhere, `uv run python scripts/eval/status.py --watch`.
- **Pilots done:** Llama-3.1-8B, Qwen3-8B, Gemma-4-E4B (BF16, 500 items, all 3 languages).
- **Cost:** this host's bandwidth is expensive. Cached models on its disk: Llama-3.1-8B, Qwen3-8B-Base, Gemma-4-E4B, Qwen3-0.6B. Credit $2.47 as of 25 Oct ~00:55.
- **Open decisions:**
  - what to do with Llama-3.1-8B on `ur` and `roman_ur` (near chance);
  - **scoring method:** letters (current) vs answer-text log-likelihood. Gemma's low label mass is diagnosed: it writes the answer text, especially "Unknown" (`Nam…`), instead of the letter. Answer-text scoring is recommended and needs a new scorer plus a re-pilot.

## Log

### 25 Oct 2026: Vast setup

- Rented instance 57582615, an RTX 3090 Ti, Ampere, so it has native BF16. A Tesla V100 listing was rejected: no BF16.
- `./setup.sh --smoke` passed:
  - torch 2.14.1+cu126, transformers 5.20.2, bitsandbytes 0.50.2;
  - HF token works, and the results repo is private and writable;
  - all 9 models accessible, including gated Llama 3.1/3.2;
  - data `en`/`ur`/`roman_ur` downloaded, 6,432 items each, validated.
- Smoke test, Qwen3-0.6B-Base, 30 items per language: pipeline works, 66–140 rows/s, accuracy near chance as expected for 0.6B.

### 25 Oct 2026: pilot, Llama-3.1-8B BF16 (500 items, en / ur / roman_ur)

- Model load: 165 s on first download, 13 s cached. 16.1 GB on the GPU. Speed: en 40 rows/s, ur 17.6, roman_ur 23.4 (rows = items × 3 option orders).
- Results: see table below. English works (84% disambiguated). **Urdu and Roman Urdu fail**:
  - accuracy within any single option order is at chance (27–34%);
  - 0% of items get the same answer across the 3 orders;
  - label mass is 0.48 / 0.59.
- Diagnosis: the model barely reads Urdu or Roman Urdu content here, so letter priors dominate. One displayed slot never wins: `ا` in `ur`, `B` in `roman_ur`.
  - Prompts were checked and are built correctly.
  - Labels are single tokens: Llama ids `' ا'` 13258, `' ب'` 28946, `' پ'` 88517, `' A'` 362.
  - So this is a model limitation, not a bug.
- English: the model almost never picks "Unknown" (ambiguous accuracy 3.9%) and prefers displayed A (~55%).

### 25 Oct 2026: pilot, Qwen3-8B-Base BF16 (500 items, en / ur / roman_ur)

- Load 172 s, 16.4 GB. Speed: en 38 rows/s, ur 15.8, roman_ur 22.0.
- Works in all three languages:
  - disambiguated accuracy 92 / 80 / 64;
  - label mass ≥ 0.88;
  - order agreement 80 / 21 / 50%.
- So the Llama Urdu failure is model-specific; the protocol works for a model that reads Urdu.
- Roman Urdu is the weakest language for Qwen (64% disambiguated); `ur` does better than `roman_ur`.

### 25 Oct 2026: pilot, Gemma-4-E4B BF16 (500 items, en / ur / roman_ur)

- Load 164 s, 15.9 GB. Fastest model: en 52.5 rows/s, ur 43.4, roman_ur 35.5. Its tokenizer is efficient on Urdu script, so `ur` is faster than `roman_ur`. A full BF16 pass over 3 languages is about 23 min.
- Understands all three languages: disambiguated 86 / 76 / 66.
- Almost never picks "Unknown": ambiguous accuracy 0.4 / 0.8 / 5.5%.
- Low label mass: 0.59 / **0.12** / 0.30. Most next-token probability goes to something other than the 3 labels; to be diagnosed by checking Gemma's top next tokens after the answer cue.

### 25 Oct 2026: cost finding, bandwidth dominates

- Credit fell from $5.00 to $2.47 in about 50 min. Confirmed in Vast billing: almost all of it was **bandwidth for model downloads** (about 50 GB: three 8B models, torch, data), not GPU time.
- This host charges heavily per TB of traffic ("$/hr plus bandwidth" in the listing). GPU time for the same period was only about $0.2.
- Consequences:
  - Do not destroy this instance while its cached models are needed; re-downloading costs money again.
  - Avoid large uploads too: Wanda checkpoints (about 16 GB each) should not be uploaded to HF from here.
  - For future instances, filter on bandwidth price ($/TB) and compare total cost, not just $/hr.

### 25 Oct 2026: Gemma label-mass diagnosis (top-10 next tokens after the answer cue)

- Probability mass on the 3 labels: en 0.58–0.66, ur **0.06–0.13**, roman_ur 0.21–0.32. The rest goes to:
  1. formatting tokens: `<eos>`, `\n`, `\n\n` (in ur, `<eos>` ~0.2 and `\n\n` ~0.15);
  2. **the start of the answer text instead of its letter**: `Nam…` / `نام…` (= Namaloom / نامعلوم, i.e. "Unknown"), `Ch…`, `Bur…`, `چھوٹ…`.
- Implication: Gemma tends to express "Unknown" by writing the word, which letter scoring cannot see. Its near-zero ambiguous accuracy (0.4% en, 0.8% ur) is at least partly a scoring artifact. This distorts the ambiguous-context bias score.
- Proposal (not yet decided):
  - switch to **answer-text (cloze) scoring**: log-likelihood of each option's text as the continuation, as lm-eval-harness does for BBQ;
  - it also removes the letter-position bias seen with Llama (dead `ا` / `B` slot);
  - re-pilot all 3 models on 500 items with it before full runs.

  Alternative: a few-shot letter prompt (keeps position bias and departs from the authors' zero-shot setup).
- Diagnostic script used: `scratchpad/gemma_top_tokens.py`, copied to `/tmp` on Vast; not in the repo.

### 25 Oct 2026, 01:02: decision and start of Qwen3-8B Exp 1 full runs (letter scoring)

- Decided:
  - Llama is skipped for `ur`/`roman_ur`;
  - Qwen goes ahead now with the current **letter scoring**, because its label mass is ≥ 0.88 in all languages;
  - Gemma waits for the scoring decision.
- Caveat: if the project later switches to answer-text scoring for Gemma, Qwen must be re-scored the same way for cross-model comparability.
- Wanda pruning script `scripts/compression/wanda_prune.py` was written in another chat and tested on CPU with Qwen3-0.6B.
  - Pruned models are kept local in `models/` (git-ignored), not uploaded, because of bandwidth cost.
  - `configs/models.yaml` has `qwen3-8b.wanda50 = models/qwen3-8b-wanda50`.
  - Note: it re-runs the model from the start for each block (cost grows with depth squared), so about 2–3 h for an 8B model on the 3090 Ti, versus ~15 min for the official implementation.
- Queue: tmux session `qwen-exp1` on Vast, script `logs/qwen_queue.sh` (on the instance only), log `logs/qwen-exp1.log`.
  - Order: bf16 → nf4 → int8 → wanda_prune → wanda50, all with `--lang en ur roman_ur`, full 6,432 items.
  - Each step logs `=== … step exit N`; the end logs `QUEUE DONE`.
  - Estimated 6.5–9 h, about $1.7–2.5 GPU. Credit was $2.47 at start; asked the user to top up.
- Code at this point: branch `setup/initial`, commit `31dca4f`.

### 25 Oct 2026, 01:22: Gemma-4-E4B Exp 1 queued after Qwen (letter scoring)

- User decision: run Gemma right after Qwen, also with letter scoring.
- Caveat recorded: letter scoring undercounts Gemma's "Unknown" (it writes "Namaloom" instead of the letter). Its ambiguous-context bias scores need care; disambiguated accuracy and flips are fine. If answer-text scoring is adopted later, re-score both Qwen and Gemma.
- tmux `gemma-exp1` runs `logs/gemma_queue.sh` (on the instance), log `logs/gemma-exp1.log`.
  - It waits for `QUEUE DONE` in `logs/qwen-exp1.log`.
  - Then: bf16 → nf4 → int8 → wanda_prune → wanda50, 3 languages, full 6,432 items.
  - `configs/models.yaml`: `gemma-4-e4b.wanda50 = models/gemma-4-e4b-wanda50` (commit `aaffe76`).
- Estimate: Gemma 4–5 h, about $1–1.3. With Qwen, total about 11–14 h, $3–4 GPU.
- Progress at 01:22: Qwen bf16 `en` complete (19,296 rows, 40 rows/s); `ur` running at about 11–15 rows/s. Urdu is slower because Qwen tokenizes it to 2.4× more tokens: en 131, roman_ur 226, ur 315 tokens per prompt.

### 25 Oct 2026, 01:30: Wanda 75% added; queues restructured

- New variant `wanda75` (Wanda 75% unstructured) for qwen3-8b and gemma-4-e4b (commit `d1be34f`):
  - `run.py` now resolves any `wandaNN` variant from the model's config key;
  - `configs/runs.yaml` Exp 1 variants are now `[bf16, int8, nf4, wanda50, wanda75]`, so the grid has 60 runs.
  - Expect heavy degradation at 75%; it is the strong-compression point.
- Disk: each pruned copy is about 16 GB and four would not fit (57 GB free). Each copy is deleted after its 3 runs complete; its `wanda_meta.json` (perplexity before/after, per-layer sparsity) is kept in `logs/wanda_meta/` on the instance.
- The waiting Gemma queue (`gemma-exp1`) was replaced by tmux `after-qwen`, running `logs/after_qwen_queue.sh` (copy in `scripts/eval/queues/`) with log `logs/after-qwen.log`. It waits for the Qwen queue's `QUEUE DONE`, then:
  1. deletes `models/qwen3-8b-wanda50` once its runs are complete;
  2. prunes Qwen to 75% → runs wanda75 → deletes the copy;
  3. runs Gemma bf16 → nf4 → int8;
  4. prunes Gemma to 50% → runs wanda50 → deletes;
  5. prunes Gemma to 75% → runs wanda75 → deletes.
- **Logging bug in `qwen-exp1`:** its `step exit N` lines always say 0, because `$?` is read after `$(date)` in the same echo. Judge Qwen-queue steps by `complete:` lines or `results/runs/*/meta.json` status instead. The `after-qwen` queue records exit codes correctly.
- New estimate: Qwen queue 7–9 h, then `after-qwen` about 9.5–12 h (Qwen 75% about 3–3.75 h; Gemma about 6.5–8 h, since each Gemma prune is about 1.5–2 h). Total about 17–21 h, about $4.5–6 GPU at ~$0.25/h.

### 25 Oct 2026, 01:33: Wanda 75% replaced by Wanda 25%

- The user meant a 25% drop, so the second pruning level is **Wanda 25%** (light pruning), not 75% (commit `cb95956`). Nothing at 75% had run.
- `configs`: variant `wanda25`, local paths `models/<model>-wanda25`. The Exp 1 variants are now `[bf16, int8, nf4, wanda50, wanda25]` (60-run grid).
- `after-qwen` was restarted with the updated `scripts/eval/queues/after_qwen_queue.sh`: Qwen 25% after the Qwen queue, and Gemma 50% + 25%. The pruning cost per level is the same as before, so the time estimate is unchanged at about 17–21 h total.
- Progress at 01:33: Qwen bf16 `en` and `ur` complete; `roman_ur` running.

### 25 Oct 2026, 01:42: single queue, low-risk steps first

- User decision: run the low-risk steps first so most results arrive even if pruning fails. Order: **Qwen BF16 → Gemma BF16 → NF4/INT8 for both → Wanda 50%/25% for both**.
- The old sessions `qwen-exp1` (stopped with Ctrl-C during Qwen bf16 `roman_ur`; 11,376/19,296 rows were saved) and `after-qwen` (still waiting) were replaced by **one tmux session `exp1`**.
  - It runs `scripts/eval/queues/exp1_queue.sh` (commit `8df80db`), log `logs/exp1.log`.
  - It skips complete runs (per `meta.json`), records real exit codes, and only prunes if no pruned copy exists.
  - Each pruned copy is deleted after its runs complete (`wanda_meta.json` is kept in `logs/wanda_meta/`).
  - It is safe to restart at any time: `tmux new-session -d -s exp1 "bash scripts/eval/queues/exp1_queue.sh 2>&1 | tee -a logs/exp1.log"`.
- Resumed correctly: Qwen bf16 `roman_ur` continued from row 11,376; `en`/`ur` were already complete.
- Untested on this GPU: NF4/INT8 loading (low risk; checked at load time) and Wanda on 8B models (medium risk; only CPU-tested on 0.6B). These run last.
- Older queue scripts (`qwen_queue.sh`, `after_qwen_queue.sh`) are kept in `scripts/eval/queues/` for the record only.

### 25 Oct 2026, ~01:50: preds.jsonl now stores the top-10 next tokens

- Each row now also has `top`: the 10 most likely next tokens `[token id, text, log-prob]`, taken from the same forward pass (commit `069d5b6`). This lets us measure later how often a model "answers" in words, e.g. Gemma writing `Namaloom`/`نامعلوم` for "Unknown", without re-running. It adds about 4 MB per run.
- Pulled on Vast before Gemma started, so **all Gemma runs and all Qwen runs after bf16 (nf4, int8, wanda) have `top`**. Qwen bf16 (en, ur, roman_ur) does not: those were already running on the old code. Its label mass is ≈0.9, so it matters less.
- Tested on CPU with Qwen3-0.6B, 3 items: rows are correct and `accuracy.py` still reads them.

### 25 Oct 2026, 06:34: status, 24/30 Exp 1 runs done

- Complete (3 languages each, 19,296 rows, uploaded): **Qwen3-8B bf16, nf4, int8, wanda50, wanda25** and **Gemma-4-E4B bf16, nf4, int8**. Every step exited 0.
- Running: Gemma prune 50% (block 36/42), then the Gemma wanda50 runs, then the 25% prune and wanda25 runs. Expected finish about 07:45–08:00.
- Actual timings on the 3090 Ti were much faster than estimated:
  - Qwen bf16 roman_ur remainder about 5 min; Gemma bf16 23 min (3 languages);
  - Qwen nf4 43 min, Qwen int8 34 min; Gemma nf4 23 min, Gemma int8 23 min;
  - Wanda prune about 19–20 min per model (not 2–3 h); each Qwen wanda run 43 min.
  - Whole Exp 1 for 2 models is about 6.3 h, not 17–21 h.
- Wanda sanity checks (`logs/wanda_meta/*.json` on the instance), Qwen3-8B, C4 validation perplexity 7.163:
  - wanda25 → 7.252 (+1%), exactly 25% of pruned-layer weights zero;
  - wanda50 → 8.566 (+20%), exactly 50% zero.
- Pruned copies are deleted after their runs; disk steady at 54/109 GB.

### Pilot results so far (500 items each, BF16)

| Run | Acc | Ambig | Disambig | Same answer in all 3 orders | Median label mass | rows/s |
|---|---:|---:|---:|---:|---:|---:|
| llama-3.1-8b en | 43.4 | 3.9 | 84.1 | 36.8 | 0.86 | 40.1 |
| llama-3.1-8b ur | 29.2 | 2.4 | 56.9 | 0.0 | 0.48 | 17.6 |
| llama-3.1-8b roman_ur | 39.2 | 33.5 | 45.1 | 0.0 | 0.59 | 23.4 |
| qwen3-8b en | 65.8 | 40.6 | 91.9 | 80.4 | 0.95 | 38.4 |
| qwen3-8b ur | 48.4 | 18.1 | 79.7 | 21.0 | 0.88 | 15.8 |
| qwen3-8b roman_ur | 39.6 | 15.7 | 64.2 | 49.8 | 0.92 | 22.0 |
| gemma-4-e4b en | 42.6 | 0.4 | 86.2 | 56.4 | 0.59 | 52.5 |
| gemma-4-e4b ur | 37.6 | 0.8 | 75.6 | 4.6 | 0.12 | 43.4 |
| gemma-4-e4b roman_ur | 35.4 | 5.5 | 66.3 | 9.2 | 0.30 | 35.5 |

Estimated full runs (6,432 items × 3 orders = 19,296 rows) per BF16 variant, 3 languages, on the 3090 Ti:
- Qwen about 45 min, Llama about 40 min, Gemma about 23 min.
- Wanda runs at BF16 speed, plus about 15 min to prune once.
- NF4 is expected to be about 1.5–2× slower and INT8 about 2–4× slower (bitsandbytes). Not measured yet; a 100-item timing test is suggested before scheduling.
- Rough total for Qwen + Gemma, Exp 1, all 4 variants: 8–10 GPU hours.

### Next

- ~~Diagnose Gemma's low label mass~~: done, see the Gemma label-mass diagnosis entry.
- Decide on the scoring method. If answer-text: write the scorer (no GPU), then re-pilot the 3 models on 500 items (~15 min GPU).
- Time INT8 and NF4 on 100 items.
- Set up an unattended overnight queue (`run.py --machine` with models assigned in `configs/runs.yaml`), optionally with instance auto-stop.
- Decide on Llama `ur`/`roman_ur`:
  - (a) exclude from bias claims, per the plan's near-chance rule;
  - (b) few-shot prompt;
  - (c) score answer text instead of letters (cloze log-likelihood).
