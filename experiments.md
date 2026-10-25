# Experiment log

Running record of what was run, where, what came out, and what was decided. **Read this first in a new session, and append to it after every run or decision.** Newest entries go at the bottom of the Log. Plan and protocol live in `CLAUDE.md` (section "Experiment plan"); this file is the history.

## Current state

- **Phase:** 1, pilot (500-item runs per model × language), in progress.
- **Machine:** Vast.ai instance 57582615, 1× RTX 3090 Ti 24 GB, 109 GB disk, no persistent volume (destroy wipes it; stop/start keeps it).
  - SSH: `ssh vast` (entry in `~/.ssh/config` on Ali's Mac, key `~/.ssh/vast_ed25519`, proxy `ssh4.vast.ai:10033`). Update the Port/HostName if the instance is recreated.
  - Repo at `/root/low_compute_fairness`, branch `setup/initial`. `.env` there holds only `HF_TOKEN`, `HF_DATA_REPO`, `HF_RESULTS_REPO`.
  - Models cache: `/workspace/.hf_home`.
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
