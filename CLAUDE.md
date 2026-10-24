# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research project (Group 4): **does model compression change social bias in LLMs, measured on PakBBQ in English, Urdu and Roman Urdu?** The plan lives in `Compression_Bias_PakBBQ_group 4.pptx`. It is a loose idea being refined, not a fixed spec. The core design: every compressed variant (INT8, NF4 4-bit, Wanda-pruned, distilled) is compared answer-by-answer against its own uncompressed BF16 original, using accuracy, BBQ bias scores (ambiguous/disambiguated), flip rate and per-group change.

The repo is at an early stage. Data prep and the Roman Urdu split are done; there is no evaluation runner or tests yet.

Scripts live in category folders under `scripts/`, one folder per pipeline stage: `data_prep/`, `transliteration_to_ru/`, and more as the project grows. Put every new script in the folder for its stage, creating the folder if needed; never put scripts directly in `scripts/`. Scripts are run from the repo root, and anything that resolves repo files from `__file__` must account for the extra folder level.

The environment is managed by `uv` (`pyproject.toml` / `uv.lock`). Run scripts with `uv run python scripts/<category>/<name>.py`.

Secrets go in git-ignored files: `.env` (variable names listed in `.env.example`) and `credentials.json`, a GCP service account. Never write their values into code, configs, results or docs.

## Roman Urdu split

**Status: done (25 Oct 2026).** `data/our_data/roman_ur/` holds all 6,432 items, passes `validate_splits.py`, and has no non-Latin words left. It contains only the 8 category files, like `en/` and `ur/`. Bookkeeping lives in `data/our_data/_build/roman_ur/`: `meta.json`, `answer_cache.json`, `failures.log`, `needs_cleanup.log`, `fallback_items.log` and `run.log`.

`scripts/transliteration_to_ru/transliterate_roman_urdu.py` turns `data/our_data/ur/` into `data/our_data/roman_ur/` using Gemini on Vertex AI (service account in `credentials.json`; default model `gemini-3.5-flash-lite`).
- `--work-dir` holds all bookkeeping (default `data/our_data/_build/roman_ur`), so `--out` stays data-only.
- Rate limits are handled with fallback model `gemini-2.5-flash-lite`: a 429 is retried at once on the other model. `--round-robin` alternates items between the two models.
- The final run used 8 workers with round-robin. 6+ workers on a single model hit 429s.
- Output records are identical to the Urdu input except `context`, `question` and `ans0`–`ans2`; downstream code relies on this.
- `scripts/data_prep/validate_splits.py` enforces this for every split against `en`: same files, row counts, ID order, key order and non-text fields, plus Latin-only text for `roman_ur`. The transliteration script also runs it at the end of each run. `answer_info` is compared on group tags only, because 8 source Urdu items (Language formality, ids 4308–4315) have an Urdu-script display name there.
- The system prompt is read from `prompts/roman_urdu_transliteration.txt`. `prompts/README.md` documents settings and prompt versions for the paper; the first 140 Age items used v1. Update that README whenever the prompt or settings change.
- It is resumable: re-run the same command to continue and to retry items listed in `failures.log`.
- Answer options are made consistent through `answer_cache.json` in the work dir: the same Urdu answer string always gets the same Roman spelling. Context and question spelling is intentionally left free to vary (temperature 0.7), like real Roman Urdu.
- The models leave non-Latin words in about 8–10% of items: Urdu, Devanagari, Bengali or Cyrillic letters, or accented Latin. These outputs are accepted and logged to `needs_cleanup.log`. Answer spellings with non-Latin words are never put in the answer cache.
- `scripts/transliteration_to_ru/clean_roman_urdu.py` fixes them:
  - `extract` adds leftover words to the review sheet `prompts/roman_urdu_word_fixes.csv`;
  - after the user reviews the `replacement` column, `apply` replaces those whole words in completed files.

  142 words (1,178 occurrences) were fixed this way.
- Sheet rows whose `word` contains a space are exact phrase fixes, added by hand for errors in Latin letters that `extract` cannot see. 4 were applied, e.g. `Islamabad terutama` → `Islamabadi`; the model had inserted an Indonesian word in 12 Regional items.
- More Latin-script errors of this kind may remain; a native-speaker spot check is still pending.
- Retries are bounded:
  - at most `--retries` API retries per item;
  - items that fail are tried again in at most `--max-failed-runs` runs (default 3; `--retry-failed` overrides);
  - the run stops after `--max-consecutive-failures` failed items in a row.
- Non-data files in a split folder must not end in `.jsonl`; loaders glob `*.jsonl`.
- Known source-data issue carried through: the Urdu answer option for "Muhajir" is مہاجرین ("refugees"), so the Roman option is "Muhajreen".

## Data

`data/our_data/` is committed: the `en`/`ur`/`roman_ur` splits, `subset_ids.json` and the `_build/` provenance. Only `data/PakBBQ/` is git-ignored: it is a nested git clone (136 MB) and is rebuilt with the clone command below.

- `data/PakBBQ/` is an untouched clone of https://github.com/Hashmat02/PakBBQ-A-Culturally-Adapted-Bias-Benchmark-for-QA (commit `c079dba`). Raw data is in `PakBBQ/PakBBQ_ENG/` and `PakBBQ/PakBBQ_UR/`, 17,180 items per language, 8 category JSONL files each. It also holds the authors' eval scripts and per-model results (`Analysis/`).
- `data/our_data/{en,ur}/` is our working copy, **already filtered** to a stratified subset: 6,432 items per language, i.e. min(total, 1000) per category. In it, `ENG-linguistic(language_formality_biases).jsonl` is renamed `Language_formality.jsonl`.
- GPU machines get `data/our_data/` from a **private** Hugging Face dataset (`HF_DATA_REPO` in `.env`), not from git: `uv run python scripts/data_prep/hf_data.py pull` (download + validate) and `push` (upload; refuses a public repo). PakBBQ has no license file, so the data is not published. The GitHub repo is public.
- `data/our_data/subset_ids.json` is the fixed subset (IDs per category, cap, seed). Treat it as the source of truth.

Rebuild from scratch:
```bash
git clone https://github.com/Hashmat02/PakBBQ-A-Culturally-Adapted-Bias-Benchmark-for-QA.git data/PakBBQ
git -C data/PakBBQ checkout c079dba
# copy PakBBQ_ENG -> data/our_data/en and PakBBQ_UR -> data/our_data/ur (renaming the linguistic file), then
python3 scripts/data_prep/make_subset.py          # writes subset_ids.json (seed 0, cap 1000)
# then filter our_data/*/*.jsonl to the IDs in subset_ids.json
```
`scripts/data_prep/make_subset.py` reads `--data` (default `data/our_data/en`). Run it only on the **full** data: re-running it on the already-filtered folder silently produces a different subset. It fills each of the 4 cells (ambig/disambig × neg/nonneg) round-robin over templates (`question_index`).

Facts about the data that matter for code:
- `id` is unique across all files, and EN and UR files are row-aligned with identical `id`s. One ID list therefore covers every language, including the planned Roman Urdu split.
- The "Unknown" option is always the literal `Unknown`. Its position (ans0/1/2) is balanced, and its tag in `answer_info` is lowercase `unknown`.
- There are 199 templates by `question_index`, unevenly expanded: up to 336 items from one template. Items within a template are correlated, so bootstrap confidence intervals over templates, not items.
- Category sizes are very unequal: Language formality has only 88 items.
- `data_inconsistency_notes.md` records tag errors, broken items, questionable name genders and Language formality translation problems in the source data. These are documented only, **not corrected**: we use the data as it is. How bias scoring handles the unmatched items is deferred; `scripts/scoring/bias_targets.py` and `annotations/*.csv` hold an unused draft of corrections.
- **macOS is case-insensitive:** never create `data/pakbbq` next to `data/PakBBQ`.

## Reproducing PakBBQ's protocol

The original prompt is in `data/PakBBQ/GPT_and_DeepSeek_eval.py/streamline_batch_pipeline.py`:
- system message "You are a Pakistani person. Answer the following questions according to the context. Only output the letter (A, B, or C)…";
- a `Context / Question / Options A–C` user turn;
- temperature 0, at most 5 output tokens.

Urdu uses the labels ا/ب/پ. Reuse this text verbatim for comparability.

Known problems in the authors' code. Do not copy them:
- No option permutation or majority vote is implemented, although the paper claims it.
- In `Analysis/analysis.py`, the bias score:
  - ignores question polarity: BBQ counts the stereotyped group as biased for `neg` questions and the non-stereotyped group for `nonneg`;
  - matches answer tags to `stereotyped_groups` by exact string (`lowSES` vs `low SES`, `Christian` vs `Christians`), so it almost never matches outside Age;
  - compares against `'Unknown'` while tags are `'unknown'`, so unknown answers stay in the denominator.

  Their **accuracy** figures are fine; their **bias scores are not**. Compare against the paper on accuracy only, and implement BBQ-correct scoring ourselves.
- The GPT pipeline's detailed CSV is built from gold labels, not model answers.

## Experiment plan

The deck is a starting point, not a spec: choices change for practical reasons (cost, tooling, time). Plan agreed 25 Oct 2026.

**Models are base (pretrained), not instruct.** Official base teacher→student pairs (NVIDIA Minitron, Llama 3.2 base) had no separate post-training, so distillation comparisons are clean. No fine-tuning or distillation training by us.

### Exp 1: quantization and pruning

Does compressing the same weights change bias? Every variant is compared with its own BF16 original.

| Model (base) | BF16 (ref) | INT8 | NF4 4-bit | Wanda 50% |
|---|---|---|---|---|
| Llama-3.1-8B | ✓ | ✓ | ✓ | ✓ |
| Qwen3-8B-Base | ✓ | ✓ | ✓ | ✓ |
| Gemma-4-E4B (`google/gemma-4-E4B`, base) | ✓ | ✓ | ✓ | ✓ |

- BF16 is each model's native precision and the only 16-bit variant (no FP16).
- INT8 and NF4 use bitsandbytes, applied at load time (no calibration, no stored checkpoints). Wanda 50% unstructured is the one variant that produces checkpoints to share.
- All fit one 24 GB GPU. Models differ in size (4B–8B); that is fine because every variant is compared only with its own original. Cross-model statements (e.g. "Gemma is more sensitive") must allow for size, since smaller models usually degrade more.
- Gemma-4-E4B: newest Gemma, not gated, better multilingual coverage than Gemma 2. 8B stored parameters (about half are per-layer embeddings), runs like a 4B; 18 of 42 layers share KV caches; needs transformers 5.x. The checkpoint is multimodal: load text only, and Wanda prunes only the language-model layers. Because bitsandbytes and Wanda leave embeddings in BF16, its compressed variants shrink less than Llama's; record real memory per variant. Fallback if the pilot shows library problems: Gemma-3-4B (`gemma-3-4b-pt`, gated). Gemma-4-12B needs a ≥40 GB GPU.
- Quantization: bitsandbytes at load time from the pinned BF16 weights, no calibration. INT8 = `load_in_8bit` (LLM.int8(), outliers >6 kept in 16-bit). NF4 = `load_in_4bit`, `nf4`, compute dtype BF16, no double quantization. Embeddings, `lm_head` and norms stay BF16 (bitsandbytes defaults).

### Exp 2: distillation

Does a student distilled from a teacher keep or change the teacher's bias?

| Teacher (BF16) | Official base students |
|---|---|
| Llama-3.1-8B (run shared with Exp 1) | Llama-3.1-Minitron-4B-Width, Llama-3.1-Minitron-4B-Depth, Llama-3.2-3B |
| Mistral-NeMo-12B-Base (optional, needs ≥40 GB GPU) | Mistral-NeMo-Minitron-8B-Base |

- Minitron models are pruned and distilled from exactly these teachers; Llama 3.2 base used Llama 3.1 8B/70B logits in pretraining. Distillation data still differs from the teacher's, so state that confound.
- Qwen3 has no distilled base student: its strong-to-weak distillation is post-training only.
- Size (model × language): Exp 1 is 36 runs, Exp 2 adds 9, so 45 in total, 51 with the Mistral pair. Llama-3.1-8B BF16 lets the paper compare 4-bit quantization and distillation at a similar size reduction.

### Protocol (both experiments)

- All 6,432 subset items in `en`, `ur`, `roman_ur`.
- Prompt text is in `prompts/pakbbq_eval_prompts.json` (documented in `prompts/README.md`): the authors' EN/UR prompt verbatim in their Gemini plain-text layout, plus an answer cue (`Answer:` / `جواب:` / `Jawab:`). Roman Urdu is our transliteration of the Urdu prompt. Labels: A/B/C for `en` and `roman_ur`, ا/ب/پ for `ur`.
- No generation: one forward pass per prompt, reading the logits of the 3 label tokens. Check each tokenizer encodes every label as one token.
- 3 cyclic option orders. Final answer = argmax of probabilities averaged over orders; also store majority vote.
- Log the probability mass on the 3 labels as a format-compliance check.
- Noise floor: answer flips between option orders, plus one BF16 rerun with a different batch size.
- Metrics, all recomputed from stored per-item probabilities: accuracy (ambig/disambig), BBQ-correct bias scores, flip rate against the original (with direction), shift in P(stereotyped answer), per-group change. Paired cluster bootstrap over templates; McNemar for flips.
- Report bias next to disambiguated accuracy and the unknown rate, so "more biased" is not confused with "weaker". Exclude a model–language pair from bias claims if its accuracy is near chance (33%).
- Optional extra: fairness instruction as a text prefix. Thinking mode is dropped (base models have none).

### Infrastructure (built)

How to set up a machine and run is in `README.md`. Key facts for code:
- `scripts/eval/run.py` scores one forward pass per prompt and stores, per (item, option order), the log-probs of the 3 label tokens in **original option order** (`ans0..ans2`) plus `mass` (full-vocabulary probability on the labels). Prompts are built by `scripts/eval/common.py` from `prompts/pakbbq_eval_prompts.json`; order k shows original option (pos + k) % 3 at display position pos.
- Left padding with explicit `position_ids`; `logits_to_keep=1`. Batches are by token budget (`--token-budget`, halved on OOM). Batched vs single scoring agrees to 4e-5 in FP32; in BF16 it differs by up to ~0.1 log-prob from rounding, so BF16 reruns with another batch size are the noise floor.
- Label tokens are checked at startup (single, distinct, stable). They are single tokens (" A"/" B"/" C", " ا"/" ب"/" پ") for all 9 configured models.
- Results are stored in the private HF dataset `HF_RESULTS_REPO` (`scripts/eval/hub.py`, CommitScheduler every 5 min + final upload), mirroring `results/runs/` and `results/smoke/`. Not in git: the GitHub repo is public. A run with no local files fetches what HF has first, so runs resume across machines. A per-run `fcntl` lock stops two processes scoring the same run (they would duplicate rows).
- `meta.json` records model revision, quantization config, linear-layer counts (int8/nf4 fail if no bnb layers), memory footprint, versions, GPU, host, git commit, data and prompt hashes, and one entry per session. A run is `complete` when it has every item in all 3 orders.
- Environment: Python 3.12, torch 2.14.1 (Linux: CUDA 12.6 build, drivers >= 525), transformers 5.20.2, bitsandbytes (Linux only). `smoke-qwen3-0.6b` in `configs/models.yaml` is for CPU smoke tests only.
- Not built yet: `scripts/compression/wanda_prune.py` (set `wanda50` per model in `configs/models.yaml` once pushed), and the full metrics/analysis step.

### Phases

0. Harness and scorer, no GPU: prompt builder, label scoring, run layout, metrics with a tested tag→`stereotyped_groups` mapping. Validate by re-scoring the authors' published predictions and matching their accuracy.
1. Pilot: Llama-3.1-8B BF16, `en` + `ur`, 500 items. Check label tokens, label mass, Urdu accuracy, speed.
2. BF16 references for all originals.
3. Exp 1 variants, then Exp 2 students.
4. Analysis, figures, Roman Urdu release.
