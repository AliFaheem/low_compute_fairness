# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Research question:** does model compression (quantization, pruning, distillation) change social bias in LLMs, measured on PakBBQ in English, Urdu and Roman Urdu? Every compressed variant is compared answer-by-answer with its own uncompressed BF16 original.

## Where things are (read these instead of guessing)

- **`experiments.md`** is the source of truth for current state: machines, queues, results, decisions. Read its "Current state" section at session start, and append a dated Log entry after every run, finding or decision.
- `docs/claude_reference.md`: full details (Roman Urdu pipeline, data rebuild, PakBBQ protocol, infrastructure internals). Read the section you need.
- `docs/compute_and_cost.md`: measured speeds, memory, pruning time, Vast costs; use for planning.
- `docs/related_work.md`: literature map and our contributions (verify before citing).
- `prompts/README.md`: transliteration and eval prompt provenance for the paper.
- `data_inconsistency_notes.md`: source-data problems (documented, not corrected).

## Conventions

- Scripts live in stage folders under `scripts/` (`data_prep/`, `transliteration_to_ru/`, `eval/`, `compression/`, `scoring/`, `analysis/`, `eval/queues/`); never at the top level. Run from the repo root: `uv run python scripts/<stage>/<name>.py`. Paths built from `__file__` need `../..` to reach the root.
- Environment: `uv` (`pyproject.toml`/`uv.lock`); Python 3.12, torch 2.14.1 (CUDA 12.6 on Linux), transformers 5.20.2. Do not add packages that change these pins. AWQ uses `gptqmodel` 7.7 in a separate venv (`/root/awq_test` on `vast2`); `llmcompressor` would downgrade them.
- Secrets live in `.env` and `credentials.json` (git-ignored). Never write their values anywhere.
- **The GitHub repo is public:** no data or results in git. Data lives in the private HF dataset `HF_DATA_REPO` (`scripts/data_prep/hf_data.py pull|push`); results in `HF_RESULTS_REPO`. `data/` and `results/` are git-ignored.
- **Ask before deleting** anything that isn't a temporary file (model caches, results, data, checkpoints), even to free disk. Planned cleanups the user approved, such as queues deleting pruned copies after their runs, are fine.
- Never edit a queue script while bash is executing it. Add work through new scripts or watcher sessions.
- **macOS is case-insensitive:** never create `data/pakbbq` next to `data/PakBBQ`. Non-data files in a split folder must not end in `.jsonl`; loaders glob `*.jsonl`.

## Data

- `data/our_data/{en,ur,roman_ur}/`: 6,432 items each (min(total, 1000) per category of PakBBQ, commit `c079dba`). The IDs in `subset_ids.json` are the fixed subset. `id`s are identical and row-aligned across splits; `validate_splits.py` enforces identical structure.
- `roman_ur` is our Gemini transliteration of `ur` (done; details in `docs/claude_reference.md`). `ur_abc` is a prompt variant: the `ur` data with Latin A/B/C labels (no separate data folder).
- 199 templates (`question_index`); items within a template are correlated, so bootstrap over templates. The "Unknown" option tag is lowercase `unknown`.
- Do not copy the PakBBQ authors' bias-score code: it ignores polarity, mis-matches tags and keeps "unknown" in the denominator. Their accuracy figures are fine.

## Experiment design (current; details in `experiments.md`)

- **Base (pretrained) models only.** Exp 1 main models: Qwen3-8B-Base, Gemma-4-E4B; side study Qalb-1.0-8B (Urdu-adapted Llama-3.1-8B). On hold: Llama-3.1-8B full Exp 1 and Exp 2 students (Llama-3.2-3B, Minitron-4B Width/Depth). Dropped: Ministral-3-8B.
- **Variants:** `bf16` (reference), `int8`, `nf4` (bitsandbytes at load time), `wanda50`, `wanda25` (English C4 calibration), `wanda50mix`, `wanda25mix` (calibration ⅓ C4 en + ⅓ FineWeb-2 Urdu + ⅓ Roman Urdu). Planned: AWQ 4-bit with mixed calibration only.
- **Languages:** `en`, `ur` (labels ا/ب/پ), `roman_ur`, `ur_abc`. Whether `ur` or `ur_abc` is the main Urdu result is decided at analysis time.
- **Scoring:** no generation. One forward pass per prompt; log-probs of the 3 label tokens over 3 cyclic option orders. Final answer = argmax of the order-averaged probability. `preds.jsonl` rows store `logp` (original option order), `mass` and `top` (top-10 next tokens).
- **Metrics:** accuracy (ambiguous/disambiguated); BBQ-correct bias scores (`s_dis`, `s_amb`); flips vs BF16; per-group change. Always report bias next to accuracy and the Unknown rate. Noise floor from a BF16 rerun (`--run-suffix rerun`).
- **Known issues:**
  - Gemma NF4 collapses (suspected bitsandbytes on Gemma-4 layers; under investigation);
  - Gemma rarely picks "Unknown" by letter (it writes it as a word);
  - the first Urdu label `ا` is almost never chosen by Llama/Qalb/Ministral.

## Running experiments

- GPU machines: Vast.ai, SSH hosts `vast` (Machine 1) and `vast2` (Machine 2) in `~/.ssh/config`; repo at `/root/low_compute_fairness`. Run work in tmux; logs go to `logs/`.
- `scripts/eval/run.py --model M --variant V --lang ...` is resumable, fetches existing results from HF and skips complete runs (uploading any that are complete locally but not on HF). Queues: `scripts/eval/queues/*.sh`.
- Pruning: `scripts/compression/wanda_prune.py` (`--calib c4|mix`). It records perplexity in en/ur/roman_ur, saves the calibration documents to `logs/calibration/`, and writes pruned copies to `models/` (local, never uploaded).
- Analysis so far: `scripts/analysis/gut_check.py` (accuracy, bias scores, flips).
