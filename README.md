# low_compute_fairness

Does model compression (8-bit, 4-bit, pruning, distillation) change social bias in LLMs? Measured on PakBBQ in English, Urdu and Roman Urdu. The plan is in `CLAUDE.md` (section "Experiment plan").

Code is in this repo. Data and results are in two **private** Hugging Face datasets, so a GPU machine needs only this repo and a `.env` file.

## Set up a GPU machine

Needs Linux with an NVIDIA GPU (driver >= 525) and git.

```bash
git clone https://github.com/AliFaheem/low_compute_fairness.git
cd low_compute_fairness
./setup.sh            # first run creates .env and stops
# edit .env:
#   HF_TOKEN=hf_...                           # token with write access to the two datasets
#   HF_DATA_REPO=alifaheem94/pakbbq-subset
#   HF_RESULTS_REPO=alifaheem94/pakbbq-results
./setup.sh --smoke    # installs uv + Python env, downloads data, checks GPU/models, 30-item test
```

`setup.sh` is safe to re-run. It fails with a message if something is missing: no GPU, token not set, a model license not accepted on huggingface.co (Llama models are gated).

## Run experiments

Assign models to machines in `configs/runs.yaml` (`machines:`), commit, `git pull` on each machine, then:

```bash
uv run python scripts/eval/run.py --machine <name>                     # all runs of this machine's models
uv run python scripts/eval/run.py --model qwen3-8b --variant nf4 --lang en ur roman_ur   # or explicit runs
uv run python scripts/eval/status.py                                   # progress of the whole grid, all machines
uv run python scripts/eval/accuracy.py --fetch                         # quick accuracy table of finished runs
```

Run long jobs inside `tmux` or `nohup` so they survive a closed terminal.

- **Resumable.** Results are uploaded to `HF_RESULTS_REPO` every 5 minutes and at the end. If a machine stops, re-run the same command on it or on any other machine: finished rows are downloaded and skipped.
- **One process per run.** A second process on the same run on the same machine refuses to start. Never assign one model to two machines.
- **Out of memory** halves the batch automatically. Start lower on small GPUs with `--token-budget 8192`.
- Variants: `bf16`, `int8`, `nf4` (bitsandbytes, applied at load time) and `wanda50` (once the pruned checkpoints exist). Model versions are pinned in `configs/models.yaml`.

## Layout

| Path | What |
|---|---|
| `configs/models.yaml` | Models, pinned revisions, how each variant loads |
| `configs/runs.yaml` | Experiment grid and machine assignment |
| `scripts/eval/run.py` | Scores a model variant: label probabilities per item and option order |
| `scripts/eval/status.py`, `accuracy.py`, `check_env.py` | Progress, quick results, machine checks |
| `scripts/data_prep/hf_data.py` | Push/pull the data to/from the private HF dataset |
| `prompts/pakbbq_eval_prompts.json` | Evaluation prompt per language (see `prompts/README.md`) |
| `results/runs/<model>__<variant>__<lang>/` | `preds.jsonl` + `meta.json` per run (synced with HF, not git) |
