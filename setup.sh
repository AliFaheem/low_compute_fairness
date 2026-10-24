#!/usr/bin/env bash
# One-time setup of a machine, safe to re-run:
#   ./setup.sh                    # install, download data, check everything
#   ./setup.sh --machine NAME     # check access only for the models assigned to NAME in configs/runs.yaml
#   ./setup.sh --smoke            # also run a 30-item smoke test with a tiny model
set -euo pipefail
cd "$(dirname "$0")"

SMOKE=0; CHECK_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1 ;;
    --machine) CHECK_ARGS+=(--machine "$2"); shift ;;
    --cpu) CHECK_ARGS+=(--cpu) ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done

if ! command -v uv >/dev/null; then
  echo "== Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Created .env: fill in HF_TOKEN, HF_DATA_REPO and HF_RESULTS_REPO, then run ./setup.sh again."
  exit 1
fi

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

echo "== Installing Python environment (uv.lock)"
uv sync --frozen

echo "== Downloading data"
uv run python scripts/data_prep/hf_data.py pull

echo "== Checking this machine"
uv run python scripts/eval/check_env.py "${CHECK_ARGS[@]}"

if [ "$SMOKE" = 1 ]; then
  echo "== Smoke test"
  DEVICE=cuda:0; uv run python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || DEVICE=cpu
  uv run python scripts/eval/run.py --model smoke-qwen3-0.6b --variant bf16 --lang en ur roman_ur --limit 30 --device "$DEVICE"
  uv run python scripts/eval/accuracy.py --smoke
fi
