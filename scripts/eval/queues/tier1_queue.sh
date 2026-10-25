#!/usr/bin/env bash
# Tier 1 GPU checks: Gemma NF4 diagnosis, then a BF16 noise floor (rerun with a different batch size).
# The rerun writes to runs/<model>__bf16__<lang>__rerun; compare its answers with the original BF16 run.
#
#   bash scripts/eval/queues/tier1_queue.sh
cd /root/low_compute_fairness
export PATH="$HOME/.local/bin:$PATH"
step() { echo "=== $(date "+%F %T") step: $*"; "$@"; local rc=$?; echo "=== $(date "+%F %T") step exit $rc: $*"; return $rc; }
step uv run python -u scripts/compression/check_quant.py --model gemma-4-e4b
for m in ${NOISE_MODELS:-qwen3-8b gemma-4-e4b}; do
  step uv run python -u scripts/eval/run.py --model "$m" --variant bf16 --lang en ur roman_ur ur_abc \
       --run-suffix rerun --token-budget 8192
done
echo "=== $(date "+%F %T") TIER1 DONE"
