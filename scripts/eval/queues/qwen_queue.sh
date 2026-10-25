#!/usr/bin/env bash
# Qwen3-8B Exp 1 queue (letter scoring), started 25 Oct 2026.
cd /root/low_compute_fairness
R="uv run python -u scripts/eval/run.py --model qwen3-8b --lang en ur roman_ur"
step() { echo "=== $(date "+%F %T") step: $*"; "$@"; echo "=== $(date "+%F %T") step exit $?: $*"; }
step $R --variant bf16
step $R --variant nf4
step $R --variant int8
step uv run python -u scripts/compression/wanda_prune.py --model qwen3-8b && step $R --variant wanda50
echo "=== $(date "+%F %T") QUEUE DONE"
