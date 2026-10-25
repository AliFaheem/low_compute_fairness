#!/usr/bin/env bash
# Alif-1.0-8B smoke test: merge its LoRA adapter on CPU now (GPU stays with the running queue), then wait until the
# GPU is free (WAIT_FOR marker in WAIT_LOG) and pilot 500 items x 4 languages.
#
#   tmux new-session -d -s alif "bash scripts/eval/queues/alif_pilot.sh 2>&1 | tee -a logs/alif.log"
cd /root/low_compute_fairness
export PATH="$HOME/.local/bin:$PATH"
step() { echo "=== $(date "+%F %T") step: $*"; "$@"; local rc=$?; echo "=== $(date "+%F %T") step exit $rc: $*"; return $rc; }
step uv run python -u scripts/model_prep/merge_lora.py --model alif-1.0-8b || exit 1
WAIT_LOG=${WAIT_LOG:-logs/tier1.log}; WAIT_FOR=${WAIT_FOR:-TIER1 DONE}
echo "=== $(date "+%F %T") waiting for $WAIT_FOR in $WAIT_LOG"
until grep -q "$WAIT_FOR" "$WAIT_LOG" 2>/dev/null; do sleep 60; done
step uv run python -u scripts/eval/run.py --model alif-1.0-8b --variant bf16 --lang en ur roman_ur ur_abc --limit 500
step uv run python scripts/eval/accuracy.py --smoke
echo "=== $(date "+%F %T") ALIF PILOT DONE"
