#!/usr/bin/env bash
# Pilot the Urdu-capable candidates on 500 items x 3 languages, then finish Exp 1 (skips completed runs).
# Set PILOTS_ONLY=1 to skip the Exp 1 part (e.g. when another machine is finishing Exp 1).
#
#   tmux new-session -d -s exp1 "bash scripts/eval/queues/candidates_pilot.sh 2>&1 | tee -a logs/exp1.log"
cd /root/low_compute_fairness
for m in ministral-3-8b qalb-1.0-8b; do
  echo "=== $(date "+%F %T") step: pilot $m"
  uv run python -u scripts/eval/run.py --model "$m" --variant bf16 --lang en ur roman_ur --limit 500
  echo "=== $(date "+%F %T") step exit $?: pilot $m"
done
uv run python scripts/eval/accuracy.py --smoke
echo "=== $(date "+%F %T") PILOTS DONE"
[ "${PILOTS_ONLY:-0}" = 1 ] || bash scripts/eval/queues/exp1_queue.sh
