#!/usr/bin/env bash
# AWQ 4-bit (mixed calibration) for each model in MODELS: quantize, run all languages, delete the copy (planned
# cleanup; awq_meta.json kept in logs/awq_meta/). Uses the separate gptqmodel venv (AWQ_PY). Skips completed runs.
# Optionally waits for WAIT_FOR in WAIT_LOG first.
#
#   MODELS="qwen3-8b gemma-4-e4b qalb-1.0-8b" CALIB=logs/calibration/qalb-1.0-8b-wanda50mix.jsonl \
#     tmux new-session -d -s awq "bash scripts/eval/queues/awq_queue.sh 2>&1 | tee -a logs/awq.log"
cd /root/low_compute_fairness
MODELS="${MODELS:?set MODELS}"
CALIB="${CALIB:?set CALIB to a logs/calibration/*mix.jsonl file}"
LANGS="${LANGS:-en ur roman_ur ur_abc}"
AWQ_PY="${AWQ_PY:-/root/awq_test/.venv/bin/python}"
step() { echo "=== $(date "+%F %T") step: $*"; "$@"; local rc=$?; echo "=== $(date "+%F %T") step exit $rc: $*"; return $rc; }
complete() {
  for l in $LANGS; do
    grep -q '"status": "complete"' "results/runs/$1__awq__$l/meta.json" 2>/dev/null || return 1
  done
}
cleanup() {
  mkdir -p logs/awq_meta && cp "models/$1-awq/awq_meta.json" "logs/awq_meta/$1-awq.json" && rm -rf "models/$1-awq" \
    && echo "=== $(date "+%F %T") deleted models/$1-awq (meta kept in logs/awq_meta/)"
}
if [ -n "$WAIT_FOR" ]; then
  echo "=== $(date "+%F %T") waiting for $WAIT_FOR in $WAIT_LOG"
  until grep -q "$WAIT_FOR" "$WAIT_LOG" 2>/dev/null; do sleep 60; done
fi
for m in $MODELS; do
  if complete "$m"; then echo "=== $(date "+%F %T") skip (complete): $m awq"; continue; fi
  if [ ! -f "models/$m-awq/awq_meta.json" ]; then
    step "$AWQ_PY" -u scripts/compression/awq_quant.py --model "$m" --calib-file "$CALIB" || continue
  fi
  step "$AWQ_PY" -u scripts/eval/run.py --model "$m" --variant awq --lang $LANGS && cleanup "$m"
done
echo "=== $(date "+%F %T") AWQ QUEUE DONE: $MODELS"
