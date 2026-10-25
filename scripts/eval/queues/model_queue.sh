#!/usr/bin/env bash
# Exp 1 for any list of models: BF16, then NF4/INT8, then Wanda 50%/25% (prune, run, delete copy).
# Same logic as exp1_queue.sh, with the models taken from MODELS. Skips completed runs.
#
#   MODELS="qalb-1.0-8b" LANGS="en ur roman_ur ur_abc" bash scripts/eval/queues/model_queue.sh
cd /root/low_compute_fairness
MODELS="${MODELS:?set MODELS, e.g. MODELS=qalb-1.0-8b}"
LANGS="${LANGS:-en ur roman_ur}"
step() { echo "=== $(date "+%F %T") step: $*"; "$@"; local rc=$?; echo "=== $(date "+%F %T") step exit $rc: $*"; return $rc; }
complete() {  # all languages of model $1, variant $2 finished (per their meta.json)
  for l in $LANGS; do
    grep -q '"status": "complete"' "results/runs/$1__$2__$l/meta.json" 2>/dev/null || return 1
  done
}
run() {
  if complete "$1" "$2"; then echo "=== $(date "+%F %T") skip (complete): $1 $2"; return 0; fi
  step uv run python -u scripts/eval/run.py --model "$1" --variant "$2" --lang $LANGS
}
cleanup() {  # planned cleanup: keep the pruning record, drop the ~16 GB pruned weights
  mkdir -p logs/wanda_meta && cp "models/$1/wanda_meta.json" "logs/wanda_meta/$1.json" && rm -rf "models/$1" \
    && echo "=== $(date "+%F %T") deleted models/$1 (meta kept in logs/wanda_meta/)"
}
pruned() {
  local v="wanda$2" dir="models/$1-wanda$2"
  if complete "$1" "$v"; then echo "=== $(date "+%F %T") skip (complete): $1 $v"; [ -d "$dir" ] && cleanup "$1-$v"; return 0; fi
  if [ ! -f "$dir/wanda_meta.json" ]; then
    step uv run python -u scripts/compression/wanda_prune.py --model "$1" --sparsity "0.$2" || return 1
  fi
  run "$1" "$v" && cleanup "$1-$v"
}
for m in $MODELS; do run "$m" bf16; done
for m in $MODELS; do for v in nf4 int8; do run "$m" "$v"; done; done
for m in $MODELS; do for pct in 50 25; do pruned "$m" "$pct"; done; done
echo "=== $(date "+%F %T") MODEL QUEUE DONE: $MODELS"
