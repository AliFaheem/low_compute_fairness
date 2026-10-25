#!/usr/bin/env bash
# Runs after the Qwen queue: Qwen Wanda 25%, then the full Gemma-4-E4B Exp 1 (letter scoring).
# Pruned copies (~16 GB each) are deleted after their runs succeed, to fit the 109 GB disk;
# the wanda_meta.json of each copy is kept in logs/wanda_meta/.
cd /root/low_compute_fairness
echo "=== $(date "+%F %T") waiting for Qwen queue (QUEUE DONE in logs/qwen-exp1.log)"
until grep -q "QUEUE DONE" logs/qwen-exp1.log 2>/dev/null; do sleep 60; done
step() { echo "=== $(date "+%F %T") step: $*"; "$@"; local rc=$?; echo "=== $(date "+%F %T") step exit $rc: $*"; return $rc; }
cleanup() {  # keep the pruning record, drop the weights
  mkdir -p logs/wanda_meta && cp "models/$1/wanda_meta.json" "logs/wanda_meta/$1.json" && rm -rf "models/$1" \
    && echo "=== $(date "+%F %T") deleted models/$1 (meta kept in logs/wanda_meta/)"
}
PRUNE="uv run python -u scripts/compression/wanda_prune.py"
Q="uv run python -u scripts/eval/run.py --model qwen3-8b --lang en ur roman_ur"
G="uv run python -u scripts/eval/run.py --model gemma-4-e4b --lang en ur roman_ur"

complete() {  # all 3 languages of model $1, variant $2 finished (per their meta.json)
  for l in en ur roman_ur; do
    grep -q '"status": "complete"' "results/runs/$1__$2__$l/meta.json" 2>/dev/null || return 1
  done
}

# Qwen: free the 50% copy once its runs are complete (the Qwen queue log always says exit 0), then prune to 25%
if [ -d models/qwen3-8b-wanda50 ] && complete qwen3-8b wanda50; then
  cleanup qwen3-8b-wanda50
fi
step $PRUNE --model qwen3-8b --sparsity 0.25 && step $Q --variant wanda25 && cleanup qwen3-8b-wanda25

# Gemma
step $G --variant bf16
step $G --variant nf4
step $G --variant int8
step $PRUNE --model gemma-4-e4b --sparsity 0.5 && step $G --variant wanda50 && cleanup gemma-4-e4b-wanda50
step $PRUNE --model gemma-4-e4b --sparsity 0.25 && step $G --variant wanda25 && cleanup gemma-4-e4b-wanda25
echo "=== $(date "+%F %T") QUEUE DONE"
