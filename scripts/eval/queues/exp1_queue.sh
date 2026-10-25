#!/usr/bin/env bash
# Exp 1 queue for Qwen3-8B and Gemma-4-E4B (letter scoring), low-risk steps first:
# BF16 for both, then NF4/INT8 for both, then Wanda pruning (untested on GPU) at 50% and 25%.
# Runs resume from saved rows, so restarting this queue skips finished work.
# Pruned copies (~16 GB each) are deleted after their runs complete, to fit the 109 GB disk;
# the wanda_meta.json of each copy is kept in logs/wanda_meta/.
#
#   tmux new-session -d -s exp1 "bash scripts/eval/queues/exp1_queue.sh 2>&1 | tee -a logs/exp1.log"
#   LANGS="en ur roman_ur ur_abc" bash scripts/eval/queues/exp1_queue.sh   # also the Urdu A/B/C variant
cd /root/low_compute_fairness
LANGS="${LANGS:-en ur roman_ur}"   # e.g. LANGS="en ur roman_ur ur_abc"
step() { echo "=== $(date "+%F %T") step: $*"; "$@"; local rc=$?; echo "=== $(date "+%F %T") step exit $rc: $*"; return $rc; }
complete() {  # all languages of model $1, variant $2 finished (per their meta.json)
  for l in $LANGS; do
    grep -q '"status": "complete"' "results/runs/$1__$2__$l/meta.json" 2>/dev/null || return 1
  done
}
run() {  # run model $1 variant $2 unless already complete
  if complete "$1" "$2"; then echo "=== $(date "+%F %T") skip (complete): $1 $2"; return 0; fi
  step uv run python -u scripts/eval/run.py --model "$1" --variant "$2" --lang $LANGS
}
cleanup() {  # keep the pruning record, drop the weights
  mkdir -p logs/wanda_meta && cp "models/$1/wanda_meta.json" "logs/wanda_meta/$1.json" && rm -rf "models/$1" \
    && echo "=== $(date "+%F %T") deleted models/$1 (meta kept in logs/wanda_meta/)"
}
pruned() {  # model $1, sparsity percent $2: prune if needed, run, delete the copy
  local v="wanda$2" dir="models/$1-wanda$2"
  if complete "$1" "$v"; then echo "=== $(date "+%F %T") skip (complete): $1 $v"; [ -d "$dir" ] && cleanup "$1-$v"; return 0; fi
  if [ ! -f "$dir/wanda_meta.json" ]; then
    step uv run python -u scripts/compression/wanda_prune.py --model "$1" --sparsity "0.$2" || return 1
  fi
  run "$1" "$v" && cleanup "$1-$v"
}

# 1. BF16 references for both models (tested in the pilots)
for m in qwen3-8b gemma-4-e4b; do run "$m" bf16; done

# 2. Quantized variants (bitsandbytes; quantization is checked at load time)
for m in qwen3-8b gemma-4-e4b; do
  for v in nf4 int8; do run "$m" "$v"; done
done

# 3. Higher risk: Wanda pruning
for m in qwen3-8b gemma-4-e4b; do
  for pct in 50 25; do pruned "$m" "$pct"; done
done

echo "=== $(date "+%F %T") QUEUE DONE"
