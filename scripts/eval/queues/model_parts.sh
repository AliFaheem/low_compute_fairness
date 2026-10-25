#!/usr/bin/env bash
# Full Exp 1 + calibration ablation for one model (MODEL), split in two parts so two machines can share it:
#   PART=quant  bf16, int8, nf4
#   PART=prune  wanda50, wanda25 (C4) and wanda50mix, wanda25mix
# Both parts first wait for WAIT_FOR in WAIT_LOG if set. MERGE=1 first merges a LoRA adapter model (e.g. alif-1.0-8b).
# Skips completed runs; pruned copies are deleted after their runs (planned cleanup).
#
#   MODEL=llama-3.1-8b PART=quant tmux new-session -d -s llama_q "bash scripts/eval/queues/model_parts.sh 2>&1 | tee -a logs/llama_quant.log"
cd /root/low_compute_fairness
export PATH="$HOME/.local/bin:$PATH"
M="${MODEL:?set MODEL, e.g. MODEL=llama-3.1-8b}"
PART="${PART:?set PART=quant or PART=prune}"
LANGS="${LANGS:-en ur roman_ur ur_abc}"
step() { echo "=== $(date "+%F %T") step: $*"; "$@"; local rc=$?; echo "=== $(date "+%F %T") step exit $rc: $*"; return $rc; }
complete() {
  for l in $LANGS; do
    grep -q '"status": "complete"' "results/runs/${M}__$1__$l/meta.json" 2>/dev/null || return 1
  done
}
run() {
  if complete "$1"; then echo "=== $(date "+%F %T") skip (complete): $M $1"; return 0; fi
  step uv run python -u scripts/eval/run.py --model "$M" --variant "$1" --lang $LANGS
}
cleanup() {  # planned cleanup: keep the pruning record, drop the ~16 GB pruned weights
  mkdir -p logs/wanda_meta && cp "models/$1/wanda_meta.json" "logs/wanda_meta/$1.json" && rm -rf "models/$1" \
    && echo "=== $(date "+%F %T") deleted models/$1 (meta kept in logs/wanda_meta/)"
}
pruned() {  # $1 = sparsity percent, $2 = c4 or mix
  local v="wanda$1"; [ "$2" = mix ] && v="wanda$1mix"
  local dir="models/$M-$v"
  if complete "$v"; then echo "=== $(date "+%F %T") skip (complete): $M $v"; [ -d "$dir" ] && cleanup "$M-$v"; return 0; fi
  if [ ! -f "$dir/wanda_meta.json" ]; then
    step uv run python -u scripts/compression/wanda_prune.py --model "$M" --sparsity "0.$1" --calib "$2" || return 1
  fi
  run "$v" && cleanup "$M-$v"
}
if [ "${MERGE:-0}" = 1 ]; then step uv run python -u scripts/model_prep/merge_lora.py --model "$M" || exit 1; fi
if [ -n "$WAIT_FOR" ]; then
  echo "=== $(date "+%F %T") waiting for $WAIT_FOR in $WAIT_LOG"
  until grep -q "$WAIT_FOR" "$WAIT_LOG" 2>/dev/null; do sleep 60; done
fi
if [ "$PART" = quant ]; then
  for v in bf16 int8 nf4; do run "$v"; done
else
  for pct in 50 25; do pruned "$pct" c4; done
  for pct in 50 25; do pruned "$pct" mix; done
fi
echo "=== $(date "+%F %T") $M $PART DONE"
