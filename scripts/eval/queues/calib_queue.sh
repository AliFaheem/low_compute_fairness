#!/usr/bin/env bash
# Wanda calibration-language ablation (variant wanda50mix): prune at 50% with mixed calibration
# (1/3 C4 en + 1/3 FineWeb-2 Urdu + 1/3 Roman Urdu), run all languages, delete the copy (planned cleanup).
# Then re-prune Qwen's English-calibrated 50% without saving, only to measure its Urdu/Roman Urdu perplexity
# (its original prune predates multi-language perplexity). Skips completed runs.
#
#   MODELS="qwen3-8b gemma-4-e4b qalb-1.0-8b" LANGS="en ur roman_ur ur_abc" bash scripts/eval/queues/calib_queue.sh
#   MODELS=qalb-1.0-8b PPLCHECK=0 bash scripts/eval/queues/calib_queue.sh   # e.g. on a second machine
cd /root/low_compute_fairness
MODELS="${MODELS:-qwen3-8b gemma-4-e4b qalb-1.0-8b}"
LANGS="${LANGS:-en ur roman_ur ur_abc}"
PPLCHECK="${PPLCHECK:-1}"   # 0 on machines that should not run the Qwen perplexity check
step() { echo "=== $(date "+%F %T") step: $*"; "$@"; local rc=$?; echo "=== $(date "+%F %T") step exit $rc: $*"; return $rc; }
complete() {
  for l in $LANGS; do
    grep -q '"status": "complete"' "results/runs/$1__$2__$l/meta.json" 2>/dev/null || return 1
  done
}
cleanup() {  # planned cleanup: keep the pruning record, drop the ~16 GB pruned weights
  mkdir -p logs/wanda_meta && cp "models/$1/wanda_meta.json" "logs/wanda_meta/$1.json" && rm -rf "models/$1" \
    && echo "=== $(date "+%F %T") deleted models/$1 (meta kept in logs/wanda_meta/)"
}
for m in $MODELS; do
  dir="models/$m-wanda50mix"
  if complete "$m" wanda50mix; then echo "=== $(date "+%F %T") skip (complete): $m wanda50mix"; continue; fi
  if [ ! -f "$dir/wanda_meta.json" ]; then
    step uv run python -u scripts/compression/wanda_prune.py --model "$m" --sparsity 0.5 --calib mix || continue
  fi
  step uv run python -u scripts/eval/run.py --model "$m" --variant wanda50mix --lang $LANGS && cleanup "$m-wanda50mix"
done
if [ "$PPLCHECK" = 1 ] && [ ! -f logs/wanda_meta/qwen3-8b-wanda50-pplcheck.json ]; then
  step uv run python -u scripts/compression/wanda_prune.py --model qwen3-8b --sparsity 0.5 --no-save \
       --out models/qwen3-8b-wanda50-pplcheck \
    && mkdir -p logs/wanda_meta && mv models/qwen3-8b-wanda50-pplcheck/wanda_meta.json logs/wanda_meta/qwen3-8b-wanda50-pplcheck.json \
    && rmdir models/qwen3-8b-wanda50-pplcheck
fi
echo "=== $(date "+%F %T") CALIB QUEUE DONE"
