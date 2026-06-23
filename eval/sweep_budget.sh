#!/usr/bin/env bash
# Phase 1.5 context-budget sweep (docs/phase1.5_context_budget.md §8).
#
# Runs concat + window_summary over the long + tool-medium datasets at each
# budget, so the result JSONs (tagged with `context_budget` and a `_b{N}k`
# run_id suffix) form a strategy × budget grid for the accuracy-vs-budget and
# efficiency-vs-budget curves. The backend must already be running.
#
#   source .venv/bin/activate
#   uvicorn main:app --port 8000      # in another shell
#   bash eval/sweep_budget.sh
#
# Override via env: BASE_URL, MODEL, BUDGETS, STRATEGIES, TIERS.
set -u
BASE="${BASE_URL:-http://localhost:8000}"
MODEL="${MODEL:-grok-4-fast}"
BUDGETS="${BUDGETS:-128000 32000 16000 8000 4000}"
STRATEGIES="${STRATEGIES:-concat window_summary}"
TIERS="${TIERS:-long tool-medium}"
OUT="${OUT:-eval/results}"

for strat in $STRATEGIES; do
  for tier in $TIERS; do
    for b in $BUDGETS; do
      echo ">>> strategy=$strat tier=$tier budget=$b out=$OUT"
      python eval/run.py --strategy "$strat" --tier "$tier" --model "$MODEL" \
        --context-budget "$b" --base-url "$BASE" --out "$OUT" 2>&1 | tail -4
      echo
    done
  done
done
echo "SWEEP DONE"
