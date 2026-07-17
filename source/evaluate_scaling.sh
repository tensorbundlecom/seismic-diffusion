#!/usr/bin/env bash
# SA scaling figure (GWM paper Fig. 9): GWM median/std vs magnitude (fixed
# distance) and vs Vs30 (fixed magnitude and distance), against real records
# from narrow selection windows. Extra args are passed through, e.g.:
#   bash source/evaluate_scaling.sh --station LAP        # other station
#   bash source/evaluate_scaling.sh --period 0.1         # other period
#   bash source/evaluate_scaling.sh --limit 400 --mag_step 1.75 --vs30_step 150 --n_realizations 6  # smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python eval/evaluate_scaling.py all \
    --period 0.3 \
    --dist_center 60 --dist_halfwidth 10 \
    --mag_min 1.5 --mag_max 3.5 --mag_step 0.25 \
    --scenario_mag 2.5 \
    --vs30_min 350 --vs30_max 650 --vs30_step 12.5 \
    --n_realizations 10 \
    --batch_size 50 \
    "$@"
