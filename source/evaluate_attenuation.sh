#!/usr/bin/env bash
# SA attenuation figure (GWM paper Fig. 8): GWM median/std curves from
# realization ensembles on a hypocentral-distance grid, vs a GMM and the
# binned real data (M 2-3, Vs30 window around the scenario station).
# Extra args are passed through, e.g.:
#   bash source/evaluate_attenuation.sh --gmm bssa14          # swap the GMM
#   bash source/evaluate_attenuation.sh --station LAP         # other station
#   bash source/evaluate_attenuation.sh --limit 400 --dist_step 35 --n_realizations 6  # smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python eval/evaluate_attenuation.py all \
    --mag_lo 2.0 --mag_hi 3.0 \
    --periods 0.1,0.3 \
    --dist_min 10 --dist_max 120 --dist_step 5 \
    --n_realizations 10 \
    --batch_size 50 \
    "$@"
