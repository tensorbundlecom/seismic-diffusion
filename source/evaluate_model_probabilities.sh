#!/usr/bin/env bash
# Model probability figure (GWM paper Fig. 10, Eq. 12): average probability
# each model assigns to the observed SA values, in magnitude-distance bins;
# GWM vs a GMM plus their ratio. The GMM enters at plot time only, so
# swapping it reuses all caches. Extra args are passed through, e.g.:
#   bash source/evaluate_model_probabilities.sh --gmm bssa14   # swap the GMM
#   bash source/evaluate_model_probabilities.sh plot --gmm atkinson15  # re-score only
#   bash source/evaluate_model_probabilities.sh --limit 400 --mag_step 1.0 --dist_step 55 --n_realizations 4  # smoke test
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python eval/evaluate_model_probabilities.py all \
    --period 0.3 \
    --mag_min 1.5 --mag_max 3.5 --mag_step 0.25 \
    --dist_min 10 --dist_max 120 --dist_step 10 \
    --n_realizations 20 \
    --batch_size 50 \
    "$@"
