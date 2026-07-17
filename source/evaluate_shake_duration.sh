#!/usr/bin/env bash
# Shaking duration figure (GWM paper Fig. 6): cumulative Arias Intensity for
# one example event (100 realizations) plus D5-95 vs magnitude for the whole
# test split. Extra args are passed through, e.g.:
#   bash source/evaluate_shake_duration.sh --example_mag 3.4   # other example event
#   bash source/evaluate_shake_duration.sh --n_realizations 0  # skip panel a
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python eval/evaluate_shake_duration.py all \
    --example_mag 2.0 \
    --n_realizations 100 \
    --batch_size 50 \
    "$@"
