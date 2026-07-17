#!/usr/bin/env bash
# Realization statistics figure (GWM paper Fig. 7): running median +/- std and
# Shapiro-Wilk statistic of PGA over N realizations of a single scenario
# (default: the test record closest to the split's median M/R_hyp/Vs30).
# Extra args are passed through, e.g.:
#   bash source/evaluate_distributions.sh --example_index 4482  # other scenario
#   bash source/evaluate_distributions.sh --im pgv              # PGV variant
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python eval/evaluate_distributions.py all \
    --n_realizations 200 \
    --batch_size 50 \
    "$@"
