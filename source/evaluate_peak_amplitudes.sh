#!/usr/bin/env bash
# Peak amplitude bias figure (GWM paper Fig. 4): observed vs diffusion
# synthetics and vs a GMM, binned over hypocentral distance.
# Extra args are passed through, e.g.:
#   bash source/evaluate_peak_amplitudes.sh --limit 1000   # quick smoke test
#   bash source/evaluate_peak_amplitudes.sh --gmm bssa14 --set_mode val
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python eval/evaluate_peak_amplitudes.py all \
    --gmm edwardsfah13 \
    --set_mode matched \
    --bin_km 5 \
    --num_workers 8 \
    --chunk 250 \
    "$@"
