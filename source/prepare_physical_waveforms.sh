#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
WORKERS="${WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/data/physical_waveforms_snr2_2-15hz}"
QC_SAMPLE_COUNT="${QC_SAMPLE_COUNT:-6}"
QC_SEED="${QC_SEED:-20260822}"

cd "${PROJECT_ROOT}"

# 1. Fetch missing exact NSLC/epoch responses and refuse to continue unless
# every selected three-component waveform has complete response coverage.
"${PYTHON_BIN}" eval/fetch_station_responses.py --default-network KO --strict

# 2. Convert raw digitizer counts to acceleration, then apply the common
# 2-15 Hz bandpass. Existing outputs resume only under an identical manifest.
"${PYTHON_BIN}" preprocessing/remove_instrument_response.py \
  --output-dir "${OUTPUT_DIR}" \
  --default-network KO \
  --workers "${WORKERS}"

# 3. Audit all outputs and render deterministic raw-vs-acceleration previews.
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/seismic-mpl-cache}" \
"${PYTHON_BIN}" preprocessing/qc_physical_waveforms.py \
  --corrected-dir "${OUTPUT_DIR}" \
  --raw-dir "${PROJECT_ROOT}/data/waveforms" \
  --manifest "${OUTPUT_DIR}/response_removal_manifest.json" \
  --output-dir "${OUTPUT_DIR}/qc" \
  --sample-count "${QC_SAMPLE_COUNT}" \
  --seed "${QC_SEED}"
