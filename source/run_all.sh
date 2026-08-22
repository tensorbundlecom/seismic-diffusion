#!/usr/bin/env bash
# Run the full pipeline end to end: autoencoder -> embeddings -> diffusion.
# Reuses the per-stage scripts so params live in one place. Stops on first failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> [1/3] Training autoencoder"
cd "$ROOT/ML/autoencoder"
bash "$ROOT/source/train_autoencoder.sh"

echo "==> [2/3] Creating embeddings (uses the checkpoint selected in source/create_embeddings.sh)"
cd "$ROOT/ML/diffusion"
bash "$ROOT/source/create_embeddings.sh"

echo "==> [3/3] Training diffusion"
cd "$ROOT/ML/diffusion"
bash "$ROOT/source/train_diffusion.sh"

echo "==> Pipeline complete."
