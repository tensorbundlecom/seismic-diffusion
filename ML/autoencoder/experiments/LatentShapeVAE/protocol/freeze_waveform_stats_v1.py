#!/usr/bin/env python3
"""
Freeze LatentShapeVAE V1 train-only global waveform normalization stats.

Output:
  - protocol/waveform_stats_v1.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# protocol -> LatentShapeVAE -> experiments -> autoencoder -> ML -> repo
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.LatentShapeVAE.core.waveform_dataset import WaveformDataset, collate_fn_waveform


def _read_yaml(path: Path) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze train-only waveform normalization stats for LatentShapeVAE.")
    parser.add_argument(
        "--data-config",
        default="ML/autoencoder/experiments/LatentShapeVAE/configs/data_protocol_v1.yaml",
    )
    parser.add_argument(
        "--frozen-split-manifest",
        default="ML/autoencoder/experiments/LatentShapeVAE/protocol/frozen_event_splits_v1.json",
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-files", type=int, default=0, help="0 means all train files.")
    args = parser.parse_args()

    data_cfg = _read_yaml(Path(args.data_config))
    manifest = _read_json(Path(args.frozen_split_manifest))
    train_file_list = _read_lines(Path(manifest["splits"]["train_files"]["file"]))
    if not train_file_list:
        raise RuntimeError("Train split is empty. Freeze splits first.")
    if args.max_files > 0:
        train_file_list = train_file_list[: args.max_files]

    preprocess = data_cfg.get("preprocess", {})
    band = preprocess.get("bandpass", {})
    segment_length = int(data_cfg.get("segment_length", 7001))
    sample_rate_hz = float(data_cfg.get("sample_rate_hz", 100.0))

    ds = WaveformDataset(
        file_paths=train_file_list,
        segment_length=segment_length,
        sample_rate_hz=sample_rate_hz,
        preprocess_demean=bool(preprocess.get("demean", True)),
        preprocess_detrend=bool(preprocess.get("detrend", True)),
        bandpass_enabled=bool(band.get("enabled", True)),
        bandpass_freqmin=float(band.get("freqmin", 0.5)),
        bandpass_freqmax=float(band.get("freqmax", 20.0)),
        bandpass_corners=int(band.get("corners", 4)),
        bandpass_zerophase=bool(band.get("zerophase", True)),
        normalization_stats_file=None,
        allow_missing_stats=True,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn_waveform,
        drop_last=False,
    )

    ch_sum = np.zeros((3,), dtype=np.float64)
    ch_sumsq = np.zeros((3,), dtype=np.float64)
    n_total = 0
    n_ok = 0
    n_batches = 0
    for batch in loader:
        x, _meta = batch
        n_batches += 1
        if x is None:
            continue
        arr = x.detach().cpu().numpy().astype(np.float64)  # [B, 3, T]
        ch_sum += arr.sum(axis=(0, 2))
        ch_sumsq += np.square(arr).sum(axis=(0, 2))
        n_total += arr.shape[0] * arr.shape[2]
        n_ok += arr.shape[0]
        if args.log_every > 0 and n_batches % args.log_every == 0:
            print(f"[INFO] processed batches={n_batches} files_ok={n_ok}/{len(train_file_list)}", flush=True)

    n_err = len(train_file_list) - n_ok
    if n_ok <= 0 or n_total <= 0:
        raise RuntimeError("No valid training waveform found to compute stats.")

    mean = ch_sum / float(n_total)
    var = (ch_sumsq / float(n_total)) - np.square(mean)
    var = np.clip(var, 1e-10, None)
    std = np.sqrt(var)

    out_path = Path(
        args.output
        or data_cfg.get("normalization", {}).get(
            "stats_file",
            "ML/autoencoder/experiments/LatentShapeVAE/protocol/waveform_stats_v1.json",
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": "v1",
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": Path(args.frozen_split_manifest).as_posix(),
        "n_files_total_in_train_list": len(train_file_list),
        "n_files_used": int(n_ok),
        "n_files_failed": int(n_err),
        "segment_length": int(segment_length),
        "sample_rate_hz": float(sample_rate_hz),
        "channels": ["E", "N", "Z"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "preprocess": {
            "demean": bool(preprocess.get("demean", True)),
            "detrend": bool(preprocess.get("detrend", True)),
            "bandpass": {
                "enabled": bool(band.get("enabled", True)),
                "freqmin": float(band.get("freqmin", 0.5)),
                "freqmax": float(band.get("freqmax", 20.0)),
                "corners": int(band.get("corners", 4)),
                "zerophase": bool(band.get("zerophase", True)),
            },
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("[INFO] Waveform stats written:", out_path)
    print("[INFO] mean (E,N,Z):", [round(x, 8) for x in mean.tolist()])
    print("[INFO] std  (E,N,Z):", [round(x, 8) for x in std.tolist()])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
