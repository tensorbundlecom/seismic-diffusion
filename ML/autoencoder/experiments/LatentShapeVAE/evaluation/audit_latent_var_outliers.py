#!/usr/bin/env python3
"""
Audit latent posterior variance/log-variance outliers for VAE checkpoints.

Purpose:
- Identify sample-level posterior outliers (max logvar / max var).
- Summarize where outliers come from (event/station/file).
- Provide split-wise diagnostics (test, ood_event).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# evaluation -> LatentShapeVAE -> experiments -> autoencoder -> ML -> repo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))

from ML.autoencoder.experiments.LatentShapeVAE.core.model_vae_waveform import WaveformDiagVAE
from ML.autoencoder.experiments.LatentShapeVAE.core.waveform_dataset import (
    WaveformDataset,
    collate_fn_waveform,
)


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


def _iter_checkpoint_paths(arg_list: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for arg in arg_list:
        p = Path(arg)
        if p.exists():
            out.append(p)
            continue
        for q in sorted(Path(".").glob(arg)):
            if q.is_file():
                out.append(q)

    dedup: List[Path] = []
    seen = set()
    for p in out:
        k = p.resolve().as_posix()
        if k not in seen:
            dedup.append(p)
            seen.add(k)
    return dedup


def _build_dataset(files: List[str], data_cfg: Dict, normalization_stats_file: str) -> WaveformDataset:
    preprocess = data_cfg.get("preprocess", {})
    band = preprocess.get("bandpass", {})
    return WaveformDataset(
        file_paths=files,
        segment_length=int(data_cfg.get("segment_length", 7001)),
        sample_rate_hz=float(data_cfg.get("sample_rate_hz", 100.0)),
        preprocess_demean=bool(preprocess.get("demean", True)),
        preprocess_detrend=bool(preprocess.get("detrend", True)),
        bandpass_enabled=bool(band.get("enabled", True)),
        bandpass_freqmin=float(band.get("freqmin", 0.5)),
        bandpass_freqmax=float(band.get("freqmax", 20.0)),
        bandpass_corners=int(band.get("corners", 4)),
        bandpass_zerophase=bool(band.get("zerophase", True)),
        normalization_stats_file=normalization_stats_file,
    )


def _parse_thresholds(th_str: str) -> List[float]:
    vals = []
    for tok in th_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    if not vals:
        raise ValueError("Threshold list is empty.")
    return sorted(vals)


def _safe_frac(num: int, den: int) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _summarize_group_stats(group_rows: Dict[str, Dict], ths: List[float]) -> List[Dict]:
    out = []
    for key, st in group_rows.items():
        row = {
            "group_key": key,
            "n_samples": st["n_samples"],
            "max_var_max": st["max_var_max"],
            "max_logvar_max": st["max_logvar_max"],
        }
        for th in ths:
            k = f"cnt_max_var_gt_{th:g}"
            row[k] = st[k]
            row[f"frac_max_var_gt_{th:g}"] = _safe_frac(st[k], st["n_samples"])
        out.append(row)
    out.sort(key=lambda r: r["max_var_max"], reverse=True)
    return out


def _build_model_from_ckpt(ckpt: Dict, data_cfg: Dict, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    run_cfg = dict(ckpt.get("config", {}))
    mode = str(run_cfg.get("ablation_mode", ""))
    if mode != "vae":
        raise ValueError(f"Checkpoint ablation_mode={mode} is not supported by this audit (expects 'vae').")
    latent_dim = int(run_cfg["latent_dim"])
    backbone = str(run_cfg["backbone"])
    logvar_mode = str(run_cfg.get("logvar_mode", "legacy"))
    logvar_min = float(run_cfg.get("logvar_min", -30.0))
    logvar_max = float(run_cfg.get("logvar_max", 20.0))
    input_length = int(data_cfg.get("segment_length", 7001))
    model = WaveformDiagVAE(
        in_channels=3,
        input_length=input_length,
        latent_dim=latent_dim,
        backbone=backbone,
        logvar_mode=logvar_mode,
        logvar_min=logvar_min,
        logvar_max=logvar_max,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, run_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit latent posterior outliers for VAE checkpoints.")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument(
        "--data_config",
        default="ML/autoencoder/experiments/LatentShapeVAE/configs/data_protocol_v1.yaml",
    )
    parser.add_argument(
        "--frozen_split_manifest",
        default="ML/autoencoder/experiments/LatentShapeVAE/protocol/frozen_event_splits_v1.json",
    )
    parser.add_argument("--normalization_stats_file", default="")
    parser.add_argument("--splits", nargs="+", default=["test", "ood_event"], choices=["test", "ood_event"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--var_thresholds", default="10,1000,100000")
    parser.add_argument("--logvar_clip_min", type=float, default=-30.0)
    parser.add_argument("--logvar_clip_max", type=float, default=40.0)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    ckpt_paths = _iter_checkpoint_paths(args.checkpoints)
    if not ckpt_paths:
        raise RuntimeError("No checkpoint found from --checkpoints")

    var_ths = _parse_thresholds(args.var_thresholds)
    logvar_ths = [float(np.log(v)) for v in var_ths]

    data_cfg = _read_yaml(Path(args.data_config))
    manifest = _read_json(Path(args.frozen_split_manifest))
    norm_file = args.normalization_stats_file or data_cfg.get("normalization", {}).get("stats_file", "")
    if not norm_file:
        raise ValueError("normalization_stats_file not set in args or data config.")

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"ML/autoencoder/experiments/LatentShapeVAE/results/latent_var_outlier_audit_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: List[Dict] = []

    for split in args.splits:
        split_key = f"{split}_files"
        if split_key not in manifest.get("splits", {}):
            raise ValueError(f"Split key not found in manifest: {split_key}")
        split_files = _read_lines(Path(manifest["splits"][split_key]["file"]))
        if not split_files:
            raise RuntimeError(f"Split {split} is empty.")

        ds = _build_dataset(split_files, data_cfg=data_cfg, normalization_stats_file=norm_file)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_fn_waveform,
        )

        for ckpt_path in ckpt_paths:
            t0 = time.time()
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model, run_cfg = _build_model_from_ckpt(ckpt, data_cfg=data_cfg, device=device)
            run_name = str(run_cfg.get("run_name", ckpt_path.stem.replace("_best", "")))
            latent_dim = int(run_cfg["latent_dim"])
            run_out = out_dir / split / run_name
            run_out.mkdir(parents=True, exist_ok=True)

            sample_rows: List[Dict] = []
            sum_var = np.zeros((latent_dim,), dtype=np.float64)
            n_samples = 0
            total_latent_values = 0

            dim_cnt_gt = {th: 0 for th in var_ths}
            sample_cnt_gt = {th: 0 for th in var_ths}

            max_var_global = 0.0
            max_logvar_global = -np.inf

            event_stats = defaultdict(
                lambda: {"n_samples": 0, "max_var_max": 0.0, "max_logvar_max": -np.inf, **{f"cnt_max_var_gt_{th:g}": 0 for th in var_ths}}
            )
            station_stats = defaultdict(
                lambda: {"n_samples": 0, "max_var_max": 0.0, "max_logvar_max": -np.inf, **{f"cnt_max_var_gt_{th:g}": 0 for th in var_ths}}
            )

            with torch.no_grad():
                for batch in loader:
                    x, meta = batch
                    if x is None:
                        continue
                    x = x.to(device, non_blocking=use_cuda)
                    _mu, logvar = model.encode(x)

                    lv = logvar.detach().cpu().numpy().astype(np.float64)
                    v = np.exp(np.clip(lv, args.logvar_clip_min, args.logvar_clip_max))

                    bsz = lv.shape[0]
                    n_samples += bsz
                    total_latent_values += bsz * latent_dim
                    sum_var += v.sum(axis=0)

                    for th in var_ths:
                        dim_cnt_gt[th] += int((v > th).sum())

                    max_logvar_s = lv.max(axis=1)
                    max_var_s = v.max(axis=1)
                    mean_logvar_s = lv.mean(axis=1)
                    p95_logvar_s = np.percentile(lv, 95, axis=1)
                    p99_logvar_s = np.percentile(lv, 99, axis=1)

                    max_var_global = max(max_var_global, float(max_var_s.max()))
                    max_logvar_global = max(max_logvar_global, float(max_logvar_s.max()))

                    cnts_per_th = {th: (v > th).sum(axis=1) for th in var_ths}

                    for th in var_ths:
                        sample_cnt_gt[th] += int((max_var_s > th).sum())

                    for i in range(bsz):
                        m = meta[i] if (meta is not None and i < len(meta)) else {}
                        row = {
                            "file_path": str(m.get("file_path", "")),
                            "file_name": str(m.get("file_name", "")),
                            "event_id": str(m.get("event_id", "")),
                            "station_name": str(m.get("station_name", "")),
                            "max_logvar": float(max_logvar_s[i]),
                            "max_var": float(max_var_s[i]),
                            "mean_logvar": float(mean_logvar_s[i]),
                            "p95_logvar": float(p95_logvar_s[i]),
                            "p99_logvar": float(p99_logvar_s[i]),
                        }
                        for th in var_ths:
                            row[f"num_var_gt_{th:g}"] = int(cnts_per_th[th][i])
                        sample_rows.append(row)

                        ev = row["event_id"] or "UNKNOWN"
                        st = row["station_name"] or "UNKNOWN"
                        for grp_stats, key in ((event_stats, ev), (station_stats, st)):
                            grp_stats[key]["n_samples"] += 1
                            grp_stats[key]["max_var_max"] = max(grp_stats[key]["max_var_max"], row["max_var"])
                            grp_stats[key]["max_logvar_max"] = max(grp_stats[key]["max_logvar_max"], row["max_logvar"])
                            for th in var_ths:
                                if row["max_var"] > th:
                                    grp_stats[key][f"cnt_max_var_gt_{th:g}"] += 1

            if n_samples == 0:
                raise RuntimeError(f"No valid samples processed for split={split}, run={run_name}")

            mean_sigma_diag = sum_var / float(n_samples)

            sample_max_vars = np.asarray([r["max_var"] for r in sample_rows], dtype=np.float64)
            sample_max_logvars = np.asarray([r["max_logvar"] for r in sample_rows], dtype=np.float64)

            sample_rows_sorted = sorted(sample_rows, key=lambda r: r["max_var"], reverse=True)
            top_rows = sample_rows_sorted[: max(args.top_k, 1)]

            sample_fields = [
                "file_path",
                "file_name",
                "event_id",
                "station_name",
                "max_logvar",
                "max_var",
                "mean_logvar",
                "p95_logvar",
                "p99_logvar",
            ] + [f"num_var_gt_{th:g}" for th in var_ths]
            _write_csv(run_out / "sample_metrics.csv", sample_rows_sorted, sample_fields)
            _write_csv(run_out / "topk_outliers.csv", top_rows, sample_fields)

            event_rows = _summarize_group_stats(event_stats, var_ths)
            station_rows = _summarize_group_stats(station_stats, var_ths)
            event_fields = ["group_key", "n_samples", "max_var_max", "max_logvar_max"] + [
                f"cnt_max_var_gt_{th:g}" for th in var_ths
            ] + [f"frac_max_var_gt_{th:g}" for th in var_ths]
            station_fields = event_fields
            _write_csv(run_out / "event_summary.csv", event_rows, event_fields)
            _write_csv(run_out / "station_summary.csv", station_rows, station_fields)

            summary = {
                "run_name": run_name,
                "checkpoint": ckpt_path.as_posix(),
                "split": split,
                "latent_dim": latent_dim,
                "n_samples": int(n_samples),
                "total_latent_values": int(total_latent_values),
                "max_var_global": float(max_var_global),
                "max_logvar_global": float(max_logvar_global),
                "sample_max_var_mean": float(np.mean(sample_max_vars)),
                "sample_max_var_median": float(np.median(sample_max_vars)),
                "sample_max_var_p95": float(np.percentile(sample_max_vars, 95)),
                "sample_max_var_p99": float(np.percentile(sample_max_vars, 99)),
                "sample_max_var_p999": float(np.percentile(sample_max_vars, 99.9)),
                "sample_max_logvar_mean": float(np.mean(sample_max_logvars)),
                "sample_max_logvar_median": float(np.median(sample_max_logvars)),
                "sample_max_logvar_p95": float(np.percentile(sample_max_logvars, 95)),
                "sample_max_logvar_p99": float(np.percentile(sample_max_logvars, 99)),
                "sample_max_logvar_p999": float(np.percentile(sample_max_logvars, 99.9)),
                "mean_sigma_diag_mean": float(np.mean(mean_sigma_diag)),
                "mean_sigma_diag_min": float(np.min(mean_sigma_diag)),
                "mean_sigma_diag_max": float(np.max(mean_sigma_diag)),
                "mean_sigma_diag_dims_gt_2": int((mean_sigma_diag > 2.0).sum()),
                "mean_sigma_diag_dims_gt_10": int((mean_sigma_diag > 10.0).sum()),
                "mean_sigma_diag_dims_gt_100": int((mean_sigma_diag > 100.0).sum()),
                "mean_sigma_diag_dims_gt_1000": int((mean_sigma_diag > 1000.0).sum()),
                "runtime_sec": float(time.time() - t0),
                "var_thresholds": var_ths,
                "logvar_thresholds": logvar_ths,
                "logvar_clip_min": float(args.logvar_clip_min),
                "logvar_clip_max": float(args.logvar_clip_max),
                "top_k": int(args.top_k),
            }
            for th in var_ths:
                summary[f"sample_cnt_max_var_gt_{th:g}"] = int(sample_cnt_gt[th])
                summary[f"sample_frac_max_var_gt_{th:g}"] = _safe_frac(sample_cnt_gt[th], n_samples)
                summary[f"dim_cnt_var_gt_{th:g}"] = int(dim_cnt_gt[th])
                summary[f"dim_frac_var_gt_{th:g}"] = _safe_frac(dim_cnt_gt[th], total_latent_values)

            with (run_out / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            audit_rows.append(summary)
            print(
                f"[INFO] split={split} run={run_name} n={n_samples} "
                f"max_var={summary['max_var_global']:.3e} mean_sigma_mean={summary['mean_sigma_diag_mean']:.6f}"
            )

    if not audit_rows:
        raise RuntimeError("No audit rows were produced.")

    summary_fields = [
        "split",
        "run_name",
        "checkpoint",
        "latent_dim",
        "n_samples",
        "max_var_global",
        "max_logvar_global",
        "sample_max_var_p99",
        "sample_max_var_p999",
        "sample_max_logvar_p99",
        "sample_max_logvar_p999",
        "mean_sigma_diag_mean",
        "mean_sigma_diag_max",
        "mean_sigma_diag_dims_gt_2",
        "mean_sigma_diag_dims_gt_10",
        "mean_sigma_diag_dims_gt_100",
        "mean_sigma_diag_dims_gt_1000",
    ]
    for th in var_ths:
        summary_fields.extend(
            [
                f"sample_cnt_max_var_gt_{th:g}",
                f"sample_frac_max_var_gt_{th:g}",
                f"dim_cnt_var_gt_{th:g}",
                f"dim_frac_var_gt_{th:g}",
            ]
        )
    summary_fields.append("runtime_sec")

    summary_csv = out_dir / "audit_summary.csv"
    _write_csv(summary_csv, audit_rows, summary_fields)

    summary_md = out_dir / "audit_summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# Latent Variance Outlier Audit Summary\n\n")
        f.write("| Split | Run | max_var | p99(max_var) | mean_sigma_diag_mean | dims>10 | dims>100 | dims>1000 | samples max_var>1e3 |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(audit_rows, key=lambda x: (x["split"], x["run_name"])):
            f.write(
                f"| {r['split']} | {r['run_name']} | {r['max_var_global']:.3e} | {r['sample_max_var_p99']:.3e} | "
                f"{r['mean_sigma_diag_mean']:.4f} | {r['mean_sigma_diag_dims_gt_10']} | "
                f"{r['mean_sigma_diag_dims_gt_100']} | {r['mean_sigma_diag_dims_gt_1000']} | "
                f"{r.get('sample_cnt_max_var_gt_1000', 0)} |\n"
            )

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": out_dir.as_posix(),
        "summary_csv": summary_csv.as_posix(),
        "summary_md": summary_md.as_posix(),
        "n_rows": len(audit_rows),
    }
    with (out_dir / "audit_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] audit_output_dir:", out_dir.as_posix())
    print("[INFO] audit_summary_csv:", summary_csv.as_posix())
    print("[INFO] audit_summary_md :", summary_md.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
