#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List


def _read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_by_run(path: Path) -> Dict[str, Dict[str, str]]:
    rows = _read_rows(path)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        out[row["run_name"]] = row
    return out


def _f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _pctl(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    idx = (len(xs) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(xs) - 1)
    w = idx - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


def _agg(values: List[float]) -> Dict[str, float]:
    return {
        "n": float(len(values)),
        "mean": statistics.fmean(values) if values else float("nan"),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values) if values else float("nan"),
        "p90": _pctl(values, 0.9),
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--latent_test_csv", required=True)
    p.add_argument("--latent_ood_csv", required=True)
    p.add_argument("--prior_test_csv", required=True)
    p.add_argument("--prior_ood_csv", required=True)
    p.add_argument("--audit_csv", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    latent_test = _read_by_run(Path(args.latent_test_csv))
    latent_ood = _read_by_run(Path(args.latent_ood_csv))
    prior_test = _read_by_run(Path(args.prior_test_csv))
    prior_ood = _read_by_run(Path(args.prior_ood_csv))

    audit_rows = _read_rows(Path(args.audit_csv))
    audit_by_split_run: Dict[str, Dict[str, Dict[str, str]]] = {"test": {}, "ood_event": {}}
    for row in audit_rows:
        split = row.get("split", "")
        if split in audit_by_split_run:
            audit_by_split_run[split][row["run_name"]] = row

    run_names = sorted(set(latent_test) & set(latent_ood) & set(prior_test) & set(prior_ood))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_run_rows: List[Dict[str, object]] = []
    for rn in run_names:
        t = latent_test[rn]
        o = latent_ood[rn]
        pt = prior_test[rn]
        po = prior_ood[rn]
        at = audit_by_split_run["test"].get(rn, {})
        ao = audit_by_split_run["ood_event"].get(rn, {})

        per_run_rows.append(
            {
                "run_name": rn,
                "test_diag_mae": _f(t, "diag_mae"),
                "test_offdiag_mean_abs_corr": _f(t, "offdiag_mean_abs_corr"),
                "test_kl_moment_to_std_normal": _f(t, "kl_moment_to_std_normal"),
                "test_w2_moment_to_std_normal": _f(t, "w2_moment_to_std_normal"),
                "ood_diag_mae": _f(o, "diag_mae"),
                "ood_offdiag_mean_abs_corr": _f(o, "offdiag_mean_abs_corr"),
                "ood_kl_moment_to_std_normal": _f(o, "kl_moment_to_std_normal"),
                "ood_w2_moment_to_std_normal": _f(o, "w2_moment_to_std_normal"),
                "prior_test_realism_composite": _f(pt, "realism_composite"),
                "prior_ood_realism_composite": _f(po, "realism_composite"),
                "test_max_var_global": float(at.get("max_var_global", "nan")),
                "ood_max_var_global": float(ao.get("max_var_global", "nan")),
            }
        )

    per_run_fields = list(per_run_rows[0].keys()) if per_run_rows else []
    per_run_csv = output_dir / "robustness_per_run.csv"
    _write_csv(per_run_csv, per_run_rows, per_run_fields)

    metric_keys = [
        "test_diag_mae",
        "test_offdiag_mean_abs_corr",
        "test_kl_moment_to_std_normal",
        "test_w2_moment_to_std_normal",
        "ood_diag_mae",
        "ood_offdiag_mean_abs_corr",
        "ood_kl_moment_to_std_normal",
        "ood_w2_moment_to_std_normal",
        "prior_test_realism_composite",
        "prior_ood_realism_composite",
        "test_max_var_global",
        "ood_max_var_global",
    ]

    agg_rows: List[Dict[str, object]] = []
    for k in metric_keys:
        vals = [float(r[k]) for r in per_run_rows]
        s = _agg(vals)
        agg_rows.append(
            {
                "metric": k,
                "n_runs": int(s["n"]),
                "mean": s["mean"],
                "std": s["std"],
                "median": s["median"],
                "p90": s["p90"],
                "min": s["min"],
                "max": s["max"],
            }
        )

    agg_csv = output_dir / "robustness_aggregate.csv"
    _write_csv(
        agg_csv,
        agg_rows,
        ["metric", "n_runs", "mean", "std", "median", "p90", "min", "max"],
    )

    md_path = output_dir / "robustness_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Stage-2 beta=0.1 logvarfix 10-seed Robustness Summary (v2)\n\n")
        f.write(f"- n_runs: `{len(per_run_rows)}`\n\n")
        f.write("## Per-run\n\n")
        f.write("| Run | test diag_mae | test offdiag | test KL | ood diag_mae | ood offdiag | ood KL | prior test comp | prior ood comp | test max_var | ood max_var |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in per_run_rows:
            f.write(
                f"| {r['run_name']} | {float(r['test_diag_mae']):.4f} | {float(r['test_offdiag_mean_abs_corr']):.4f} | "
                f"{float(r['test_kl_moment_to_std_normal']):.4f} | {float(r['ood_diag_mae']):.4f} | "
                f"{float(r['ood_offdiag_mean_abs_corr']):.4f} | {float(r['ood_kl_moment_to_std_normal']):.4f} | "
                f"{float(r['prior_test_realism_composite']):.4f} | {float(r['prior_ood_realism_composite']):.4f} | "
                f"{float(r['test_max_var_global']):.3e} | {float(r['ood_max_var_global']):.3e} |\n"
            )

        f.write("\n## Aggregate (across seeds)\n\n")
        f.write("| Metric | Mean | Std | Median | P90 | Min | Max |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in agg_rows:
            f.write(
                f"| {r['metric']} | {float(r['mean']):.6f} | {float(r['std']):.6f} | "
                f"{float(r['median']):.6f} | {float(r['p90']):.6f} | "
                f"{float(r['min']):.6f} | {float(r['max']):.6f} |\n"
            )

    meta = {
        "n_runs": len(per_run_rows),
        "run_names": [r["run_name"] for r in per_run_rows],
        "inputs": {
            "latent_test_csv": args.latent_test_csv,
            "latent_ood_csv": args.latent_ood_csv,
            "prior_test_csv": args.prior_test_csv,
            "prior_ood_csv": args.prior_ood_csv,
            "audit_csv": args.audit_csv,
        },
        "outputs": {
            "per_run_csv": per_run_csv.as_posix(),
            "aggregate_csv": agg_csv.as_posix(),
            "summary_md": md_path.as_posix(),
        },
    }
    with (output_dir / "robustness_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] per_run_csv :", per_run_csv.as_posix())
    print("[INFO] agg_csv     :", agg_csv.as_posix())
    print("[INFO] summary_md  :", md_path.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
