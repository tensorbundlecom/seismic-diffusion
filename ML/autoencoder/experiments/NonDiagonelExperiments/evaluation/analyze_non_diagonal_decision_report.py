import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


METRICS = [
    "ssim",
    "lsd",
    "sc",
    "s_corr",
    "sta_lta_err",
    "mr_lsd",
    "arias_err",
    "env_corr",
    "dtw",
    "xcorr",
]

HIGHER_BETTER = {"ssim", "s_corr", "env_corr", "xcorr"}
LOWER_BETTER = {"lsd", "sc", "sta_lta_err", "mr_lsd", "arias_err", "dtw"}

PRIMARY_METRICS = ["mr_lsd", "dtw", "lsd", "ssim"]
GUARDRAIL_METRICS = ["s_corr", "sc", "arias_err", "env_corr", "xcorr"]

OFFDIAG_KEYS = [
    "mean_abs_corr_offdiag",
    "p95_abs_corr_offdiag",
    "max_abs_corr_offdiag",
    "offdiag_energy_ratio",
]


def load_rows(jsonl_path: Path) -> List[Dict]:
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def signed_improvement(fullcov_value: float, baseline_value: float, metric: str) -> float:
    # Positive value means FullCov is better on that metric.
    if metric in HIGHER_BETTER:
        return float(fullcov_value - baseline_value)
    return float(baseline_value - fullcov_value)


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    if n == 0:
        return {"mean": None, "ci95_low": None, "ci95_high": None, "p_gt0": None}
    if n == 1:
        v = float(values[0])
        return {"mean": v, "ci95_low": v, "ci95_high": v, "p_gt0": 1.0 if v > 0 else 0.0}

    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx]
    boot_means = samples.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(boot_means, 0.025)),
        "ci95_high": float(np.quantile(boot_means, 0.975)),
        "p_gt0": float(np.mean(boot_means > 0.0)),
    }


def rankdata_simple(x: np.ndarray) -> np.ndarray:
    # Tie handling is simple and deterministic; sufficient for continuous metrics.
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return corr_pearson(rankdata_simple(x), rankdata_simple(y))


def build_metric_vectors(rows: List[Dict]) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    for metric in METRICS:
        b = np.array([r["metrics"]["BaselineGeo"][metric] for r in rows], dtype=np.float64)
        f = np.array([r["metrics"]["FullCovGeo"][metric] for r in rows], dtype=np.float64)
        d = np.array([signed_improvement(fv, bv, metric) for fv, bv in zip(f, b)], dtype=np.float64)
        out[metric] = {"baseline": b, "fullcov": f, "signed_fullcov_minus_baseline": d}
    return out


def group_breakdown(rows: List[Dict], key: str, scale_by_metric: Dict[str, float]) -> List[Dict]:
    groups = {}
    for r in rows:
        g = r.get(key, "UNKNOWN")
        groups.setdefault(g, []).append(r)

    out = []
    for g, items in groups.items():
        metric_vecs = build_metric_vectors(items)
        signed_primary_raw = []
        signed_primary_scaled = []
        robust_fullcov_wins = 0
        robust_baseline_wins = 0
        for m in PRIMARY_METRICS:
            mean_signed = float(np.mean(metric_vecs[m]["signed_fullcov_minus_baseline"]))
            signed_primary_raw.append(mean_signed)
            s = float(scale_by_metric.get(m, 1.0))
            if abs(s) < 1e-12:
                s = 1.0
            signed_primary_scaled.append(mean_signed / s)
            # Robust local win by sign of mean only (sample counts are small at group level).
            if mean_signed > 0:
                robust_fullcov_wins += 1
            elif mean_signed < 0:
                robust_baseline_wins += 1

        row = {
            key: g,
            "count": len(items),
            "primary_signed_mean_raw": float(np.mean(signed_primary_raw)),
            "primary_signed_mean_z": float(np.mean(signed_primary_scaled)),
            "primary_fullcov_wins": robust_fullcov_wins,
            "primary_baseline_wins": robust_baseline_wins,
        }
        for m in PRIMARY_METRICS + GUARDRAIL_METRICS:
            row[f"{m}_signed"] = float(np.mean(metric_vecs[m]["signed_fullcov_minus_baseline"]))
        out.append(row)

    out.sort(key=lambda x: (-x["count"], -x["primary_signed_mean_z"]))
    return out


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def choose_recommendation(bootstrap_summary: Dict[str, Dict]) -> Dict[str, str]:
    robust_fullcov = []
    robust_baseline = []
    uncertain = []
    for m in METRICS:
        s = bootstrap_summary[m]
        lo = s["ci95_low"]
        hi = s["ci95_high"]
        if lo is not None and lo > 0:
            robust_fullcov.append(m)
        elif hi is not None and hi < 0:
            robust_baseline.append(m)
        else:
            uncertain.append(m)

    decision = "inconclusive_keep_both"
    if len(robust_fullcov) >= len(robust_baseline) + 2:
        decision = "favor_fullcov"
    elif len(robust_baseline) >= len(robust_fullcov) + 2:
        decision = "favor_baseline"

    return {
        "decision": decision,
        "robust_fullcov_metrics": robust_fullcov,
        "robust_baseline_metrics": robust_baseline,
        "uncertain_metrics": uncertain,
    }


def format_signed(v: float, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if v > 0:
        return f"+{v:.{digits}f}"
    return f"{v:.{digits}f}"


def write_markdown_report(
    path: Path,
    meta: Dict,
    bootstrap_rows: List[Dict],
    decision: Dict,
    station_rows: List[Dict],
    event_rows: List[Dict],
    offdiag_corr_rows: List[Dict],
):
    lines = []
    lines.append("# NonDiagonel Decision Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Compare `BaselineGeo` vs `FullCovGeo` on paired OOD samples.")
    lines.append("- Signed improvement convention: positive means **FullCov better**.")
    lines.append(f"- Samples: `{meta['num_samples']}`")
    lines.append("")
    lines.append("## Bootstrap (Paired) Summary")
    lines.append("")
    lines.append("| Metric | Baseline Mean | FullCov Mean | Signed Diff (FullCov better +) | 95% CI | p(diff>0) |")
    lines.append("|:---|---:|---:|---:|:---:|---:|")
    for r in bootstrap_rows:
        ci = f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
        lines.append(
            f"| {r['metric']} | {r['baseline_mean']:.4f} | {r['fullcov_mean']:.4f} | "
            f"{format_signed(r['signed_diff_mean'])} | {ci} | {r['p_gt0']:.3f} |"
        )

    lines.append("")
    lines.append("## Robust Metric Wins")
    lines.append("")
    lines.append(f"- Decision: `{decision['decision']}`")
    lines.append(f"- Robust FullCov wins: `{decision['robust_fullcov_metrics']}`")
    lines.append(f"- Robust Baseline wins: `{decision['robust_baseline_metrics']}`")
    lines.append(f"- Uncertain: `{decision['uncertain_metrics']}`")

    lines.append("")
    lines.append("## Station Breakdown (Primary Composite)")
    lines.append("")
    lines.append("| Station | Count | Primary Z-Composite | FullCov Wins | Baseline Wins |")
    lines.append("|:---|---:|---:|---:|---:|")
    for r in station_rows:
        lines.append(
            f"| {r['station_name']} | {r['count']} | {format_signed(r['primary_signed_mean_z'])} | "
            f"{r['primary_fullcov_wins']} | {r['primary_baseline_wins']} |"
        )

    lines.append("")
    lines.append("## Event Breakdown (Primary Composite)")
    lines.append("")
    lines.append("| Event | Count | Primary Z-Composite | FullCov Wins | Baseline Wins |")
    lines.append("|:---|---:|---:|---:|---:|")
    for r in event_rows:
        lines.append(
            f"| {r['event_id']} | {r['count']} | {format_signed(r['primary_signed_mean_z'])} | "
            f"{r['primary_fullcov_wins']} | {r['primary_baseline_wins']} |"
        )

    lines.append("")
    lines.append("## Off-Diagonal vs Quality (Correlation)")
    lines.append("")
    lines.append("| OffDiag Feature | Metric | Pearson | Spearman |")
    lines.append("|:---|:---|---:|---:|")
    for r in offdiag_corr_rows:
        lines.append(
            f"| {r['offdiag_feature']} | {r['metric']} | {r['pearson']:.4f} | {r['spearman']:.4f} |"
        )

    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- Use primary metrics (`mr_lsd`, `dtw`, `lsd`, `ssim`) for direction.")
    lines.append("- Keep guardrails (`s_corr`, `sc`, `arias_err`, `env_corr`, `xcorr`) from regressing.")
    lines.append("- Prefer architecture change only if primary gains are robust and guardrail losses are limited.")
    lines.append("- `Primary Z-Composite` averages primary signed improvements after per-metric std normalization.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Produce decision report for NonDiagonel evaluation outputs.")
    parser.add_argument(
        "--eval_dir",
        default="ML/autoencoder/experiments/NonDiagonel/results/evaluations/post_training_custom_geo_repi_s42_20260216_1942",
    )
    parser.add_argument("--bootstrap_iters", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    per_sample_path = eval_dir / "metrics_per_sample.jsonl"
    manifest_path = eval_dir / "manifest.json"

    rows = load_rows(per_sample_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    metric_vecs = build_metric_vectors(rows)

    bootstrap_summary = {}
    bootstrap_rows = []
    for mi, m in enumerate(METRICS):
        b = metric_vecs[m]["baseline"]
        f = metric_vecs[m]["fullcov"]
        d = metric_vecs[m]["signed_fullcov_minus_baseline"]
        s = bootstrap_ci(d, n_boot=args.bootstrap_iters, seed=args.seed + (mi + 1) * 97)
        bootstrap_summary[m] = s
        bootstrap_rows.append(
            {
                "metric": m,
                "baseline_mean": float(np.mean(b)),
                "fullcov_mean": float(np.mean(f)),
                "signed_diff_mean": s["mean"],
                "ci95_low": s["ci95_low"],
                "ci95_high": s["ci95_high"],
                "p_gt0": s["p_gt0"],
            }
        )

    decision = choose_recommendation(bootstrap_summary)

    scale_by_metric = {}
    for m in PRIMARY_METRICS:
        d = metric_vecs[m]["signed_fullcov_minus_baseline"]
        std = float(np.std(d))
        scale_by_metric[m] = std if std > 1e-12 else 1.0

    station_rows = group_breakdown(rows, key="station_name", scale_by_metric=scale_by_metric)
    event_rows = group_breakdown(rows, key="event_id", scale_by_metric=scale_by_metric)

    # Off-diagonal feature correlation with signed improvements (all metrics).
    offdiag_corr_rows = []
    for offdiag_key in OFFDIAG_KEYS:
        x = np.array([r["fullcov_posterior"][offdiag_key] for r in rows], dtype=np.float64)
        for m in METRICS:
            y = metric_vecs[m]["signed_fullcov_minus_baseline"]
            offdiag_corr_rows.append(
                {
                    "offdiag_feature": offdiag_key,
                    "metric": m,
                    "pearson": corr_pearson(x, y),
                    "spearman": corr_spearman(x, y),
                }
            )

    # Keep top absolute relationships for markdown readability.
    offdiag_top = sorted(
        offdiag_corr_rows,
        key=lambda r: np.nan_to_num(max(abs(r["pearson"]), abs(r["spearman"])), nan=-1.0),
        reverse=True,
    )[:12]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eval_dir": str(eval_dir),
        "num_samples": len(rows),
        "bootstrap_iters": args.bootstrap_iters,
        "seed": args.seed,
        "bootstrap_summary": bootstrap_summary,
        "decision": decision,
        "station_breakdown": station_rows,
        "event_breakdown": event_rows,
        "offdiag_metric_correlations": offdiag_corr_rows,
        "manifest": manifest,
    }

    report_dir = eval_dir / "decision"
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "decision_report.json", "w") as f:
        json.dump(report, f, indent=2)

    write_csv(report_dir / "bootstrap_metric_summary.csv", bootstrap_rows)
    write_csv(report_dir / "station_breakdown.csv", station_rows)
    write_csv(report_dir / "event_breakdown.csv", event_rows)
    write_csv(report_dir / "offdiag_metric_correlations.csv", offdiag_corr_rows)

    write_markdown_report(
        path=report_dir / "decision_report.md",
        meta={"num_samples": len(rows)},
        bootstrap_rows=bootstrap_rows,
        decision=decision,
        station_rows=station_rows,
        event_rows=event_rows,
        offdiag_corr_rows=offdiag_top,
    )

    print(f"[DONE] Decision report generated: {report_dir}")
    print(f"[INFO] Decision: {decision['decision']}")
    print(f"[INFO] Robust FullCov wins: {decision['robust_fullcov_metrics']}")
    print(f"[INFO] Robust Baseline wins: {decision['robust_baseline_metrics']}")


if __name__ == "__main__":
    main()
