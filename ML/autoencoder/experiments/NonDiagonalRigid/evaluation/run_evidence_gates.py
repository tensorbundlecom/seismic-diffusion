#!/usr/bin/env python3
"""
NonDiagonalRigid - Evidence Gate Runner (Skeleton)

This script only prepares a deterministic output layout for Q61-Q64.
It does NOT compute metrics yet. Final logic will be added after the
discussion questions are fully frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def write_csv(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Q61-Q64 evidence report skeleton.")
    parser.add_argument(
        "--config",
        default="ML/autoencoder/experiments/NonDiagonalRigid/configs/offdiag_minimum_evidence_v1.yaml",
        help="Path to evidence config file.",
    )
    parser.add_argument(
        "--results-root",
        default="ML/autoencoder/experiments/NonDiagonalRigid/results",
        help="Root folder for result runs.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id (example: confirmatory_20260217_1530).",
    )
    args = parser.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        raise FileNotFoundError(f"Config not found: {cfg}")

    evidence_dir = Path(args.results_root) / args.run_id / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        evidence_dir / "signal_reality_null_test.csv",
        [
            "model_family",
            "backbone",
            "latent_dim",
            "seed",
            "metric",
            "real_value",
            "null_mean",
            "null_std",
            "z_null",
            "pass",
        ],
    )
    write_csv(
        evidence_dir / "predictive_utility_stepwise.csv",
        [
            "model_family",
            "seed",
            "from_latent",
            "to_latent",
            "odi_k",
            "quality_k",
            "quality_k1",
            "drop_k_to_k1",
            "spearman_rho_family",
            "ci_low",
            "ci_high",
            "pass",
        ],
    )
    write_csv(
        evidence_dir / "policy_benefit_comparison.csv",
        [
            "model_family",
            "distribution",
            "policy_name",
            "selected_latent",
            "quality_score",
            "odi_score",
            "param_count",
            "runtime_sec",
            "pass",
        ],
    )
    write_csv(
        evidence_dir / "cross_setting_consistency.csv",
        [
            "model_family",
            "distribution",
            "seed",
            "selected_latent",
            "quality_score",
            "odi_score",
            "param_count",
            "neighbor_step_ok",
            "pass",
        ],
    )

    summary = {
        "version": "v1-skeleton",
        "status": "PENDING_IMPLEMENTATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg),
        "run_id": args.run_id,
        "gates": {
            "Q61_signal_reality": "PENDING",
            "Q62_predictive_utility": "PENDING",
            "Q63_policy_benefit": "PENDING",
            "Q64_cross_setting_consistency": "PENDING",
        },
        "note": "Skeleton only. No metric computation yet.",
    }
    with (evidence_dir / "evidence_gate_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = [
        "# Evidence Gate Summary (Skeleton)",
        "",
        f"- Run ID: `{args.run_id}`",
        f"- Config: `{cfg}`",
        "- Status: `PENDING_IMPLEMENTATION`",
        "",
        "## Gate Status",
        "- Q61 Signal Reality: PENDING",
        "- Q62 Predictive Utility: PENDING",
        "- Q63 Policy Benefit: PENDING",
        "- Q64 Cross-Setting Consistency: PENDING",
        "",
        "This is a template output only. Final gate logic will be added after protocol freeze.",
    ]
    (evidence_dir / "evidence_gate_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] Evidence skeleton prepared: {evidence_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
