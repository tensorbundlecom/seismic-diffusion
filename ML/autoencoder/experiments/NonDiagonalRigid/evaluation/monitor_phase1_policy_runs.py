#!/usr/bin/env python3
"""
Monitor and summarize NonDiagonalRigid Phase-1 policy runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def last_history_step(history_csv: Path) -> Dict:
    if not history_csv.exists():
        return {"last_step": None, "last_val_loss": None}
    last = None
    with history_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            last = row
    if last is None:
        return {"last_step": None, "last_val_loss": None}
    return {
        "last_step": int(float(last.get("step", 0))),
        "last_val_loss": float(last.get("val_loss", 0.0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor phase-1 policy run progress.")
    parser.add_argument(
        "--results-root",
        default="ML/autoencoder/experiments/NonDiagonalRigid/results",
    )
    parser.add_argument(
        "--run-prefix",
        default="rigid_policy_pilot_",
    )
    parser.add_argument(
        "--output-dir",
        default="ML/autoencoder/experiments/NonDiagonalRigid/results/phase1_policy_monitor",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for run_dir in sorted(root.glob(f"{args.run_prefix}*")):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        cfg_path = run_dir / "run_config.json"
        summary_path = run_dir / "final_summary.json"
        hist_path = run_dir / "history.csv"

        if not cfg_path.exists():
            continue

        cfg = read_json(cfg_path)
        hist = last_history_step(hist_path)

        if summary_path.exists():
            final = read_json(summary_path)
            status = "finished"
            best_val = float(final.get("best_val", 0.0))
            best_step = int(final.get("best_step", -1))
            stop_reason = str(final.get("stop_reason", "unknown"))
        else:
            status = "running_or_interrupted"
            best_val = float(hist["last_val_loss"]) if hist["last_val_loss"] is not None else None
            best_step = int(hist["last_step"]) if hist["last_step"] is not None else None
            stop_reason = ""

        row = {
            "run_name": run_name,
            "status": status,
            "model_family": cfg.get("model_family"),
            "policy": cfg.get("policy"),
            "scale": cfg.get("scale"),
            "seed": cfg.get("seed"),
            "latent_dim": cfg.get("latent_dim"),
            "max_steps": cfg.get("max_steps"),
            "best_or_last_step": best_step,
            "best_or_last_val_loss": best_val,
            "stop_reason": stop_reason,
        }
        rows.append(row)

    rows.sort(
        key=lambda x: (
            str(x["model_family"]),
            str(x["policy"]),
            float(x["scale"]) if x["scale"] is not None else 999.0,
            int(x["seed"]) if x["seed"] is not None else 9999,
        )
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"phase1_policy_monitor_{ts}.csv"
    md_path = out_dir / f"phase1_policy_monitor_{ts}.md"
    latest_json = out_dir / "latest_phase1_policy_monitor.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "status",
                "model_family",
                "policy",
                "scale",
                "seed",
                "latent_dim",
                "max_steps",
                "best_or_last_step",
                "best_or_last_val_loss",
                "stop_reason",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_lines = [
        "# Phase-1 Policy Monitor",
        "",
        f"- generated_utc: `{datetime.now(timezone.utc).isoformat()}`",
        f"- run_prefix: `{args.run_prefix}`",
        f"- total_runs_seen: `{len(rows)}`",
        "",
        "| Run | Status | Family | Policy | Scale | Seed | Step | Val Loss | Stop |",
        "|:---|:---|:---|:---|---:|---:|---:|---:|:---|",
    ]
    for r in rows:
        step = "--" if r["best_or_last_step"] is None else str(r["best_or_last_step"])
        vloss = "--" if r["best_or_last_val_loss"] is None else f"{float(r['best_or_last_val_loss']):.4f}"
        md_lines.append(
            f"| {r['run_name']} | {r['status']} | {r['model_family']} | {r['policy']} | "
            f"{r['scale']} | {r['seed']} | {step} | {vloss} | {r['stop_reason']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latest_json.write_text(json.dumps({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "rows": rows}, indent=2))

    print(f"[OK] Monitor CSV: {csv_path}")
    print(f"[OK] Monitor MD: {md_path}")
    print(f"[OK] Latest JSON: {latest_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

