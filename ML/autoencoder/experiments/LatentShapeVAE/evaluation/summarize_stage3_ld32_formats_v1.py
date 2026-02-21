#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Dict, List


RUN_RE = re.compile(r"^lsv_stage3_vae_base_ld32_(?P<fmt>.+)_s(?P<seed>\d+)$")


def _read_by_run(path: Path) -> Dict[str, Dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return {r["run_name"]: r for r in rows}


def _f(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return fmean(values), (pstdev(values) if len(values) > 1 else 0.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--latent_test_csv", required=True)
    p.add_argument("--latent_ood_csv", required=True)
    p.add_argument("--prior_test_csv", required=True)
    p.add_argument("--prior_ood_csv", required=True)
    p.add_argument("--audit_csv", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    lt = _read_by_run(Path(args.latent_test_csv))
    lo = _read_by_run(Path(args.latent_ood_csv))
    pt = _read_by_run(Path(args.prior_test_csv))
    po = _read_by_run(Path(args.prior_ood_csv))

    audit_rows = list(csv.DictReader(Path(args.audit_csv).open("r", encoding="utf-8")))
    audit = defaultdict(dict)  # split -> run -> row
    for r in audit_rows:
        audit[r["split"]][r["run_name"]] = r

    run_names = sorted(set(lt) & set(lo) & set(pt) & set(po))

    per_run = []
    for rn in run_names:
        m = RUN_RE.match(rn)
        fmt = m.group("fmt") if m else "unknown"
        seed = int(m.group("seed")) if m else -1
        at = audit["test"].get(rn, {})
        ao = audit["ood_event"].get(rn, {})
        per_run.append(
            {
                "run_name": rn,
                "format": fmt,
                "seed": seed,
                "test_diag_mae": _f(lt[rn], "diag_mae"),
                "test_offdiag": _f(lt[rn], "offdiag_mean_abs_corr"),
                "test_kl": _f(lt[rn], "kl_moment_to_std_normal"),
                "test_w2": _f(lt[rn], "w2_moment_to_std_normal"),
                "ood_diag_mae": _f(lo[rn], "diag_mae"),
                "ood_offdiag": _f(lo[rn], "offdiag_mean_abs_corr"),
                "ood_kl": _f(lo[rn], "kl_moment_to_std_normal"),
                "ood_w2": _f(lo[rn], "w2_moment_to_std_normal"),
                "prior_test_comp": _f(pt[rn], "realism_composite"),
                "prior_ood_comp": _f(po[rn], "realism_composite"),
                "test_max_var": float(at.get("max_var_global", "nan")),
                "ood_max_var": float(ao.get("max_var_global", "nan")),
            }
        )

    by_fmt = defaultdict(list)
    for r in per_run:
        by_fmt[r["format"]].append(r)

    fmt_rows = []
    for fmt, rows in sorted(by_fmt.items()):
        out = {"format": fmt, "n_runs": len(rows)}
        for key in [
            "test_diag_mae",
            "test_offdiag",
            "test_kl",
            "test_w2",
            "ood_diag_mae",
            "ood_offdiag",
            "ood_kl",
            "ood_w2",
            "prior_test_comp",
            "prior_ood_comp",
            "test_max_var",
            "ood_max_var",
        ]:
            vals = [float(r[key]) for r in rows]
            m, s = _mean_std(vals)
            out[f"{key}_mean"] = m
            out[f"{key}_std"] = s
        fmt_rows.append(out)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run_csv = out_dir / "stage3_ld32_formats_per_run.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        w.writeheader()
        w.writerows(per_run)

    fmt_csv = out_dir / "stage3_ld32_formats_summary.csv"
    with fmt_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fmt_rows[0].keys()))
        w.writeheader()
        w.writerows(fmt_rows)

    fmt_md = out_dir / "stage3_ld32_formats_summary.md"
    with fmt_md.open("w", encoding="utf-8") as f:
        f.write("# Stage3 LD32 Format Comparison (v1)\n\n")
        f.write("| Format | n | test diag | test offdiag | test KL | ood diag | ood offdiag | ood KL | prior test | prior ood | test max_var |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in sorted(fmt_rows, key=lambda x: x["test_kl_mean"]):
            f.write(
                f"| {r['format']} | {r['n_runs']} | {r['test_diag_mae_mean']:.4f} ± {r['test_diag_mae_std']:.4f} | "
                f"{r['test_offdiag_mean']:.4f} ± {r['test_offdiag_std']:.4f} | "
                f"{r['test_kl_mean']:.4f} ± {r['test_kl_std']:.4f} | "
                f"{r['ood_diag_mae_mean']:.4f} ± {r['ood_diag_mae_std']:.4f} | "
                f"{r['ood_offdiag_mean']:.4f} ± {r['ood_offdiag_std']:.4f} | "
                f"{r['ood_kl_mean']:.4f} ± {r['ood_kl_std']:.4f} | "
                f"{r['prior_test_comp_mean']:.4f} ± {r['prior_test_comp_std']:.4f} | "
                f"{r['prior_ood_comp_mean']:.4f} ± {r['prior_ood_comp_std']:.4f} | "
                f"{r['test_max_var_mean']:.3e} |\n"
            )

    print("[INFO] per_run_csv:", per_run_csv.as_posix())
    print("[INFO] summary_csv:", fmt_csv.as_posix())
    print("[INFO] summary_md :", fmt_md.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
