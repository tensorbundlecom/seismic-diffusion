#!/usr/bin/env python3
"""
Compute a single latent-shape similarity score to N(0, I) from latent_shape_summary.csv files.

Primary absolute metrics already produced by analyze_latent_shape.py:
  - kl_moment_to_std_normal
  - w2_moment_to_std_normal
  - diag_mae
  - eig_ratio

We provide two gap scores:
  1) n01_abs_gap (absolute units, lower=better)
  2) n01_robust_gap (robust-z normalized across runs in the same split, lower=better)
and a bounded similarity:
  - n01_similarity = 1 / (1 + exp(n01_robust_gap))  in (0,1)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class Entry:
    run_name: str
    split: str
    source_csv: str
    mean_norm_l2: float
    diag_mae: float
    offdiag_mean_abs_corr: float
    eig_ratio: float
    kl_moment: float
    w2_moment: float
    eig_log_abs: float
    n01_abs_gap: float
    # Filled later
    z_kl: float = 0.0
    z_w2: float = 0.0
    z_diag: float = 0.0
    z_eig_log_abs: float = 0.0
    n01_robust_gap: float = 0.0
    n01_similarity: float = 0.0
    shape_class: str = ""


def _float(row: Dict[str, str], key: str) -> float:
    return float(row[key])


def _safe_abs_log(x: float, eps: float = 1e-12) -> float:
    return abs(math.log(max(x, eps)))


def _safe_inverse_logit(x: float) -> float:
    # Return 1 / (1 + exp(x)) with overflow-safe branches.
    if x >= 0:
        z = math.exp(-x) if x < 700 else 0.0
        return z / (1.0 + z)
    z = math.exp(x)
    return 1.0 / (1.0 + z)


def _shape_class(diag: float, offdiag: float, eig_ratio: float, kl: float) -> str:
    # Lower thresholds are stricter.
    if kl <= 0.5 and diag <= 0.05 and offdiag <= 0.02 and eig_ratio <= 3.0:
        return "very_close"
    if kl <= 2.0 and diag <= 0.10 and offdiag <= 0.05 and eig_ratio <= 6.0:
        return "close"
    if kl <= 10.0 and diag <= 0.50 and offdiag <= 0.20 and eig_ratio <= 30.0:
        return "moderate"
    return "distorted"


def _read_entries(csv_path: Path) -> List[Entry]:
    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    out: List[Entry] = []
    for r in rows:
        split = r.get("split", "")
        if not split:
            # Fallback from directory name.
            parent = csv_path.parent.name
            split = "ood_event" if "ood" in parent else "test"

        diag = _float(r, "diag_mae")
        offdiag = _float(r, "offdiag_mean_abs_corr")
        eig = _float(r, "eig_ratio")
        kl = _float(r, "kl_moment_to_std_normal")
        w2 = _float(r, "w2_moment_to_std_normal")
        eig_log_abs = _safe_abs_log(eig)

        # Absolute gap in native units (lower=better).
        n01_abs_gap = kl + w2 + 0.5 * diag + 0.25 * eig_log_abs

        out.append(
            Entry(
                run_name=r["run_name"],
                split=split,
                source_csv=csv_path.as_posix(),
                mean_norm_l2=_float(r, "mean_norm_l2"),
                diag_mae=diag,
                offdiag_mean_abs_corr=offdiag,
                eig_ratio=eig,
                kl_moment=kl,
                w2_moment=w2,
                eig_log_abs=eig_log_abs,
                n01_abs_gap=n01_abs_gap,
            )
        )
    return out


def _robust_z(values: List[float], eps: float = 1e-12) -> List[float]:
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    if mad > eps:
        denom = 1.4826 * mad
        return [(v - med) / denom for v in values]

    # Fallback to standard z-score if MAD is degenerate.
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 1.0
    std = std if std > eps else 1.0
    return [(v - mean) / std for v in values]


def _dedup_best(entries: Iterable[Entry]) -> List[Entry]:
    best: Dict[Tuple[str, str], Entry] = {}
    for e in entries:
        key = (e.run_name, e.split)
        cur = best.get(key)
        if cur is None or e.n01_abs_gap < cur.n01_abs_gap:
            best[key] = e
    return list(best.values())


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_markdown(path: Path, split: str, entries: List[Entry]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# N(0,1) Similarity Ranking - {split}\n\n")
        f.write("| Rank | Run | Class | KL | W2 | diag_mae | offdiag | eig_ratio | n01_abs_gap | n01_robust_gap | n01_similarity |\n")
        f.write("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for i, e in enumerate(sorted(entries, key=lambda x: x.n01_robust_gap), 1):
            f.write(
                f"| {i} | {e.run_name} | {e.shape_class} | {e.kl_moment:.4f} | {e.w2_moment:.4f} | "
                f"{e.diag_mae:.4f} | {e.offdiag_mean_abs_corr:.4f} | {e.eig_ratio:.4f} | "
                f"{e.n01_abs_gap:.4f} | {e.n01_robust_gap:.4f} | {e.n01_similarity:.4f} |\n"
            )

        f.write("\nScoring:\n")
        f.write("- `n01_abs_gap = KL + W2 + 0.5*diag_mae + 0.25*|log(eig_ratio)|`\n")
        f.write("- `n01_robust_gap = rz(KL) + rz(W2) + 0.5*rz(diag_mae) + 0.25*rz(|log(eig_ratio)|)`\n")
        f.write("- `n01_similarity = 1/(1+exp(n01_robust_gap))`\n")
        f.write("- Lower `gap` is better, higher `similarity` is better.\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_glob",
        default="ML/autoencoder/experiments/LatentShapeVAE/results/*/latent_shape_summary.csv",
    )
    p.add_argument(
        "--output_dir",
        default="ML/autoencoder/experiments/LatentShapeVAE/results/n01_similarity_v1",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["test", "ood_event"],
        help="Only keep entries for these splits.",
    )
    args = p.parse_args()

    csv_paths = [Path(x) for x in sorted(glob.glob(args.input_glob))]
    if not csv_paths:
        raise SystemExit(f"[ERROR] no files matched: {args.input_glob}")

    all_entries: List[Entry] = []
    for cp in csv_paths:
        all_entries.extend(_read_entries(cp))

    allowed_splits = set(args.splits)
    all_entries = [e for e in all_entries if e.split in allowed_splits]
    if not all_entries:
        raise SystemExit("[ERROR] no entries after split filter.")

    dedup = _dedup_best(all_entries)

    by_split: Dict[str, List[Entry]] = defaultdict(list)
    for e in dedup:
        by_split[e.split].append(e)

    # Robust normalization per split.
    for split, entries in by_split.items():
        kls = [e.kl_moment for e in entries]
        w2s = [e.w2_moment for e in entries]
        diags = [e.diag_mae for e in entries]
        eigs = [e.eig_log_abs for e in entries]

        z_kl = _robust_z(kls)
        z_w2 = _robust_z(w2s)
        z_diag = _robust_z(diags)
        z_eig = _robust_z(eigs)

        for i, e in enumerate(entries):
            e.z_kl = z_kl[i]
            e.z_w2 = z_w2[i]
            e.z_diag = z_diag[i]
            e.z_eig_log_abs = z_eig[i]
            e.n01_robust_gap = e.z_kl + e.z_w2 + 0.5 * e.z_diag + 0.25 * e.z_eig_log_abs
            # bounded similarity in (0,1); larger = more N(0,1)-like relative to peers.
            e.n01_similarity = _safe_inverse_logit(e.n01_robust_gap)
            e.shape_class = _shape_class(e.diag_mae, e.offdiag_mean_abs_corr, e.eig_ratio, e.kl_moment)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_rows: List[Dict[str, object]] = []
    for split in sorted(by_split.keys()):
        for e in sorted(by_split[split], key=lambda x: x.n01_robust_gap):
            per_rows.append(
                {
                    "run_name": e.run_name,
                    "split": e.split,
                    "shape_class": e.shape_class,
                    "source_csv": e.source_csv,
                    "mean_norm_l2": e.mean_norm_l2,
                    "diag_mae": e.diag_mae,
                    "offdiag_mean_abs_corr": e.offdiag_mean_abs_corr,
                    "eig_ratio": e.eig_ratio,
                    "kl_moment_to_std_normal": e.kl_moment,
                    "w2_moment_to_std_normal": e.w2_moment,
                    "eig_log_abs": e.eig_log_abs,
                    "n01_abs_gap": e.n01_abs_gap,
                    "z_kl": e.z_kl,
                    "z_w2": e.z_w2,
                    "z_diag": e.z_diag,
                    "z_eig_log_abs": e.z_eig_log_abs,
                    "n01_robust_gap": e.n01_robust_gap,
                    "n01_similarity": e.n01_similarity,
                }
            )

    per_csv = out_dir / "n01_similarity_per_run_split.csv"
    _write_csv(
        per_csv,
        per_rows,
        [
            "run_name",
            "split",
            "shape_class",
            "source_csv",
            "mean_norm_l2",
            "diag_mae",
            "offdiag_mean_abs_corr",
            "eig_ratio",
            "kl_moment_to_std_normal",
            "w2_moment_to_std_normal",
            "eig_log_abs",
            "n01_abs_gap",
            "z_kl",
            "z_w2",
            "z_diag",
            "z_eig_log_abs",
            "n01_robust_gap",
            "n01_similarity",
        ],
    )

    for split, entries in by_split.items():
        _write_markdown(out_dir / f"n01_similarity_ranking_{split}.md", split, entries)

    meta = {
        "input_glob": args.input_glob,
        "matched_files": [p.as_posix() for p in csv_paths],
        "n_input_rows": len(all_entries),
        "n_dedup_rows": len(dedup),
        "splits": sorted(by_split.keys()),
        "outputs": {
            "per_run_csv": per_csv.as_posix(),
            "ranking_md": [
                (out_dir / f"n01_similarity_ranking_{s}.md").as_posix() for s in sorted(by_split.keys())
            ],
        },
    }
    with (out_dir / "n01_similarity_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[INFO] output_dir:", out_dir.as_posix())
    print("[INFO] per_run_csv:", per_csv.as_posix())
    for s in sorted(by_split.keys()):
        print("[INFO] ranking_md:", (out_dir / f"n01_similarity_ranking_{s}.md").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
