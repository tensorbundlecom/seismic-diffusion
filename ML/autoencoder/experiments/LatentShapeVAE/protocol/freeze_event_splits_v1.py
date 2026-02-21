#!/usr/bin/env python3
"""
Freeze LatentShapeVAE V1 event-wise splits.

Outputs:
  - protocol/splits_v1/{train,val,test,ood_event}_files.txt
  - protocol/splits_v1/{train,val,test,ood_event}_event_ids.txt
  - protocol/frozen_event_splits_v1.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _read_yaml(path: Path) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_event_catalog(event_file: Path) -> Dict[str, Dict]:
    for enc in ["latin1", "windows-1254", "iso-8859-9", "utf-8"]:
        try:
            with event_file.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode event file: {event_file}")

    out: Dict[str, Dict] = {}
    for raw in rows:
        row = {str(k).strip(): v for k, v in raw.items()}
        event_id = str(row.get("Deprem Kodu", "")).strip()
        if not event_id:
            continue
        out[event_id] = {
            "event_id": event_id,
            "date": str(row.get("Olus tarihi", "")).strip(),
            "time": str(row.get("Olus zamani", "")).strip(),
            "lat": str(row.get("Enlem", "")).strip(),
            "lon": str(row.get("Boylam", "")).strip(),
            "depth_km": str(row.get("Der(km)", "")).strip(),
            "ml": str(row.get("ML", "")).strip(),
        }
    return out


def _event_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 1:
        return parts[0]
    return stem


def _parse_event_ts(event_id: str, event_row: Optional[Dict]) -> Optional[datetime]:
    # Prefer event_id timestamp (YYYYmmddHHMMSS)
    if len(event_id) >= 14 and event_id[:14].isdigit():
        try:
            return datetime.strptime(event_id[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    # Fallback to catalog fields
    if not event_row:
        return None
    d = str(event_row.get("date", "")).strip()
    t = str(event_row.get("time", "")).strip()
    if not d:
        return None
    d = d.replace(".", "-").replace("/", "-")
    t = t.split(".")[0]
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            if fmt.endswith("%S"):
                return datetime.strptime(f"{d} {t}", fmt).replace(tzinfo=timezone.utc)
            return datetime.strptime(d, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _sha256_lines(lines: Iterable[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _write_list(path: Path, items: List[str]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")
    return _sha256_lines(items)


def _split_train_val_test(events: List[str], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    if not events:
        return [], [], []
    rng = np.random.default_rng(seed)
    perm = rng.permutation(events).tolist()
    n = len(perm)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    n_train = max(1, min(n_train, n - 2)) if n >= 3 else max(1, n - 1)
    n_val = max(1, min(n_val, n - n_train - 1)) if n - n_train >= 2 else max(0, n - n_train)
    train = sorted(perm[:n_train])
    val = sorted(perm[n_train : n_train + n_val])
    test = sorted(perm[n_train + n_val :])
    return train, val, test


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze LatentShapeVAE event-wise split manifest (V1).")
    parser.add_argument(
        "--data-config",
        default="ML/autoencoder/experiments/LatentShapeVAE/configs/data_protocol_v1.yaml",
    )
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--event-file", default="")
    parser.add_argument("--output-dir", default="ML/autoencoder/experiments/LatentShapeVAE/protocol")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--ood-ratio", type=float, default=0.10)
    parser.add_argument("--ood-strategy", choices=["temporal", "random"], default="temporal")
    parser.add_argument("--min-files-per-event", type=int, default=1)
    args = parser.parse_args()

    cfg = _read_yaml(Path(args.data_config))
    data_dir = Path(args.data_dir or cfg["data_root"])
    event_file = Path(args.event_file or cfg["event_catalog"])
    out_dir = Path(args.output_dir)
    splits_dir = out_dir / "splits_v1"
    splits_dir.mkdir(parents=True, exist_ok=True)

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio and val_ratio must be > 0 and sum to < 1.0")
    if not (0.0 < args.ood_ratio < 0.5):
        raise ValueError("ood_ratio must be in (0, 0.5)")

    event_lookup = _read_event_catalog(event_file)
    files = sorted(data_dir.glob("*.mseed"))
    if not files:
        raise RuntimeError(f"No mseed files found in: {data_dir}")

    grouped: Dict[str, List[str]] = defaultdict(list)
    skipped_no_event = 0
    for fp in files:
        eid = _event_id_from_filename(fp.name)
        if eid not in event_lookup:
            skipped_no_event += 1
            continue
        grouped[eid].append(fp.as_posix())

    grouped = {k: sorted(v) for k, v in grouped.items() if len(v) >= args.min_files_per_event}
    event_ids = sorted(grouped.keys())
    if len(event_ids) < 20:
        raise RuntimeError(f"Too few events after filtering: {len(event_ids)}")

    n_events = len(event_ids)
    n_ood = max(1, int(round(args.ood_ratio * n_events)))

    if args.ood_strategy == "temporal":
        ranked = []
        for eid in event_ids:
            ts = _parse_event_ts(eid, event_lookup.get(eid))
            key = ts.timestamp() if ts else float("-inf")
            ranked.append((eid, key))
        ranked.sort(key=lambda x: x[1])
        ood_events = sorted([eid for eid, _ in ranked[-n_ood:]])
    else:
        rng = np.random.default_rng(args.seed)
        ood_events = sorted(rng.choice(event_ids, size=n_ood, replace=False).tolist())

    id_events = sorted(set(event_ids) - set(ood_events))
    train_events, val_events, test_events = _split_train_val_test(
        id_events,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Leakage checks
    s_train, s_val, s_test, s_ood = set(train_events), set(val_events), set(test_events), set(ood_events)
    if (s_train & s_val) or (s_train & s_test) or (s_val & s_test):
        raise RuntimeError("Leakage among train/val/test event sets.")
    if s_ood & (s_train | s_val | s_test):
        raise RuntimeError("OOD event leakage into train/val/test.")

    train_files = sorted([p for e in train_events for p in grouped[e]])
    val_files = sorted([p for e in val_events for p in grouped[e]])
    test_files = sorted([p for e in test_events for p in grouped[e]])
    ood_files = sorted([p for e in ood_events for p in grouped[e]])

    train_files_txt = splits_dir / "train_files.txt"
    val_files_txt = splits_dir / "val_files.txt"
    test_files_txt = splits_dir / "test_files.txt"
    ood_files_txt = splits_dir / "ood_event_files.txt"
    train_events_txt = splits_dir / "train_event_ids.txt"
    val_events_txt = splits_dir / "val_event_ids.txt"
    test_events_txt = splits_dir / "test_event_ids.txt"
    ood_events_txt = splits_dir / "ood_event_ids.txt"

    hashes = {
        "train_files": _write_list(train_files_txt, train_files),
        "val_files": _write_list(val_files_txt, val_files),
        "test_files": _write_list(test_files_txt, test_files),
        "ood_event_files": _write_list(ood_files_txt, ood_files),
        "train_event_ids": _write_list(train_events_txt, train_events),
        "val_event_ids": _write_list(val_events_txt, val_events),
        "test_event_ids": _write_list(test_events_txt, test_events),
        "ood_event_ids": _write_list(ood_events_txt, ood_events),
    }

    manifest = {
        "version": "v1",
        "status": "FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "policy": {
            "data_dir": data_dir.as_posix(),
            "event_file": event_file.as_posix(),
            "ood_strategy": args.ood_strategy,
            "ood_ratio": float(args.ood_ratio),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "min_files_per_event": int(args.min_files_per_event),
        },
        "counts": {
            "events_total": len(event_ids),
            "events_train": len(train_events),
            "events_val": len(val_events),
            "events_test": len(test_events),
            "events_ood": len(ood_events),
            "files_total_scanned": len(files),
            "files_total_used": len(train_files) + len(val_files) + len(test_files) + len(ood_files),
            "files_train": len(train_files),
            "files_val": len(val_files),
            "files_test": len(test_files),
            "files_ood": len(ood_files),
            "files_skipped_missing_event": skipped_no_event,
        },
        "splits": {
            "train_files": {
                "file": train_files_txt.as_posix(),
                "count": len(train_files),
                "sha256": hashes["train_files"],
            },
            "val_files": {
                "file": val_files_txt.as_posix(),
                "count": len(val_files),
                "sha256": hashes["val_files"],
            },
            "test_files": {
                "file": test_files_txt.as_posix(),
                "count": len(test_files),
                "sha256": hashes["test_files"],
            },
            "ood_event_files": {
                "file": ood_files_txt.as_posix(),
                "count": len(ood_files),
                "sha256": hashes["ood_event_files"],
            },
            "train_event_ids": {
                "file": train_events_txt.as_posix(),
                "count": len(train_events),
                "sha256": hashes["train_event_ids"],
            },
            "val_event_ids": {
                "file": val_events_txt.as_posix(),
                "count": len(val_events),
                "sha256": hashes["val_event_ids"],
            },
            "test_event_ids": {
                "file": test_events_txt.as_posix(),
                "count": len(test_events),
                "sha256": hashes["test_event_ids"],
            },
            "ood_event_ids": {
                "file": ood_events_txt.as_posix(),
                "count": len(ood_events),
                "sha256": hashes["ood_event_ids"],
            },
        },
    }

    out_manifest = out_dir / "frozen_event_splits_v1.json"
    with out_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[INFO] Frozen split manifest written:", out_manifest)
    print(
        "[INFO] Events train/val/test/ood:",
        len(train_events),
        len(val_events),
        len(test_events),
        len(ood_events),
    )
    print(
        "[INFO] Files train/val/test/ood:",
        len(train_files),
        len(val_files),
        len(test_files),
        len(ood_files),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
