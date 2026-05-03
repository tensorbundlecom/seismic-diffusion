from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.frozen_config import load_frozen_config
from setup.metadata import dump_json, dump_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Build event-wise PaperRepro split artifacts from the condition manifest.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def _resolve_artifact_paths(cfg: dict) -> dict[str, Path]:
    experiment_root = Path(__file__).resolve().parents[1]
    artifact_cfg = cfg["operations"]["artifacts"]
    return {key: experiment_root / rel_path for key, rel_path in artifact_cfg.items()}


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _zscore_stats(values: list[float]) -> dict[str, float]:
    mean_value = sum(values) / max(len(values), 1)
    variance = sum((value - mean_value) ** 2 for value in values) / max(len(values), 1)
    std_value = variance ** 0.5
    if std_value <= 0.0:
        std_value = 1.0
    return {"mean": mean_value, "std": std_value}


def _assign_event_splits(event_ids: list[str], *, split_seed: int, ood_ratio: float, main_ratios: dict[str, float]) -> dict[str, list[str]]:
    rng = random.Random(split_seed)
    ordered = sorted(event_ids)
    rng.shuffle(ordered)

    n_total = len(ordered)
    n_ood = int(round(n_total * ood_ratio))
    n_ood = min(max(n_ood, 1), max(n_total - 3, 1)) if n_total >= 4 else 0
    ood_events = ordered[:n_ood]
    remaining = ordered[n_ood:]

    train_ratio = float(main_ratios["train"])
    val_ratio = float(main_ratios["validation"])
    test_ratio = float(main_ratios["test"])
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    n_remaining = len(remaining)
    n_train = int(n_remaining * train_ratio)
    n_val = int(n_remaining * val_ratio)
    n_test = n_remaining - n_train - n_val

    train_events = remaining[:n_train]
    val_events = remaining[n_train : n_train + n_val]
    test_events = remaining[n_train + n_val : n_train + n_val + n_test]

    return {
        "train": sorted(train_events),
        "validation": sorted(val_events),
        "test": sorted(test_events),
        "ood": sorted(ood_events),
    }


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config)
    artifact_paths = _resolve_artifact_paths(cfg)

    split_json_path = artifact_paths["event_split_json"]
    split_md_path = artifact_paths["event_split_md"]
    sample_manifest_path = artifact_paths["sample_manifest_jsonl"]
    sample_manifest_meta_path = artifact_paths["sample_manifest_meta_json"]
    norm_stats_path = artifact_paths["condition_norm_stats_json"]

    if not args.overwrite and any(path.exists() for path in [split_json_path, sample_manifest_path, norm_stats_path]):
        raise SystemExit("Refusing to overwrite existing split artifacts without --overwrite.")

    manifest_rows = _load_jsonl(artifact_paths["condition_manifest_jsonl"])
    usable_rows = [row for row in manifest_rows if bool(row.get("sample_usable", False))]
    if not usable_rows:
        raise SystemExit("No usable rows found in condition manifest.")

    event_to_rows: dict[str, list[dict]] = defaultdict(list)
    for row in usable_rows:
        event_to_rows[str(row["event_id"])].append(row)

    split_cfg = cfg["split"]
    split_map = _assign_event_splits(
        list(event_to_rows.keys()),
        split_seed=int(split_cfg["split_seed"]),
        ood_ratio=float(split_cfg["ood_ratio"]),
        main_ratios=split_cfg["ratios"],
    )
    event_to_split = {event_id: split_name for split_name, events in split_map.items() for event_id in events}

    sample_manifest_rows: list[dict] = []
    split_sample_counts = Counter()
    split_event_counts = {split_name: len(events) for split_name, events in split_map.items()}
    for row in usable_rows:
        split_name = event_to_split[str(row["event_id"])]
        sample_manifest_row = dict(row)
        sample_manifest_row["split"] = split_name
        sample_manifest_rows.append(sample_manifest_row)
        split_sample_counts[split_name] += 1

    dump_jsonl(sample_manifest_path, sample_manifest_rows)

    norm_features = list(cfg["conditions"]["normalization"]["zscore"])
    train_rows = [row for row in sample_manifest_rows if row["split"] == "train"]
    condition_norm_stats = {
        "split_seed": int(split_cfg["split_seed"]),
        "train_sample_count": len(train_rows),
        "zscore_features": norm_features,
        "passthrough_features": list(cfg["conditions"]["normalization"]["passthrough"]),
        "stats": {
            feature: _zscore_stats([float(row[feature]) for row in train_rows if row.get(feature) is not None])
            for feature in norm_features
        },
    }
    dump_json(norm_stats_path, condition_norm_stats)

    split_payload = {
        "version": cfg["version"],
        "split_seed": int(split_cfg["split_seed"]),
        "ood_ratio": float(split_cfg["ood_ratio"]),
        "main_ratios": dict(split_cfg["ratios"]),
        "event_counts": split_event_counts,
        "sample_counts": dict(split_sample_counts),
        "events": split_map,
    }
    dump_json(split_json_path, split_payload)

    sample_manifest_meta = {
        "version": cfg["version"],
        "usable_sample_count": len(sample_manifest_rows),
        "split_seed": int(split_cfg["split_seed"]),
        "split_event_counts": split_event_counts,
        "split_sample_counts": dict(split_sample_counts),
    }
    dump_json(sample_manifest_meta_path, sample_manifest_meta)

    lines = [
        "# Event Splits",
        "",
        f"- `version`: `{cfg['version']}`",
        f"- `split_seed`: `{split_cfg['split_seed']}`",
        f"- `ood_ratio`: `{split_cfg['ood_ratio']}`",
        f"- `split_event_counts`: `{split_event_counts}`",
        f"- `split_sample_counts`: `{dict(split_sample_counts)}`",
    ]
    split_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
