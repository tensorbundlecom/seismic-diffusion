from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from core.datasets import build_stage1_dataset_from_config
from core.frozen_config import load_frozen_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit Stage-1 dataset tensor contract across splits.")
    parser.add_argument("--config", type=str, default=None, help="Frozen config path.")
    parser.add_argument("--per-split", type=int, default=4, help="Number of examples to inspect per split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_frozen_config(args.config)
    output_json = EXPERIMENT_ROOT / "results" / "setup" / "stage1_dataset_audit_paper_repro_v1.json"
    output_md = EXPERIMENT_ROOT / "results" / "setup" / "stage1_dataset_audit_paper_repro_v1.md"
    if not args.overwrite and (output_json.exists() or output_md.exists()):
        raise SystemExit("Refusing to overwrite existing dataset audit artifacts without --overwrite.")

    split_summaries = {}
    for split in ["train", "validation", "test", "ood"]:
        dataset = build_stage1_dataset_from_config(cfg, splits=[split])
        examples = []
        for index in range(min(args.per_split, len(dataset))):
            item = dataset[index]
            rep = item["representation"]
            examples.append(
                {
                    "index": index,
                    "event_id": item["event_id"],
                    "station_code": item["station_code"],
                    "shape": list(rep.shape),
                    "min": float(np.min(rep)),
                    "max": float(np.max(rep)),
                    "requires_left_pad": bool(item["requires_left_pad"]),
                    "requires_right_pad": bool(item["requires_right_pad"]),
                }
            )
        split_summaries[split] = {
            "dataset_len": len(dataset),
            "examples": examples,
        }

    payload = {"version": cfg["version"], "splits": split_summaries}
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# Stage1 Dataset Audit", ""]
    for split, summary in split_summaries.items():
        lines.append(f"## {split}")
        lines.append("")
        lines.append(f"- `dataset_len`: `{summary['dataset_len']}`")
        for example in summary["examples"]:
            lines.append(
                f"- `idx={example['index']}` `{example['event_id']}_{example['station_code']}` "
                f"shape=`{example['shape']}` range=`[{example['min']:.4f}, {example['max']:.4f}]` "
                f"left_pad=`{example['requires_left_pad']}` right_pad=`{example['requires_right_pad']}`"
            )
        lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
