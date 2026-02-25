import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence


def build_eventwise_split_indices(
    event_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("split ratios must sum to 1.0")

    e2i = defaultdict(list)
    for i, eid in enumerate(event_ids):
        e2i[eid].append(i)

    events = list(e2i.keys())
    rng = random.Random(seed)
    rng.shuffle(events)

    n = len(events)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_events = set(events[:n_train])
    val_events = set(events[n_train : n_train + n_val])

    out = {"train": [], "val": [], "test": []}
    for e, idxs in e2i.items():
        if e in train_events:
            out["train"].extend(idxs)
        elif e in val_events:
            out["val"].extend(idxs)
        else:
            out["test"].extend(idxs)
    for k in out:
        out[k].sort()
    return out


def save_split(split_indices: Dict[str, List[int]], event_ids: Sequence[str], out_path: str):
    payload = {}
    for name, idxs in split_indices.items():
        payload[name] = {
            "num_samples": len(idxs),
            "num_events": len(set(event_ids[i] for i in idxs)),
            "indices": idxs,
        }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

