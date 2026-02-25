import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def build_eventwise_split_indices(
    event_ids: Sequence[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    event_to_indices = defaultdict(list)
    for idx, eid in enumerate(event_ids):
        event_to_indices[eid].append(idx)

    unique_events = list(event_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_events)

    n = len(unique_events)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_events = set(unique_events[:n_train])
    val_events = set(unique_events[n_train : n_train + n_val])
    test_events = set(unique_events[n_train + n_val :])

    out = {"train": [], "val": [], "test": []}
    for eid, idxs in event_to_indices.items():
        if eid in train_events:
            out["train"].extend(idxs)
        elif eid in val_events:
            out["val"].extend(idxs)
        else:
            out["test"].extend(idxs)

    for k in out:
        out[k].sort()
    return out


def save_split_file(
    split_indices: Dict[str, List[int]],
    event_ids: Sequence[str],
    out_path: str,
) -> None:
    payload = {}
    for split_name, idxs in split_indices.items():
        evs = sorted(set(event_ids[i] for i in idxs))
        payload[split_name] = {"num_samples": len(idxs), "num_events": len(evs), "indices": idxs, "events": evs}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_split_file(split_file: str) -> Dict[str, List[int]]:
    with open(split_file, "r") as f:
        payload = json.load(f)
    return {
        "train": payload["train"]["indices"],
        "val": payload["val"]["indices"],
        "test": payload["test"]["indices"],
    }


def summarize_split(split_indices: Dict[str, List[int]], event_ids: Sequence[str]) -> Dict[str, Dict[str, int]]:
    out = {}
    for name, idxs in split_indices.items():
        out[name] = {"num_samples": len(idxs), "num_events": len(set(event_ids[i] for i in idxs))}
    return out

