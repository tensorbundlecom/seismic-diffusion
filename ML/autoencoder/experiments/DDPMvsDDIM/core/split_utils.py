import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def pick_event_magnitude(event_catalog, event_id: str, primary_col: str = "ML", fallback_col: str = "xM"):
    rows = event_catalog[event_catalog["event_id"] == event_id]
    if len(rows) == 0:
        return None
    row = rows.iloc[0]
    primary_value = row.get(primary_col)
    if primary_value == primary_value:
        return float(primary_value)
    fallback_value = row.get(fallback_col)
    if fallback_value == fallback_value:
        return float(fallback_value)
    return None


def magnitude_bin_label(magnitude):
    if magnitude is None:
        return "nan"
    if magnitude < 3.0:
        return "lt3"
    if magnitude < 4.0:
        return "3to4"
    if magnitude < 5.0:
        return "4to5"
    return "ge5"


def build_event_index(event_ids):
    event_to_indices = defaultdict(list)
    for idx, event_id in enumerate(event_ids):
        event_to_indices[event_id].append(idx)
    return event_to_indices


def split_event_list(events, train_ratio, val_ratio, rng):
    events = list(events)
    rng.shuffle(events)
    n_events = len(events)
    n_train = int(math.floor(n_events * train_ratio))
    n_val = int(math.floor(n_events * val_ratio))
    train_events = events[:n_train]
    val_events = events[n_train : n_train + n_val]
    test_events = events[n_train + n_val :]
    return train_events, val_events, test_events


def build_hybrid_eventwise_split(
    event_ids,
    event_catalog,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    primary_mag_col="ML",
    fallback_mag_col="xM",
):
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    event_to_indices = build_event_index(event_ids)
    unique_events = sorted(event_to_indices.keys())

    event_info = {}
    bin_to_events = defaultdict(list)
    for event_id in unique_events:
        magnitude = pick_event_magnitude(event_catalog, event_id, primary_col=primary_mag_col, fallback_col=fallback_mag_col)
        bin_label = magnitude_bin_label(magnitude)
        event_info[event_id] = {"magnitude": magnitude, "bin": bin_label}
        bin_to_events[bin_label].append(event_id)

    rng = random.Random(seed)
    split_events = {"train": [], "val": [], "test": []}

    for bin_label in ("lt3", "3to4", "4to5", "nan"):
        events = list(bin_to_events.get(bin_label, []))
        train_events, val_events, test_events = split_event_list(events, train_ratio, val_ratio, rng)
        split_events["train"].extend(train_events)
        split_events["val"].extend(val_events)
        split_events["test"].extend(test_events)

    # Frozen rare-event policy: all M>=5 events go to train.
    split_events["train"].extend(sorted(bin_to_events.get("ge5", [])))

    split_indices = {"train": [], "val": [], "test": []}
    for split_name, events in split_events.items():
        for event_id in events:
            split_indices[split_name].extend(event_to_indices[event_id])
        split_indices[split_name].sort()

    return split_indices, event_info


def summarize_split(split_indices, event_ids, event_info):
    summary = {}
    for split_name, indices in split_indices.items():
        events = sorted(set(event_ids[idx] for idx in indices))
        bin_counter = Counter(event_info[event_id]["bin"] for event_id in events)
        summary[split_name] = {
            "num_samples": len(indices),
            "num_events": len(events),
            "magnitude_bins": dict(sorted(bin_counter.items())),
            "events": events,
        }
    return summary


def save_split_artifacts(split_indices, event_ids, event_info, split_file, summary_file):
    split_payload = {}
    summary_payload = summarize_split(split_indices, event_ids, event_info)
    for split_name, indices in split_indices.items():
        split_payload[split_name] = {
            "indices": indices,
            "events": summary_payload[split_name]["events"],
            "num_samples": summary_payload[split_name]["num_samples"],
            "num_events": summary_payload[split_name]["num_events"],
            "magnitude_bins": summary_payload[split_name]["magnitude_bins"],
        }

    Path(split_file).parent.mkdir(parents=True, exist_ok=True)
    with open(split_file, "w") as handle:
        json.dump(split_payload, handle, indent=2)

    Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "w") as handle:
        json.dump(summary_payload, handle, indent=2)


def load_split_indices(split_file):
    with open(split_file, "r") as handle:
        payload = json.load(handle)
    return {
        "train": payload["train"]["indices"],
        "val": payload["val"]["indices"],
        "test": payload["test"]["indices"],
    }
