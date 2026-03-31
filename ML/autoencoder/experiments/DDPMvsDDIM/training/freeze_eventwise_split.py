import argparse
import json

from ML.autoencoder.experiments.DDPMvsDDIM.core.split_utils import (
    build_hybrid_eventwise_split,
    save_split_artifacts,
)
from ML.autoencoder.experiments.DDPMvsDDIM.core.stft_dataset import SeismicSTFTDatasetWithMetadata


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze hybrid event-wise split for DDPMvsDDIM.")
    parser.add_argument("--data-dir", default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument(
        "--event-file",
        default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt",
    )
    parser.add_argument("--station-list-file", default="data/station_list_external_full.json")
    parser.add_argument("--channels", nargs="+", default=["HH"])
    parser.add_argument("--magnitude-col", default="ML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--split-file",
        default="ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_v1.json",
    )
    parser.add_argument(
        "--summary-file",
        default="ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_summary_v1.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.station_list_file, "r") as handle:
        station_list = json.load(handle)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=args.channels,
        magnitude_col=args.magnitude_col,
        station_list=station_list,
    )
    event_ids = [dataset._extract_event_id_from_filename(path.name) for path in dataset.file_paths]
    split_indices, event_info = build_hybrid_eventwise_split(
        event_ids=event_ids,
        event_catalog=dataset.event_catalog,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        primary_mag_col="ML",
        fallback_mag_col="xM",
    )
    save_split_artifacts(split_indices, event_ids, event_info, args.split_file, args.summary_file)
    print(f"[INFO] Saved event-wise split: {args.split_file}")
    print(f"[INFO] Saved split summary: {args.summary_file}")


if __name__ == "__main__":
    main()
