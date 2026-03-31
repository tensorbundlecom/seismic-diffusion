import argparse
import json

import torch

from ML.autoencoder.experiments.DDPMvsDDIM.core.model_legacy_cond_baseline import LegacyCondBaselineCVAE
from ML.autoencoder.experiments.DDPMvsDDIM.core.stft_dataset import (
    SeismicSTFTDatasetWithMetadata,
    collate_fn_with_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test isolated DDPMvsDDIM baseline copy.")
    parser.add_argument("--data-dir", default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument(
        "--event-file",
        default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt",
    )
    parser.add_argument("--station-list-file", default="data/station_list_external_full.json")
    parser.add_argument("--channels", nargs="+", default=["HH"])
    parser.add_argument("--magnitude-col", default="ML")
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.station_list_file, "r") as handle:
        station_list = json.load(handle)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=args.channels,
        magnitude_col=args.magnitude_col,
        station_list=station_list,
    )

    batch_items = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if "error" not in item[-1]:
            batch_items.append(item)
        if len(batch_items) == args.batch_size:
            break

    if len(batch_items) < args.batch_size:
        raise RuntimeError(f"Could not assemble a valid smoke batch of size {args.batch_size}")

    specs, mags, locs, stations, metadata = collate_fn_with_metadata(batch_items)
    model = LegacyCondBaselineCVAE(
        in_channels=specs.shape[1],
        latent_dim=128,
        num_stations=len(station_list),
        cond_embedding_dim=64,
    ).to(device)
    model.train()

    specs = specs.to(device)
    mags = mags.to(device)
    locs = locs.to(device)
    stations = stations.to(device)

    recon, mu, logvar = model(specs, mags, locs, stations)
    generated = model.sample(
        num_samples=specs.shape[0],
        magnitude=mags,
        location=locs,
        station_idx=stations,
        device=device,
    )

    print("[SMOKE] device:", device)
    print("[SMOKE] batch shape:", tuple(specs.shape))
    print("[SMOKE] recon shape:", tuple(recon.shape))
    print("[SMOKE] mu shape:", tuple(mu.shape))
    print("[SMOKE] logvar shape:", tuple(logvar.shape))
    print("[SMOKE] sample shape:", tuple(generated.shape))
    print("[SMOKE] files:", [meta["file_name"] for meta in metadata])


if __name__ == "__main__":
    main()
