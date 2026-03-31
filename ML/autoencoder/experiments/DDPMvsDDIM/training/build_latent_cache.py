import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from ML.autoencoder.experiments.DDPMvsDDIM.core.model_legacy_cond_baseline import LegacyCondBaselineCVAE
from ML.autoencoder.experiments.DDPMvsDDIM.core.split_utils import load_split_indices
from ML.autoencoder.experiments.DDPMvsDDIM.core.stft_dataset import (
    SeismicSTFTDatasetWithMetadata,
    collate_fn_with_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build latent cache from localized baseline VAE.")
    parser.add_argument(
        "--checkpoint",
        default="ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/stage1_eventwise_v1_best.pt",
    )
    parser.add_argument("--data-dir", default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument(
        "--event-file",
        default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt",
    )
    parser.add_argument("--station-list-file", default="data/station_list_external_full.json")
    parser.add_argument("--channels", nargs="+", default=["HH"])
    parser.add_argument("--magnitude-col", default="ML")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1",
    )
    parser.add_argument("--split-file", default="ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_v1.json")
    return parser.parse_args()


def apply_limit(indices, limit):
    if limit > 0:
        return indices[:limit]
    return indices


@torch.no_grad()
def extract_split_cache(model, loader, device):
    z_mu_all = []
    z_logvar_all = []
    cond_embedding_all = []
    raw_condition_all = []
    station_idx_all = []
    magnitude_all = []
    location_all = []
    meta = {"dataset_index": [], "event_id": [], "file_name": [], "file_path": []}

    for batch in loader:
        specs, mags, locs, stas, metas = batch
        if specs is None:
            continue

        specs = specs.to(device)
        mags = mags.to(device)
        locs = locs.to(device)
        stas = stas.to(device)

        mu, logvar, cond_embedding = model.encode(specs, mags, locs, stas)
        raw_condition = torch.cat([model._normalize_magnitude(mags).unsqueeze(1), torch.clamp(locs, 0.0, 1.0)], dim=1)

        z_mu_all.append(mu.cpu())
        z_logvar_all.append(logvar.cpu())
        cond_embedding_all.append(cond_embedding.cpu())
        raw_condition_all.append(raw_condition.cpu())
        station_idx_all.append(stas.cpu())
        magnitude_all.append(mags.cpu())
        location_all.append(locs.cpu())

        for item in metas:
            meta["dataset_index"].append(int(item.get("dataset_index", -1)))
            meta["event_id"].append(item.get("event_id", "UNKNOWN"))
            meta["file_name"].append(item.get("file_name", "UNKNOWN"))
            meta["file_path"].append(item.get("file_path", ""))

    return {
        "z_mu": torch.cat(z_mu_all, dim=0),
        "z_logvar": torch.cat(z_logvar_all, dim=0),
        "cond_embedding": torch.cat(cond_embedding_all, dim=0),
        "raw_condition": torch.cat(raw_condition_all, dim=0),
        "station_idx": torch.cat(station_idx_all, dim=0),
        "magnitude": torch.cat(magnitude_all, dim=0),
        "location": torch.cat(location_all, dim=0),
        "meta": meta,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.station_list_file, "r") as handle:
        station_list = json.load(handle)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=args.channels,
        magnitude_col=args.magnitude_col,
        station_list=station_list,
    )

    split_indices = load_split_indices(args.split_file)
    split_indices["train"] = apply_limit(split_indices["train"], args.limit_train)
    split_indices["val"] = apply_limit(split_indices["val"], args.limit_val)
    split_indices["test"] = apply_limit(split_indices["test"], args.limit_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)
    config = state.get("config", {})
    model = LegacyCondBaselineCVAE(
        in_channels=config.get("in_channels", 3),
        latent_dim=config.get("latent_dim", 128),
        num_stations=config.get("num_stations", len(station_list)),
        w_dim=config.get("w_dim", config.get("cond_embedding_dim", 64)),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    payloads = {}
    for split_name in ("train", "val", "test"):
        subset = Subset(dataset, split_indices[split_name])
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_fn_with_metadata,
        )
        print(f"[INFO] Extracting latent cache for split={split_name} n={len(subset)}")
        payload = extract_split_cache(model, loader, device)
        payloads[split_name] = payload
        out_path = output_dir / f"{split_name}_latent_cache.pt"
        torch.save(payload, out_path)
        print(f"[INFO] Saved cache: {out_path}")

    z_train = payloads["train"]["z_mu"]
    z_mean = z_train.mean(dim=0)
    z_std = z_train.std(dim=0, unbiased=False).clamp(min=1e-6)
    torch.save({"z_mean": z_mean, "z_std": z_std}, output_dir / "latent_stats.pt")
    summary = {
        "latent_dim": int(z_mean.numel()),
        "train_size": int(z_train.size(0)),
        "z_mean_abs_mean": float(z_mean.abs().mean().item()),
        "z_std_mean": float(z_std.mean().item()),
        "z_std_min": float(z_std.min().item()),
        "z_std_max": float(z_std.max().item()),
    }
    with open(output_dir / "latent_stats_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[INFO] Saved latent stats: {output_dir / 'latent_stats.pt'}")


if __name__ == "__main__":
    main()
