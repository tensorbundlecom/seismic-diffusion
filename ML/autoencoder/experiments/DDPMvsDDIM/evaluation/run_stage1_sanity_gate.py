import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ML.autoencoder.experiments.DDPMvsDDIM.core.metrics import spec_corr
from ML.autoencoder.experiments.DDPMvsDDIM.core.model_legacy_cond_baseline import LegacyCondBaselineCVAE
from ML.autoencoder.experiments.DDPMvsDDIM.core.split_utils import load_split_indices
from ML.autoencoder.experiments.DDPMvsDDIM.core.stft_dataset import (
    SeismicSTFTDatasetWithMetadata,
    collate_fn_with_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen latent sanity gate for Stage-1 event-wise checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default="ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/stage1_eventwise_v1_best.pt",
    )
    parser.add_argument(
        "--split-file",
        default="ML/autoencoder/experiments/DDPMvsDDIM/protocol/eventwise_split_v1.json",
    )
    parser.add_argument("--data-dir", default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument(
        "--event-file",
        default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt",
    )
    parser.add_argument("--station-list-file", default="data/station_list_external_full.json")
    parser.add_argument("--channels", nargs="+", default=["HH"])
    parser.add_argument("--magnitude-col", default="ML")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--plot-count", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        default="ML/autoencoder/experiments/DDPMvsDDIM/results/stage1_sanity_gate",
    )
    return parser.parse_args()


def build_model(checkpoint_path, num_stations, device):
    state = torch.load(checkpoint_path, map_location=device)
    config = state.get("config", {})
    model = LegacyCondBaselineCVAE(
        in_channels=config.get("in_channels", 3),
        latent_dim=config.get("latent_dim", 128),
        num_stations=config.get("num_stations", num_stations),
        w_dim=config.get("w_dim", config.get("cond_embedding_dim", 64)),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, config


def has_finite_tensor(*tensors):
    return all(torch.isfinite(tensor).all().item() for tensor in tensors)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    plot_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    with open(args.station_list_file, "r") as handle:
        station_list = json.load(handle)
    split_indices = load_split_indices(args.split_file)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=args.channels,
        magnitude_col=args.magnitude_col,
        station_list=station_list,
    )
    val_indices = split_indices["val"][: args.subset_size]
    subset = Subset(dataset, val_indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn_with_metadata,
    )

    model, config = build_model(args.checkpoint, num_stations=len(station_list), device=device)
    all_mu = []
    spec_corrs = []
    mu_norms = []
    logvar_means = []
    finite_ok = True
    plot_payloads = []

    with torch.no_grad():
        for batch in loader:
            specs, mags, locs, stations, metas = batch
            if specs is None:
                continue
            specs = specs.to(device)
            mags = mags.to(device)
            locs = locs.to(device)
            stations = stations.to(device)

            recon, mu, logvar = model(specs, mags, locs, stations)
            finite_ok = finite_ok and has_finite_tensor(recon, mu, logvar)
            all_mu.append(mu.cpu())
            mu_norms.extend(mu.norm(dim=1).cpu().tolist())
            logvar_means.extend(logvar.mean(dim=1).cpu().tolist())

            for sample_idx in range(specs.size(0)):
                original_spec = specs[sample_idx, 2].cpu().numpy()
                recon_spec = recon[sample_idx, 2].cpu().numpy()
                spec_corrs.append(spec_corr(original_spec, recon_spec))
                if len(plot_payloads) < args.plot_count:
                    plot_payloads.append(
                        {
                            "original_spec": original_spec,
                            "recon_spec": recon_spec,
                            "meta": metas[sample_idx],
                        }
                    )

    if not all_mu:
        raise RuntimeError("Stage-1 sanity gate received no valid samples.")

    mu_tensor = torch.cat(all_mu, dim=0)
    per_dim_std_mean = float(mu_tensor.std(dim=0, unbiased=False).mean().item())
    mean_spec_corr = float(np.mean(spec_corrs))
    mean_mu_norm = float(np.mean(mu_norms))
    mean_logvar = float(np.mean(logvar_means))

    gate_checks = {
        "finite_tensors": bool(finite_ok),
        "mean_spec_corr_ge_0p88": mean_spec_corr >= 0.88,
        "mean_mu_norm_in_range": 0.5 <= mean_mu_norm <= 25.0,
        "mean_logvar_in_range": -6.0 <= mean_logvar <= 2.0,
        "per_dim_mu_std_mean_ge_0p01": per_dim_std_mean >= 0.01,
        "visual_review_required": True,
    }
    automatic_gate_pass = all(
        gate_checks[key]
        for key in (
            "finite_tensors",
            "mean_spec_corr_ge_0p88",
            "mean_mu_norm_in_range",
            "mean_logvar_in_range",
            "per_dim_mu_std_mean_ge_0p01",
        )
    )

    for plot_idx, payload in enumerate(plot_payloads):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(payload["original_spec"], aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Original Z STFT")
        axes[1].imshow(payload["recon_spec"], aspect="auto", origin="lower", cmap="viridis")
        axes[1].set_title(f"Recon Z STFT\ncorr={spec_corr(payload['original_spec'], payload['recon_spec']):.4f}")
        fig.suptitle(f"{payload['meta']['event_id']} @ {payload['meta']['station_name']}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{plot_idx:02d}_{payload['meta']['event_id']}_{payload['meta']['station_name']}.png", dpi=150)
        plt.close(fig)

    summary = {
        "checkpoint": args.checkpoint,
        "config": config,
        "subset_size": len(mu_tensor),
        "mean_spec_corr": mean_spec_corr,
        "mean_mu_norm": mean_mu_norm,
        "mean_logvar": mean_logvar,
        "per_dim_mu_std_mean": per_dim_std_mean,
        "gate_checks": gate_checks,
        "automatic_gate_pass": automatic_gate_pass,
        "visual_review_required": True,
        "visual_review_note": "First 8 STFT sanity plots must be reviewed manually before diffusion training.",
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Stage-1 sanity gate summary saved to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
