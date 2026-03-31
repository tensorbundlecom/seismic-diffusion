import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import obspy
import torch
from scipy import signal

from ML.autoencoder.experiments.DDPMvsDDIM.core.model_legacy_cond_baseline import LegacyCondBaselineCVAE
from ML.autoencoder.experiments.DDPMvsDDIM.core.stft_dataset import SeismicSTFTDatasetWithMetadata


def parse_args():
    parser = argparse.ArgumentParser(description="Run baseline-only sanity evaluation inside DDPMvsDDIM.")
    parser.add_argument(
        "--checkpoint",
        default="ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/legacy_cond_baseline_best.pt",
    )
    parser.add_argument("--data-dir", default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument(
        "--event-file",
        default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt",
    )
    parser.add_argument("--station-list-file", default="data/station_list_external_full.json")
    parser.add_argument("--channels", nargs="+", default=["HH"])
    parser.add_argument("--magnitude-col", default="ML")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        default="ML/autoencoder/experiments/DDPMvsDDIM/visualizations/baseline_sanity",
    )
    return parser.parse_args()


def reconstruct_signal(magnitude_spec, mag_min, mag_max, fs=100.0, nperseg=256, noverlap=192, nfft=256, n_iter=64):
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)

    phase = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    for _ in range(n_iter):
        stft_complex = spec * phase
        _, waveform = signal.istft(
            stft_complex,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary="zeros",
        )
        _, _, new_zxx = signal.stft(
            waveform,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary="zeros",
        )
        if new_zxx.shape != spec.shape:
            min_f = min(new_zxx.shape[0], spec.shape[0])
            min_t = min(new_zxx.shape[1], spec.shape[1])
            next_phase = np.zeros_like(spec, dtype=complex)
            next_phase[:min_f, :min_t] = np.exp(1j * np.angle(new_zxx[:min_f, :min_t]))
            phase = next_phase
        else:
            phase = np.exp(1j * np.angle(new_zxx))

    stft_complex = spec * phase
    _, waveform = signal.istft(
        stft_complex,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        boundary="zeros",
    )
    return waveform


def load_original_z_waveform(file_path):
    stream = obspy.read(file_path)
    stream.merge(fill_value=0)
    z_stream = stream.select(component="Z")
    trace = z_stream[0] if len(z_stream) > 0 else stream[0]
    return trace.data.astype(np.float32), float(trace.stats.sampling_rate)


def select_valid_samples(dataset, max_samples):
    selected = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        if "error" in item[-1]:
            continue
        selected.append((idx, item))
        if len(selected) == max_samples:
            break
    return selected


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


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    specs_dir = output_dir / "specs"
    wav_dir = output_dir / "waveforms"
    output_dir.mkdir(parents=True, exist_ok=True)
    specs_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    with open(args.station_list_file, "r") as handle:
        station_list = json.load(handle)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=args.channels,
        magnitude_col=args.magnitude_col,
        station_list=station_list,
    )
    model, config = build_model(args.checkpoint, num_stations=len(station_list), device=device)
    print(f"[INFO] Loaded local checkpoint: {args.checkpoint}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Config: {config}")

    results = []
    for idx, item in select_valid_samples(dataset, args.max_samples):
        spec, magnitude, location, station_idx, metadata = item

        spec_in = spec.unsqueeze(0).to(device)
        mag_in = magnitude.unsqueeze(0).to(device)
        loc_in = location.unsqueeze(0).to(device)
        sta_in = station_idx.unsqueeze(0).to(device)

        with torch.no_grad():
            recon, mu, logvar = model(spec_in, mag_in, loc_in, sta_in)
            sampled = model.sample(
                num_samples=1,
                magnitude=mag_in,
                location=loc_in,
                station_idx=sta_in,
                device=device,
                output_size=spec.shape[1:],
            )

        original_spec = spec[2].cpu().numpy()
        recon_spec = recon[0, 2].cpu().numpy()
        sample_spec = sampled[0, 2].cpu().numpy()

        mag_min = float(metadata.get("mag_min", 0.0))
        mag_max = float(metadata.get("mag_max", 1.0))
        original_waveform, fs = load_original_z_waveform(metadata["file_path"])
        recon_waveform = reconstruct_signal(recon_spec, mag_min=mag_min, mag_max=mag_max, fs=fs)
        sample_waveform = reconstruct_signal(sample_spec, mag_min=mag_min, mag_max=mag_max, fs=fs)

        spec_mse = float(np.mean((original_spec - recon_spec) ** 2))
        spec_corr = float(np.corrcoef(original_spec.flatten(), recon_spec.flatten())[0, 1])
        result = {
            "dataset_index": idx,
            "file_name": metadata["file_name"],
            "event_id": metadata["event_id"],
            "station_name": metadata["station_name"],
            "magnitude": float(metadata["magnitude"]),
            "spec_mse": spec_mse,
            "spec_corr": spec_corr,
            "mu_norm": float(mu.norm().item()),
            "logvar_mean": float(logvar.mean().item()),
            "waveform_note": "generated waveforms use Griffin-Lim and are qualitative sanity outputs",
        }
        results.append(result)

        title_root = f"{metadata['event_id']} @ {metadata['station_name']}"

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].imshow(original_spec, aspect="auto", origin="lower", cmap="viridis")
        axes[0].set_title("Original STFT (Z)")
        axes[1].imshow(recon_spec, aspect="auto", origin="lower", cmap="viridis")
        axes[1].set_title(f"Recon STFT\ncorr={spec_corr:.4f}")
        axes[2].imshow(sample_spec, aspect="auto", origin="lower", cmap="viridis")
        axes[2].set_title("Condition Sample STFT")
        fig.suptitle(title_root)
        fig.tight_layout()
        fig.savefig(specs_dir / f"{metadata['event_id']}_{metadata['station_name']}_specs.png", dpi=150)
        plt.close(fig)

        min_len = min(len(original_waveform), len(recon_waveform), len(sample_waveform))
        time_axis = np.arange(min_len) / fs
        fig, axes = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
        axes[0].plot(time_axis, original_waveform[:min_len], linewidth=0.8)
        axes[0].set_title("Original Z waveform")
        axes[1].plot(time_axis, recon_waveform[:min_len], linewidth=0.8)
        axes[1].set_title("Recon waveform (Griffin-Lim)")
        axes[2].plot(time_axis, sample_waveform[:min_len], linewidth=0.8)
        axes[2].set_title("Condition sample waveform (Griffin-Lim)")
        axes[2].set_xlabel("Time (s)")
        fig.suptitle(title_root)
        fig.tight_layout()
        fig.savefig(wav_dir / f"{metadata['event_id']}_{metadata['station_name']}_waveforms.png", dpi=150)
        plt.close(fig)

        print(
            f"[INFO] Saved sanity plots for {metadata['file_name']} | "
            f"spec_mse={spec_mse:.6f} spec_corr={spec_corr:.6f}"
        )

    summary = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_samples": len(results),
        "results": results,
    }
    with open(output_dir / "sanity_metrics.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[INFO] Sanity metrics saved to: {output_dir / 'sanity_metrics.json'}")


if __name__ == "__main__":
    main()
