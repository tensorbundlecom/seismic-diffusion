import argparse
import csv
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import obspy
import psutil
import torch
from scipy import signal

from ML.autoencoder.experiments.DDPMvsDDIM.core.metrics import lsd, mr_lsd, spec_corr
from ML.autoencoder.experiments.DDPMvsDDIM.core.diffusion_utils import (
    DiffusionSchedule,
    build_condition_tensor,
    sample_ddim,
    sample_ddpm,
)
from ML.autoencoder.experiments.DDPMvsDDIM.core.model_diffusion_resmlp import build_diffusion_denoiser
from ML.autoencoder.experiments.DDPMvsDDIM.core.model_legacy_cond_baseline import LegacyCondBaselineCVAE
from ML.autoencoder.experiments.DDPMvsDDIM.core.stft_dataset import SeismicSTFTDatasetWithMetadata


def parse_args():
    parser = argparse.ArgumentParser(description="Compare DDPM and DDIM samplers from the same trained latent denoiser.")
    parser.add_argument(
        "--stage1-checkpoint",
        default="ML/autoencoder/experiments/DDPMvsDDIM/checkpoints/stage1_eventwise_v1_best.pt",
    )
    parser.add_argument(
        "--diffusion-checkpoint",
        default="ML/autoencoder/experiments/DDPMvsDDIM/runs/diffusion/diffusion_eventwise_v1/checkpoints/best.pt",
    )
    parser.add_argument(
        "--test-cache",
        default="ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/test_latent_cache.pt",
    )
    parser.add_argument(
        "--stats-file",
        default="ML/autoencoder/experiments/DDPMvsDDIM/data_cache/latent_cache_eventwise_v1/latent_stats.pt",
    )
    parser.add_argument("--station-list-file", default="data/station_list_external_full.json")
    parser.add_argument("--data-dir", default="data/external_dataset/extracted/data/filtered_waveforms")
    parser.add_argument(
        "--event-file",
        default="data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt",
    )
    parser.add_argument("--channels", nargs="+", default=["HH"])
    parser.add_argument("--magnitude-col", default="ML")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--selection-mode", choices=["first", "evenly_spaced"], default="first")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-eta", type=float, default=0.0)
    parser.add_argument("--sampler-base-seed", type=int, default=1234)
    parser.add_argument("--resource-poll-interval-sec", type=float, default=0.5)
    parser.add_argument("--save-plots", choices=["none", "subset", "all"], default="subset")
    parser.add_argument("--plot-count", type=int, default=25)
    parser.add_argument(
        "--output-dir",
        default="ML/autoencoder/experiments/DDPMvsDDIM/results/sampler_comparison_eventwise_v1",
    )
    return parser.parse_args()


class ResourceMonitor:
    def __init__(self, gpu_index=None, poll_interval_sec=0.5):
        self.gpu_index = gpu_index
        self.poll_interval_sec = poll_interval_sec
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(interval=None)
        self._stop_event = threading.Event()
        self._thread = None
        self.samples = []

    def _read_gpu_stats(self):
        if self.gpu_index is None:
            return None
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return None

        for line in output.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                index = int(parts[0])
            except ValueError:
                continue
            if index != self.gpu_index:
                continue
            try:
                return {
                    "gpu_util_percent": float(parts[1]),
                    "gpu_memory_used_mb": float(parts[2]),
                    "gpu_memory_total_mb": float(parts[3]),
                }
            except ValueError:
                return None
        return None

    def _sample_once(self):
        sample = {
            "timestamp_sec": time.perf_counter(),
            "cpu_percent": float(self.process.cpu_percent(interval=None)),
            "rss_mb": float(self.process.memory_info().rss / (1024 ** 2)),
        }
        gpu_stats = self._read_gpu_stats()
        if gpu_stats is not None:
            sample.update(gpu_stats)
        self.samples.append(sample)

    def _run(self):
        while not self._stop_event.is_set():
            self._sample_once()
            self._stop_event.wait(self.poll_interval_sec)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2.0 * self.poll_interval_sec))
        self._sample_once()

    def summary(self):
        cpu_samples = [sample["cpu_percent"] for sample in self.samples]
        rss_samples = [sample["rss_mb"] for sample in self.samples]
        gpu_util_samples = [sample["gpu_util_percent"] for sample in self.samples if "gpu_util_percent" in sample]
        gpu_mem_samples = [sample["gpu_memory_used_mb"] for sample in self.samples if "gpu_memory_used_mb" in sample]
        gpu_mem_total_values = [sample["gpu_memory_total_mb"] for sample in self.samples if "gpu_memory_total_mb" in sample]

        summary = {
            "poll_interval_sec": float(self.poll_interval_sec),
            "num_samples": int(len(self.samples)),
            "cpu_percent_avg": float(statistics.fmean(cpu_samples)) if cpu_samples else None,
            "cpu_percent_peak": float(max(cpu_samples)) if cpu_samples else None,
            "rss_mb_avg": float(statistics.fmean(rss_samples)) if rss_samples else None,
            "rss_mb_peak_poll": float(max(rss_samples)) if rss_samples else None,
            "rss_mb_peak_process": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            "gpu_util_percent_avg": float(statistics.fmean(gpu_util_samples)) if gpu_util_samples else None,
            "gpu_util_percent_peak": float(max(gpu_util_samples)) if gpu_util_samples else None,
            "gpu_memory_used_mb_avg": float(statistics.fmean(gpu_mem_samples)) if gpu_mem_samples else None,
            "gpu_memory_used_mb_peak_poll": float(max(gpu_mem_samples)) if gpu_mem_samples else None,
            "gpu_memory_total_mb": float(gpu_mem_total_values[-1]) if gpu_mem_total_values else None,
        }
        return summary


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(sync_device, fn, *args, **kwargs):
    synchronize_device(sync_device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    synchronize_device(sync_device)
    elapsed_sec = time.perf_counter() - start
    return result, elapsed_sec


def reconstruct_signal(magnitude_spec, mag_min, mag_max, fs=100.0, nperseg=256, noverlap=192, nfft=256, n_iter=64):
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)

    phase = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    for _ in range(n_iter):
        stft_complex = spec * phase
        _, waveform = signal.istft(stft_complex, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary="zeros")
        _, _, new_zxx = signal.stft(waveform, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary="zeros")
        if new_zxx.shape != spec.shape:
            min_f = min(new_zxx.shape[0], spec.shape[0])
            min_t = min(new_zxx.shape[1], spec.shape[1])
            next_phase = np.zeros_like(spec, dtype=complex)
            next_phase[:min_f, :min_t] = np.exp(1j * np.angle(new_zxx[:min_f, :min_t]))
            phase = next_phase
        else:
            phase = np.exp(1j * np.angle(new_zxx))

    stft_complex = spec * phase
    _, waveform = signal.istft(stft_complex, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary="zeros")
    return waveform


def load_stage1_model(checkpoint_path, num_stations, device):
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
    return model


def load_diffusion_model(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    args = state["config"]
    cond_embedding_dim = int(state["cond_embedding_dim"])
    raw_condition_dim = int(state["raw_condition_dim"])
    latent_dim = int(state["latent_dim"])
    if args["cond_mode"] == "embedding_only":
        cond_dim = cond_embedding_dim
    elif args["cond_mode"] == "raw_only":
        cond_dim = raw_condition_dim
    else:
        cond_dim = cond_embedding_dim + raw_condition_dim

    model = build_diffusion_denoiser(
        model_type=args.get("model_type", "resmlp"),
        latent_dim=latent_dim,
        cond_dim=cond_dim,
        hidden_dim=args["hidden_dim"],
        depth=args["depth"],
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    schedule = DiffusionSchedule(args["timesteps"], beta_schedule=args["beta_schedule"]).to(device)
    return model, schedule, args, latent_dim


def load_original_waveform(file_path):
    stream = obspy.read(file_path)
    stream.merge(fill_value=0)
    z_stream = stream.select(component="Z")
    trace = z_stream[0] if len(z_stream) > 0 else stream[0]
    return trace.data.astype(np.float32), float(trace.stats.sampling_rate)


def unnormalize_latent(z_norm, stats):
    return z_norm * (stats["z_std"].to(z_norm.device) + 1e-8) + stats["z_mean"].to(z_norm.device)


def choose_plot_indices(num_samples, mode, plot_count):
    if mode == "none" or num_samples <= 0:
        return set()
    if mode == "all" or plot_count >= num_samples:
        return set(range(num_samples))
    count = max(1, min(plot_count, num_samples))
    return {int(round(x)) for x in np.linspace(0, num_samples - 1, num=count)}


def choose_sample_indices(total_count, max_samples, selection_mode):
    sample_count = min(max_samples, total_count)
    if sample_count <= 0:
        return []
    if selection_mode == "first":
        return list(range(sample_count))
    if selection_mode == "evenly_spaced":
        return [int(round(x)) for x in np.linspace(0, total_count - 1, num=sample_count)]
    raise ValueError(f"Unsupported selection mode: {selection_mode}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_index = torch.cuda.current_device() if device.type == "cuda" else None
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
    cache = torch.load(args.test_cache, map_location="cpu")
    stats = torch.load(args.stats_file, map_location="cpu")
    stage1 = load_stage1_model(args.stage1_checkpoint, len(station_list), device)
    diffusion_model, schedule, train_args, latent_dim = load_diffusion_model(args.diffusion_checkpoint, device)

    results = []
    selected_indices = choose_sample_indices(cache["z_mu"].size(0), args.max_samples, args.selection_mode)
    num_samples = len(selected_indices)
    plot_indices = choose_plot_indices(num_samples, args.save_plots, args.plot_count)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    resource_monitor = ResourceMonitor(gpu_index=gpu_index, poll_interval_sec=args.resource_poll_interval_sec)
    resource_monitor.start()
    evaluation_start = time.perf_counter()
    oracle_decode_times_sec = []
    ddpm_sampling_times_sec = []
    ddpm_total_times_sec = []
    ddim_sampling_times_sec = []
    ddim_total_times_sec = []
    if plot_indices:
        specs_dir = output_dir / "specs"
        wav_dir = output_dir / "waveforms"
        specs_dir.mkdir(parents=True, exist_ok=True)
        wav_dir.mkdir(parents=True, exist_ok=True)

    for local_idx, cache_idx in enumerate(selected_indices):
        dataset_index = int(cache["meta"]["dataset_index"][cache_idx])
        spec, magnitude, location, station_idx, metadata = dataset[dataset_index]
        mag_in = magnitude.unsqueeze(0).to(device)
        loc_in = location.unsqueeze(0).to(device)
        sta_in = station_idx.unsqueeze(0).to(device)
        cond_embedding = stage1.build_condition_embedding(mag_in, loc_in, sta_in)
        raw_condition = torch.cat([stage1._normalize_magnitude(mag_in).unsqueeze(1), torch.clamp(loc_in, 0.0, 1.0)], dim=1)
        cond = build_condition_tensor(train_args["cond_mode"], cond_embedding=cond_embedding, raw_condition=raw_condition)

        with torch.no_grad():
            oracle_z = cache["z_mu"][cache_idx : cache_idx + 1].to(device)
            oracle_spec, oracle_decode_time_sec = timed_call(
                device,
                stage1.decode,
                oracle_z,
                cond_embedding,
                original_size=spec.shape[1:],
            )
            oracle_decode_times_sec.append(oracle_decode_time_sec)
            sample_seed = args.sampler_base_seed + dataset_index
            init_generator = torch.Generator(device=device).manual_seed(sample_seed)
            ddpm_generator = torch.Generator(device=device).manual_seed(sample_seed)
            ddim_generator = torch.Generator(device=device).manual_seed(sample_seed)
            initial_noise = torch.randn(1, latent_dim, generator=init_generator, device=device)
            ddim_z_norm, ddim_sampling_time_sec = timed_call(
                device,
                sample_ddim,
                diffusion_model,
                schedule,
                cond,
                latent_dim=latent_dim,
                device=device,
                num_inference_steps=args.ddim_steps,
                eta=args.ddim_eta,
                initial_noise=initial_noise,
                generator=ddim_generator,
                prediction_target=train_args.get("prediction_target", "epsilon"),
            )
            ddim_sampling_times_sec.append(ddim_sampling_time_sec)
            ddpm_z_norm, ddpm_sampling_time_sec = timed_call(
                device,
                sample_ddpm,
                diffusion_model,
                schedule,
                cond,
                latent_dim=latent_dim,
                device=device,
                initial_noise=initial_noise,
                generator=ddpm_generator,
                prediction_target=train_args.get("prediction_target", "epsilon"),
            )
            ddpm_sampling_times_sec.append(ddpm_sampling_time_sec)
            ddpm_z = unnormalize_latent(ddpm_z_norm, stats)
            ddim_z = unnormalize_latent(ddim_z_norm, stats)
            ddpm_spec, ddpm_decode_time_sec = timed_call(
                device,
                stage1.decode,
                ddpm_z,
                cond_embedding,
                original_size=spec.shape[1:],
            )
            ddim_spec, ddim_decode_time_sec = timed_call(
                device,
                stage1.decode,
                ddim_z,
                cond_embedding,
                original_size=spec.shape[1:],
            )
            ddpm_total_times_sec.append(ddpm_sampling_time_sec + ddpm_decode_time_sec)
            ddim_total_times_sec.append(ddim_sampling_time_sec + ddim_decode_time_sec)

        original_spec = spec[2].cpu().numpy()
        oracle_spec_z = oracle_spec[0, 2].cpu().numpy()
        ddpm_spec_z = ddpm_spec[0, 2].cpu().numpy()
        ddim_spec_z = ddim_spec[0, 2].cpu().numpy()

        metrics = {
            "cache_index": int(cache_idx),
            "dataset_index": dataset_index,
            "file_name": metadata["file_name"],
            "station_name": metadata["station_name"],
            "event_id": metadata["event_id"],
            "oracle_spec_corr": spec_corr(original_spec, oracle_spec_z),
            "oracle_lsd": lsd(original_spec, oracle_spec_z),
            "oracle_mr_lsd": mr_lsd(original_spec, oracle_spec_z),
            "ddpm_spec_corr": spec_corr(original_spec, ddpm_spec_z),
            "ddpm_lsd": lsd(original_spec, ddpm_spec_z),
            "ddpm_mr_lsd": mr_lsd(original_spec, ddpm_spec_z),
            "ddim_spec_corr": spec_corr(original_spec, ddim_spec_z),
            "ddim_lsd": lsd(original_spec, ddim_spec_z),
            "ddim_mr_lsd": mr_lsd(original_spec, ddim_spec_z),
            "oracle_decode_time_ms": float(oracle_decode_time_sec * 1000.0),
            "ddpm_sampling_time_ms": float(ddpm_sampling_time_sec * 1000.0),
            "ddpm_total_time_ms": float((ddpm_sampling_time_sec + ddpm_decode_time_sec) * 1000.0),
            "ddim_sampling_time_ms": float(ddim_sampling_time_sec * 1000.0),
            "ddim_total_time_ms": float((ddim_sampling_time_sec + ddim_decode_time_sec) * 1000.0),
        }
        results.append(metrics)

        if local_idx in plot_indices:
            title_root = f"{metadata['event_id']} @ {metadata['station_name']}"
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            axes[0].imshow(original_spec, aspect="auto", origin="lower", cmap="viridis")
            axes[0].set_title("Original")
            axes[1].imshow(oracle_spec_z, aspect="auto", origin="lower", cmap="viridis")
            axes[1].set_title(f"Oracle mu\ncorr={metrics['oracle_spec_corr']:.4f}")
            axes[2].imshow(ddpm_spec_z, aspect="auto", origin="lower", cmap="viridis")
            axes[2].set_title(f"DDPM\ncorr={metrics['ddpm_spec_corr']:.4f}")
            axes[3].imshow(ddim_spec_z, aspect="auto", origin="lower", cmap="viridis")
            axes[3].set_title(f"DDIM\ncorr={metrics['ddim_spec_corr']:.4f}")
            fig.suptitle(title_root)
            fig.tight_layout()
            fig.savefig(specs_dir / f"{metadata['event_id']}_{metadata['station_name']}_comparison.png", dpi=150)
            plt.close(fig)

            mag_min = float(metadata.get("mag_min", 0.0))
            mag_max = float(metadata.get("mag_max", 1.0))
            original_waveform, fs = load_original_waveform(metadata["file_path"])
            oracle_waveform = reconstruct_signal(oracle_spec_z, mag_min=mag_min, mag_max=mag_max, fs=fs)
            ddpm_waveform = reconstruct_signal(ddpm_spec_z, mag_min=mag_min, mag_max=mag_max, fs=fs)
            ddim_waveform = reconstruct_signal(ddim_spec_z, mag_min=mag_min, mag_max=mag_max, fs=fs)

            min_len = min(len(original_waveform), len(oracle_waveform), len(ddpm_waveform), len(ddim_waveform))
            time_axis = np.arange(min_len) / fs
            fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
            axes[0].plot(time_axis, original_waveform[:min_len], linewidth=0.8)
            axes[0].set_title("Original")
            axes[1].plot(time_axis, oracle_waveform[:min_len], linewidth=0.8)
            axes[1].set_title("Oracle mu")
            axes[2].plot(time_axis, ddpm_waveform[:min_len], linewidth=0.8)
            axes[2].set_title("DDPM")
            axes[3].plot(time_axis, ddim_waveform[:min_len], linewidth=0.8)
            axes[3].set_title("DDIM")
            axes[3].set_xlabel("Time (s)")
            fig.suptitle(title_root)
            fig.tight_layout()
            fig.savefig(wav_dir / f"{metadata['event_id']}_{metadata['station_name']}_comparison.png", dpi=150)
            plt.close(fig)

        print(
            f"[INFO] Compared samplers for {metadata['file_name']} | "
            f"oracle={metrics['oracle_spec_corr']:.4f} ddpm={metrics['ddpm_spec_corr']:.4f} ddim={metrics['ddim_spec_corr']:.4f}"
        )

    synchronize_device(device)
    evaluation_wall_time_sec = time.perf_counter() - evaluation_start
    resource_monitor.stop()
    resource_summary = resource_monitor.summary()
    if device.type == "cuda":
        resource_summary["torch_peak_allocated_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        resource_summary["torch_peak_reserved_mb"] = float(torch.cuda.max_memory_reserved(device) / (1024 ** 2))
    else:
        resource_summary["torch_peak_allocated_mb"] = None
        resource_summary["torch_peak_reserved_mb"] = None

    summary = {
        "diffusion_checkpoint": args.diffusion_checkpoint,
        "stage1_checkpoint": args.stage1_checkpoint,
        "num_samples": len(results),
        "ddim_steps": int(args.ddim_steps),
        "ddim_eta": float(args.ddim_eta),
        "prediction_target": train_args.get("prediction_target", "epsilon"),
        "model_type": train_args.get("model_type", "resmlp"),
        "timesteps": int(train_args.get("timesteps", schedule.timesteps)),
        "avg_oracle_spec_corr": float(np.mean([item["oracle_spec_corr"] for item in results])),
        "avg_oracle_lsd": float(np.mean([item["oracle_lsd"] for item in results])),
        "avg_oracle_mr_lsd": float(np.mean([item["oracle_mr_lsd"] for item in results])),
        "avg_ddpm_spec_corr": float(np.mean([item["ddpm_spec_corr"] for item in results])),
        "avg_ddpm_lsd": float(np.mean([item["ddpm_lsd"] for item in results])),
        "avg_ddpm_mr_lsd": float(np.mean([item["ddpm_mr_lsd"] for item in results])),
        "avg_ddim_spec_corr": float(np.mean([item["ddim_spec_corr"] for item in results])),
        "avg_ddim_lsd": float(np.mean([item["ddim_lsd"] for item in results])),
        "avg_ddim_mr_lsd": float(np.mean([item["ddim_mr_lsd"] for item in results])),
        "save_plots": args.save_plots,
        "plot_count_requested": int(args.plot_count),
        "plot_count_written": int(len(plot_indices)),
        "runtime": {
            "evaluation_wall_time_sec": float(evaluation_wall_time_sec),
            "evaluation_wall_time_min": float(evaluation_wall_time_sec / 60.0),
            "samples_per_sec": float(len(results) / evaluation_wall_time_sec) if evaluation_wall_time_sec > 0 else None,
            "avg_oracle_decode_time_ms": float(statistics.fmean(oracle_decode_times_sec) * 1000.0) if oracle_decode_times_sec else None,
            "avg_ddpm_sampling_time_ms": float(statistics.fmean(ddpm_sampling_times_sec) * 1000.0) if ddpm_sampling_times_sec else None,
            "avg_ddpm_total_time_ms": float(statistics.fmean(ddpm_total_times_sec) * 1000.0) if ddpm_total_times_sec else None,
            "avg_ddim_sampling_time_ms": float(statistics.fmean(ddim_sampling_times_sec) * 1000.0) if ddim_sampling_times_sec else None,
            "avg_ddim_total_time_ms": float(statistics.fmean(ddim_total_times_sec) * 1000.0) if ddim_total_times_sec else None,
        },
        "resources": resource_summary,
        "results": results,
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    with open(output_dir / "per_sample_metrics.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[INFO] Sampler comparison summary saved to: {output_dir / 'summary.json'}")
    print(
        "[INFO] Runtime | "
        f"wall_time_min={summary['runtime']['evaluation_wall_time_min']:.3f} "
        f"samples_per_sec={summary['runtime']['samples_per_sec']:.3f} "
        f"ddpm_total_ms={summary['runtime']['avg_ddpm_total_time_ms']:.3f} "
        f"ddim_total_ms={summary['runtime']['avg_ddim_total_time_ms']:.3f}"
    )
    print(
        "[INFO] Resources | "
        f"peak_rss_mb={summary['resources']['rss_mb_peak_process']:.2f} "
        f"peak_gpu_alloc_mb={summary['resources']['torch_peak_allocated_mb'] if summary['resources']['torch_peak_allocated_mb'] is not None else 'NA'} "
        f"gpu_util_avg={summary['resources']['gpu_util_percent_avg'] if summary['resources']['gpu_util_percent_avg'] is not None else 'NA'}"
    )


if __name__ == "__main__":
    main()
