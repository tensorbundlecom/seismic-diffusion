import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import obspy
import torch
from scipy import signal
from skimage.metrics import structural_similarity as ssim

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE
from ML.autoencoder.experiments.LegacyCondBaseline.core.model_wbaseline import WBaselineCVAE
from ML.autoencoder.experiments.LegacyCondFullCov.core.model_wfullcov import WFullCovCVAE
from ML.autoencoder.experiments.LegacyCondFlow.core.model_wflow import WFlowCVAE


def reconstruct_signal(magnitude_spec, mag_min=0.0, mag_max=1.0, nperseg=256, noverlap=192, nfft=256, fs=100.0, n_iter=64):
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)

    phase = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    for _ in range(n_iter):
        stft_complex = spec * phase
        _, waveform = signal.istft(stft_complex, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary='zeros')
        _, _, new_zxx = signal.stft(waveform, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary='zeros')
        if new_zxx.shape != spec.shape:
            min_f = min(new_zxx.shape[0], spec.shape[0])
            min_t = min(new_zxx.shape[1], spec.shape[1])
            next_phase = np.zeros_like(spec, dtype=complex)
            next_phase[:min_f, :min_t] = np.exp(1j * np.angle(new_zxx[:min_f, :min_t]))
            phase = next_phase
        else:
            phase = np.exp(1j * np.angle(new_zxx))

    stft_complex = spec * phase
    _, waveform = signal.istft(stft_complex, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary='zeros')
    return waveform


def sample_condition_only(model, magnitude, location, station_idx, device):
    if hasattr(model, 'sample'):
        out = model.sample(magnitude.size(0), magnitude, location, station_idx, device=device)
        if isinstance(out, tuple):
            out = out[0]
        return out

    if hasattr(model, 'decoder') and hasattr(model, 'latent_dim'):
        z = torch.randn(magnitude.size(0), int(model.latent_dim), device=device)
        return model.decoder(z, magnitude, location, station_idx)

    raise AttributeError(f'Model {model.__class__.__name__} does not support condition-only sampling.')


def extract_spec_channel(pred, channel_idx, target_hw):
    if pred.shape[2:] != target_hw:
        pred = torch.nn.functional.interpolate(pred, size=target_hw, mode='bilinear', align_corners=False)
    return pred[0, channel_idx].detach().cpu().numpy()


def safe_ssim(a, b):
    a_n = (a - np.min(a)) / (np.max(a) - np.min(a) + 1e-8)
    b_n = (b - np.min(b)) / (np.max(b) - np.min(b) + 1e-8)
    return float(ssim(a_n, b_n, data_range=1.0))


def safe_xcorr(a, b):
    x1 = (a - np.mean(a)) / (np.std(a) + 1e-8)
    x2 = (b - np.mean(b)) / (np.std(b) + 1e-8)
    m = min(len(x1), len(x2))
    xc = np.correlate(x1[:m], x2[:m], mode='full')
    return float(np.max(np.abs(xc)) / max(1, m))


def parse_args():
    p = argparse.ArgumentParser(description='Visualize post-training custom OOD for all models (STFT + waveform).')
    p.add_argument('--ood_data_dir', default='data/ood_waveforms/post_training_custom/filtered')
    p.add_argument('--ood_catalog', default='data/events/ood_catalog_post_training.txt')
    p.add_argument('--station_list_file', default='data/station_list_external_full.json')
    p.add_argument('--station_subset_file', default='data/station_list_post_custom.json')
    p.add_argument(
        '--inference_mode',
        choices=['reconstruct', 'condition_only'],
        default='condition_only',
        help='reconstruct: model(x,c), condition_only: model.sample(c)',
    )
    p.add_argument('--max_samples', type=int, default=-1, help='-1 for all.')
    p.add_argument('--baseline_checkpoint', default='ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt')
    p.add_argument('--fullcov_checkpoint', default='ML/autoencoder/experiments/FullCovariance/checkpoints/full_cov_external_best.pt')
    p.add_argument('--flow_checkpoint', default='ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_external_best.pt')
    p.add_argument('--legacy_cond_baseline_checkpoint', default='ML/autoencoder/experiments/LegacyCondBaseline/checkpoints/wbaseline_external_best.pt')
    p.add_argument('--legacy_cond_fullcov_checkpoint', default='ML/autoencoder/experiments/LegacyCondFullCov/checkpoints/wfullcov_external_best.pt')
    p.add_argument('--legacy_cond_flow_checkpoint', default='ML/autoencoder/experiments/LegacyCondFlow/checkpoints/wflow_external_best.pt')
    p.add_argument('--mrloss_checkpoint', default='ML/autoencoder/experiments/CVAE_MRLoss/checkpoints/cvae_mrloss_pilot_b0p1_l0p5_fix_20260211_155549_best.pt')
    p.add_argument(
        '--output_dir',
        default='ML/autoencoder/experiments/General/visualizations/post_training_custom_ood_all_models_condition_only',
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_fs = 100.0

    os.makedirs(args.output_dir, exist_ok=True)
    specs_dir = os.path.join(args.output_dir, 'specs')
    wav_dir = os.path.join(args.output_dir, 'waveforms')
    os.makedirs(specs_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)

    with open(args.station_list_file, 'r') as f:
        station_list = json.load(f)
    station_subset = None
    if args.station_subset_file and os.path.exists(args.station_subset_file):
        with open(args.station_subset_file, 'r') as f:
            station_subset = set(json.load(f))

    num_stations = len(station_list)

    baseline = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    baseline.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device)['model_state_dict'])
    baseline.eval()

    fullcov = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    fullcov.load_state_dict(torch.load(args.fullcov_checkpoint, map_location=device)['model_state_dict'])
    fullcov.eval()

    flow = FlowCVAE(in_channels=3, latent_dim=128, num_stations=num_stations, flow_layers=8).to(device)
    flow.load_state_dict(torch.load(args.flow_checkpoint, map_location=device)['model_state_dict'])
    flow.eval()

    legacy_cond_baseline = WBaselineCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    legacy_cond_baseline.load_state_dict(torch.load(args.legacy_cond_baseline_checkpoint, map_location=device)['model_state_dict'])
    legacy_cond_baseline.eval()

    legacy_cond_fullcov = WFullCovCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    legacy_cond_fullcov.load_state_dict(torch.load(args.legacy_cond_fullcov_checkpoint, map_location=device)['model_state_dict'])
    legacy_cond_fullcov.eval()

    legacy_cond_flow = WFlowCVAE(in_channels=3, latent_dim=128, num_stations=num_stations, flow_layers=8).to(device)
    legacy_cond_flow.load_state_dict(torch.load(args.legacy_cond_flow_checkpoint, map_location=device)['model_state_dict'])
    legacy_cond_flow.eval()

    mr_state = torch.load(args.mrloss_checkpoint, map_location=device)
    mr_cfg = mr_state.get('config', {})
    mrloss_model = ConditionalVariationalAutoencoder(
        in_channels=3,
        latent_dim=mr_cfg.get('latent_dim', 128),
        num_stations=mr_cfg.get('num_stations', num_stations),
    ).to(device)
    mrloss_model.load_state_dict(mr_state['model_state_dict'])
    mrloss_model.eval()

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.ood_data_dir,
        event_file=args.ood_catalog,
        channels=['HH'],
        magnitude_col='xM',
        station_list=station_list,
    )

    processed = 0
    with torch.no_grad():
        for i in range(len(dataset)):
            if args.max_samples > 0 and processed >= args.max_samples:
                break

            spec, mag, loc, station_idx, meta = dataset[i]
            if 'error' in meta:
                continue

            file_path = meta['file_path']
            file_name = os.path.basename(file_path)
            parts = file_name.split('_')
            if len(parts) >= 4:
                event_id = '_'.join(parts[:3])
                station_name = parts[3]
            else:
                event_id = meta.get('event_id', f'event_{i:04d}')
                station_name = station_list[station_idx.item()]

            if station_subset is not None and station_name not in station_subset:
                continue

            st_gt = obspy.read(file_path)
            st_gt.resample(target_fs)
            tr_gt = st_gt.select(component='Z')[0] if st_gt.select(component='Z') else st_gt[0]
            gt_wav = tr_gt.data.astype(np.float32)

            target_len = 7300
            if len(gt_wav) > target_len:
                gt_wav = gt_wav[:target_len]
            elif len(gt_wav) < target_len:
                gt_wav = np.pad(gt_wav, (0, target_len - len(gt_wav)))

            _, _, zxx = signal.stft(gt_wav, fs=target_fs, nperseg=256, noverlap=192, nfft=256, boundary='zeros')
            mag_spec = np.log1p(np.abs(zxx))
            mag_min, mag_max = mag_spec.min(), mag_spec.max()
            if mag_max > mag_min:
                mag_spec = (mag_spec - mag_min) / (mag_max - mag_min + 1e-8)

            spec_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).to(device).float().repeat(1, 3, 1, 1)
            if spec_in.shape[2:] != (129, 111):
                spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode='bilinear', align_corners=False)

            mag_in = mag.unsqueeze(0).to(device)
            loc_in = loc.unsqueeze(0).to(device)
            sta_in = station_idx.unsqueeze(0).to(device)

            if args.inference_mode == 'reconstruct':
                r_base, _, _ = baseline(spec_in, mag_in, loc_in, sta_in)
                r_fc, _, _ = fullcov(spec_in, mag_in, loc_in, sta_in)
                r_flow, _, _, _, _ = flow(spec_in, mag_in, loc_in, sta_in)
                r_lcb, _, _ = legacy_cond_baseline(spec_in, mag_in, loc_in, sta_in)
                r_lcf, _, _ = legacy_cond_fullcov(spec_in, mag_in, loc_in, sta_in)
                r_lcflow, _, _, _, _ = legacy_cond_flow(spec_in, mag_in, loc_in, sta_in)
                r_mr, _, _ = mrloss_model(spec_in, mag_in, loc_in, sta_in)
            else:
                r_base = sample_condition_only(baseline, mag_in, loc_in, sta_in, device)
                r_fc = sample_condition_only(fullcov, mag_in, loc_in, sta_in, device)
                r_flow = sample_condition_only(flow, mag_in, loc_in, sta_in, device)
                r_lcb = sample_condition_only(legacy_cond_baseline, mag_in, loc_in, sta_in, device)
                r_lcf = sample_condition_only(legacy_cond_fullcov, mag_in, loc_in, sta_in, device)
                r_lcflow = sample_condition_only(legacy_cond_flow, mag_in, loc_in, sta_in, device)
                r_mr = sample_condition_only(mrloss_model, mag_in, loc_in, sta_in, device)

            target_hw = spec_in.shape[2:]
            gt_spec = spec_in[0, 2].detach().cpu().numpy()
            specs = {
                'Baseline': extract_spec_channel(r_base, 2, target_hw),
                'FullCov': extract_spec_channel(r_fc, 2, target_hw),
                'Flow': extract_spec_channel(r_flow, 2, target_hw),
                'LegacyCondBaseline': extract_spec_channel(r_lcb, 2, target_hw),
                'LegacyCondFullCov': extract_spec_channel(r_lcf, 2, target_hw),
                'LegacyCondFlow': extract_spec_channel(r_lcflow, 2, target_hw),
                'CVAE_MRLoss': extract_spec_channel(r_mr, 2, target_hw),
            }

            waves = {name: reconstruct_signal(sp, mag_min, mag_max, fs=target_fs) for name, sp in specs.items()}
            min_len = min([len(gt_wav)] + [len(w) for w in waves.values()])
            gt_w = gt_wav[:min_len]
            time = np.arange(min_len) / target_fs

            # STFT grid
            fig, axes = plt.subplots(2, 4, figsize=(18, 9))
            axes = axes.flatten()
            axes[0].imshow(gt_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[0].set_title(f'GT | {event_id} @ {station_name}')
            order = ['Baseline', 'FullCov', 'Flow', 'LegacyCondBaseline', 'LegacyCondFullCov', 'LegacyCondFlow', 'CVAE_MRLoss']
            for j, name in enumerate(order, start=1):
                sc = safe_ssim(gt_spec, specs[name])
                axes[j].imshow(specs[name], aspect='auto', origin='lower', cmap='viridis')
                axes[j].set_title(f'{name}\nSSIM={sc:.3f}')
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(specs_dir, f'{event_id}_{station_name}_specs.png'), dpi=150, bbox_inches='tight')
            plt.close()

            # Waveform stack
            fig, axes = plt.subplots(8, 1, figsize=(15, 14), sharex=True)
            axes[0].plot(time, gt_w, 'k-', linewidth=0.8)
            axes[0].set_title(f'GT | {event_id} @ {station_name} | mode={args.inference_mode}')
            axes[0].grid(True, alpha=0.3)
            for j, name in enumerate(order, start=1):
                pw = waves[name][:min_len]
                xc = safe_xcorr(gt_w, pw)
                axes[j].plot(time, pw, linewidth=0.8)
                axes[j].set_title(f'{name} | XCorr={xc:.3f}')
                axes[j].grid(True, alpha=0.3)
            axes[-1].set_xlabel('Time (s)')
            plt.tight_layout()
            plt.savefig(os.path.join(wav_dir, f'{event_id}_{station_name}_waveforms.png'), dpi=150, bbox_inches='tight')
            plt.close()

            processed += 1
            if processed % 10 == 0:
                print(f'[INFO] Visualized {processed} samples')

    print(f'[INFO] Completed visualization. samples={processed}')
    print(f'[INFO] STFT dir: {specs_dir}')
    print(f'[INFO] Waveform dir: {wav_dir}')


if __name__ == '__main__':
    main()
