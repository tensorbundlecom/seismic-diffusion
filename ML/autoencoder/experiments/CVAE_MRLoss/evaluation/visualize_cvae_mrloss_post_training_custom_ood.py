import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import obspy
import torch
from fastdtw import fastdtw
from scipy import signal
from scipy.signal import hilbert
from scipy.spatial.distance import euclidean
from skimage.metrics import structural_similarity as ssim

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE


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


def calculate_metrics(target_wav, pred_wav, target_spec, pred_spec, fs=100.0):
    out = {}
    s1 = (target_spec - np.min(target_spec)) / (np.max(target_spec) - np.min(target_spec) + 1e-8)
    s2 = (pred_spec - np.min(pred_spec)) / (np.max(pred_spec) - np.min(pred_spec) + 1e-8)
    out['ssim'] = float(ssim(s1, s2, data_range=1.0))
    out['lsd'] = float(np.sqrt(np.mean((np.log(target_spec + 1e-8) - np.log(pred_spec + 1e-8)) ** 2)))

    a_target = (np.pi / (2 * 9.81)) * np.trapz(target_wav ** 2, dx=1 / fs)
    a_pred = (np.pi / (2 * 9.81)) * np.trapz(pred_wav ** 2, dx=1 / fs)
    out['arias_err'] = float(np.abs(a_target - a_pred) / (np.abs(a_target) + 1e-8))

    env1 = np.abs(hilbert(target_wav))
    env2 = np.abs(hilbert(pred_wav))
    min_len = min(len(env1), len(env2))
    out['env_corr'] = float(np.corrcoef(env1[:min_len], env2[:min_len])[0, 1])

    factor = max(1, len(target_wav) // 500)
    s_target = target_wav[::factor].reshape(-1, 1)
    s_pred = pred_wav[::factor].reshape(-1, 1)
    dtw_dist, _ = fastdtw(s_target, s_pred, dist=euclidean)
    out['dtw'] = float(dtw_dist / len(s_target))

    x1 = (target_wav - np.mean(target_wav)) / (np.std(target_wav) + 1e-8)
    x2 = (pred_wav - np.mean(pred_wav)) / (np.std(pred_wav) + 1e-8)
    min_len = min(len(x1), len(x2))
    xcorr = np.correlate(x1[:min_len], x2[:min_len], mode='full')
    out['xcorr'] = float(np.max(np.abs(xcorr)) / len(x1[:min_len]))

    return out


def get_fas(sig, fs=100.0):
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    fas = np.abs(np.fft.rfft(sig))
    return freqs, fas


def main():
    parser = argparse.ArgumentParser(description='Visualize CVAE_MRLoss on custom post-training OOD.')
    parser.add_argument('--ood_data_dir', default='data/ood_waveforms/post_training_custom/filtered', help='OOD data root.')
    parser.add_argument('--ood_catalog', default='data/events/ood_catalog_post_training.txt', help='OOD catalog path.')
    parser.add_argument('--station_list_file', default='data/station_list_external_full.json', help='Station list JSON.')
    parser.add_argument('--station_subset_file', default='data/station_list_post_custom.json', help='Subset stations JSON.')
    parser.add_argument(
        '--output_dir',
        default='ML/autoencoder/experiments/CVAE_MRLoss/visualizations/post_training_custom_ood_evaluation',
        help='Output directory for visuals.',
    )
    parser.add_argument(
        '--output_metrics',
        default='ML/autoencoder/experiments/CVAE_MRLoss/results/post_training_custom_ood_metrics.json',
        help='Output JSON metrics path.',
    )
    parser.add_argument(
        '--mrloss_checkpoint',
        default='ML/autoencoder/experiments/CVAE_MRLoss/checkpoints/cvae_mrloss_pilot_b0p1_l0p5_fix_20260211_155549_best.pt',
        help='MRLoss checkpoint path.',
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/specs', exist_ok=True)
    os.makedirs(f'{output_dir}/specs/fas', exist_ok=True)
    os.makedirs(f'{output_dir}/waveforms', exist_ok=True)
    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)

    baseline_chk = 'ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt'
    full_cov_chk = 'ML/autoencoder/experiments/FullCovariance/checkpoints/full_cov_external_best.pt'
    flow_chk = 'ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_external_best.pt'

    with open(args.station_list_file, 'r') as f:
        station_list = json.load(f)
    station_subset = None
    if args.station_subset_file:
        with open(args.station_subset_file, 'r') as f:
            station_subset = set(json.load(f))

    num_stations = len(station_list)

    baseline = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    baseline.load_state_dict(torch.load(baseline_chk, map_location=device)['model_state_dict'])
    baseline.eval()

    full_cov = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    full_cov.load_state_dict(torch.load(full_cov_chk, map_location=device)['model_state_dict'])
    full_cov.eval()

    flow = FlowCVAE(in_channels=3, latent_dim=128, num_stations=num_stations, flow_layers=8).to(device)
    flow.load_state_dict(torch.load(flow_chk, map_location=device)['model_state_dict'])
    flow.eval()

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

    target_fs = 100.0
    metrics = {k: [] for k in ['ssim', 'lsd', 'arias_err', 'env_corr', 'dtw', 'xcorr']}
    with torch.no_grad():
        for i in range(len(dataset)):
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
            mag_min = mag_spec.min()
            mag_max = mag_spec.max()
            if mag_max > mag_min:
                mag_spec = (mag_spec - mag_min) / (mag_max - mag_min + 1e-8)

            spec_in = torch.from_numpy(mag_spec).unsqueeze(0).unsqueeze(0).float().to(device).repeat(1, 3, 1, 1)
            if spec_in.shape[2:] != (129, 111):
                spec_in = torch.nn.functional.interpolate(spec_in, size=(129, 111), mode='bilinear', align_corners=False)

            mag_in = mag.unsqueeze(0).to(device)
            loc_in = loc.unsqueeze(0).to(device)
            sta_in = station_idx.unsqueeze(0).to(device)

            r_base, _, _ = baseline(spec_in, mag_in, loc_in, sta_in)
            r_fc, _, _ = full_cov(spec_in, mag_in, loc_in, sta_in)
            r_flow, _, _, _, _ = flow(spec_in, mag_in, loc_in, sta_in)
            r_mr, _, _ = mrloss_model(spec_in, mag_in, loc_in, sta_in)

            orig_spec = spec_in[0, 2].cpu().numpy()
            base_spec = r_base[0, 2].cpu().numpy()
            fc_spec = r_fc[0, 2].cpu().numpy()
            flow_spec = r_flow[0, 2].cpu().numpy()
            mr_spec = r_mr[0, 2].cpu().numpy()

            wav_base = reconstruct_signal(base_spec, mag_min, mag_max, fs=target_fs)
            wav_fc = reconstruct_signal(fc_spec, mag_min, mag_max, fs=target_fs)
            wav_flow = reconstruct_signal(flow_spec, mag_min, mag_max, fs=target_fs)
            wav_mr = reconstruct_signal(mr_spec, mag_min, mag_max, fs=target_fs)

            min_len = min(len(gt_wav), len(wav_base), len(wav_fc), len(wav_flow), len(wav_mr))
            m_mr = calculate_metrics(gt_wav[:min_len], wav_mr[:min_len], orig_spec, mr_spec, fs=target_fs)
            for k in metrics:
                metrics[k].append(m_mr[k])

            fig, axes = plt.subplots(2, 3, figsize=(16, 8))
            axes[0, 0].imshow(orig_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[0, 0].set_title(f'Original - {event_id} @ {station_name}')
            axes[0, 1].imshow(base_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[0, 1].set_title('Baseline')
            axes[0, 2].imshow(fc_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[0, 2].set_title('FullCov')
            axes[1, 0].imshow(flow_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[1, 0].set_title('Flow')
            axes[1, 1].imshow(mr_spec, aspect='auto', origin='lower', cmap='viridis')
            axes[1, 1].set_title('MRLoss')
            axes[1, 2].axis('off')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/specs/{event_id}_{station_name}_specs.png', dpi=150, bbox_inches='tight')
            plt.close()

            f0, fas0 = get_fas(gt_wav[:min_len], fs=target_fs)
            fb, fasb = get_fas(wav_base[:min_len], fs=target_fs)
            ffc, fasfc = get_fas(wav_fc[:min_len], fs=target_fs)
            ffl, fasfl = get_fas(wav_flow[:min_len], fs=target_fs)
            fmr, fasmr = get_fas(wav_mr[:min_len], fs=target_fs)

            plt.figure(figsize=(10, 6))
            plt.loglog(f0, fas0, 'k', alpha=0.5, label='Original', linewidth=1.2)
            plt.loglog(fb, fasb, 'b', alpha=0.7, label='Baseline', linewidth=0.8)
            plt.loglog(ffc, fasfc, 'g', alpha=0.7, label='FullCov', linewidth=0.8)
            plt.loglog(ffl, fasfl, 'r', alpha=0.7, label='Flow', linewidth=0.8)
            plt.loglog(fmr, fasmr, 'm', alpha=0.8, label='MRLoss', linewidth=0.9)
            plt.grid(True, which='both', ls='-', alpha=0.3)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title(f'FAS - {event_id} @ {station_name}')
            plt.legend()
            plt.savefig(f'{output_dir}/specs/fas/{event_id}_{station_name}_fas.png', dpi=150, bbox_inches='tight')
            plt.close()

            time = np.arange(min_len) / target_fs
            fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
            axes[0].plot(time, gt_wav[:min_len], 'k-', linewidth=0.8, label='Ground Truth')
            axes[0].set_title(f'{event_id} @ {station_name} - HH')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(time, wav_base[:min_len], 'b-', linewidth=0.8, label='Baseline')
            axes[1].legend(loc='upper right')
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(time, wav_fc[:min_len], 'g-', linewidth=0.8, label='FullCov')
            axes[2].legend(loc='upper right')
            axes[2].grid(True, alpha=0.3)

            axes[3].plot(time, wav_flow[:min_len], 'r-', linewidth=0.8, label='Flow')
            axes[3].legend(loc='upper right')
            axes[3].grid(True, alpha=0.3)

            axes[4].plot(time, wav_mr[:min_len], 'm-', linewidth=0.8, label='MRLoss')
            axes[4].set_xlabel('Time (s)')
            axes[4].legend(loc='upper right')
            axes[4].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'{output_dir}/waveforms/{event_id}_{station_name}_waveforms.png', dpi=150, bbox_inches='tight')
            plt.close()

    summary = {k: float(np.mean(v)) if v else None for k, v in metrics.items()}
    with open(args.output_metrics, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
