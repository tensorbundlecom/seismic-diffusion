import json
import os
import sys

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
from ML.autoencoder.experiments.WFullCov.core.model_wfullcov import WFullCovCVAE


def reconstruct_signal(magnitude_spec, mag_min=0.0, mag_max=1.0, fs=100.0):
    spec = magnitude_spec.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)

    phase = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    for _ in range(64):
        stft_complex = spec * phase
        _, waveform = signal.istft(stft_complex, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary='zeros')
        _, _, new_zxx = signal.stft(waveform, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary='zeros')
        if new_zxx.shape != spec.shape:
            min_f = min(new_zxx.shape[0], spec.shape[0])
            min_t = min(new_zxx.shape[1], spec.shape[1])
            next_phase = np.zeros_like(spec, dtype=complex)
            next_phase[:min_f, :min_t] = np.exp(1j * np.angle(new_zxx[:min_f, :min_t]))
            phase = next_phase
        else:
            phase = np.exp(1j * np.angle(new_zxx))

    stft_complex = spec * phase
    _, waveform = signal.istft(stft_complex, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary='zeros')
    return waveform


def calculate_metrics(target_wav, pred_wav, target_spec, pred_spec, fs=100.0):
    out = {}

    s1 = (target_spec - np.min(target_spec)) / (np.max(target_spec) - np.min(target_spec) + 1e-8)
    s2 = (pred_spec - np.min(pred_spec)) / (np.max(pred_spec) - np.min(pred_spec) + 1e-8)
    out['ssim'] = float(ssim(s1, s2, data_range=1.0))

    out['lsd'] = float(np.sqrt(np.mean((np.log(target_spec + 1e-8) - np.log(pred_spec + 1e-8)) ** 2)))

    out['arias_err'] = float(
        np.abs(
            ((np.pi / (2 * 9.81)) * np.trapz(target_wav ** 2, dx=1 / fs))
            - ((np.pi / (2 * 9.81)) * np.trapz(pred_wav ** 2, dx=1 / fs))
        )
        / (np.abs((np.pi / (2 * 9.81)) * np.trapz(target_wav ** 2, dx=1 / fs)) + 1e-8)
    )

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


def load_models(device, num_stations):
    baseline_chk = 'ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt'
    full_cov_chk = 'ML/autoencoder/experiments/FullCovariance/checkpoints/full_cov_external_best.pt'
    flow_chk = 'ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_external_best.pt'
    wfullcov_chk = 'ML/autoencoder/experiments/WFullCov/checkpoints/wfullcov_external_best.pt'

    for p in [baseline_chk, full_cov_chk, flow_chk, wfullcov_chk]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing checkpoint: {p}')

    baseline = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    baseline.load_state_dict(torch.load(baseline_chk, map_location=device)['model_state_dict'])
    baseline.eval()

    full_cov = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    full_cov.load_state_dict(torch.load(full_cov_chk, map_location=device)['model_state_dict'])
    full_cov.eval()

    flow = FlowCVAE(in_channels=3, latent_dim=128, num_stations=num_stations, flow_layers=8).to(device)
    flow.load_state_dict(torch.load(flow_chk, map_location=device)['model_state_dict'])
    flow.eval()

    wb_state = torch.load(wfullcov_chk, map_location=device)
    cfg = wb_state.get('config', {})
    wfullcov = WFullCovCVAE(
        in_channels=3,
        latent_dim=cfg.get('latent_dim', 128),
        num_stations=cfg.get('num_stations', num_stations),
        w_dim=cfg.get('w_dim', 64),
    ).to(device)
    wfullcov.load_state_dict(wb_state['model_state_dict'])
    wfullcov.eval()

    return baseline, full_cov, flow, wfullcov


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = 'ML/autoencoder/experiments/WFullCov/results'
    os.makedirs(output_dir, exist_ok=True)

    station_list_file = 'data/station_list_external_full.json'
    with open(station_list_file, 'r') as f:
        station_list = json.load(f)

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir='data/ood_waveforms/post_training/filtered',
        event_file='data/events/ood_catalog_post_training.txt',
        channels=['HH'],
        magnitude_col='xM',
        station_list=station_list,
    )

    baseline, full_cov, flow, wfullcov = load_models(device=device, num_stations=len(station_list))

    metric_keys = ['ssim', 'lsd', 'arias_err', 'env_corr', 'dtw', 'xcorr']
    results = {m: {k: [] for k in metric_keys} for m in ['Baseline', 'FullCov', 'Flow', 'WFullCov']}

    with torch.no_grad():
        for i in range(len(dataset)):
            spec, mag, loc, station_idx, meta = dataset[i]
            if 'error' in meta:
                continue

            st = obspy.read(meta['file_path'])
            st.resample(100.0)
            tr = st.select(component='Z')[0] if st.select(component='Z') else st[0]
            gt_wav = tr.data.astype(np.float32)
            target_len = 7300
            if len(gt_wav) > target_len:
                gt_wav = gt_wav[:target_len]
            elif len(gt_wav) < target_len:
                gt_wav = np.pad(gt_wav, (0, target_len - len(gt_wav)))

            _, _, zxx = signal.stft(gt_wav, fs=100.0, nperseg=256, noverlap=192, nfft=256, boundary='zeros')
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
            r_wb, _, _ = wfullcov(spec_in, mag_in, loc_in, sta_in)

            orig_spec = spec_in[0, 2].cpu().numpy()
            specs = {
                'Baseline': r_base[0, 2].cpu().numpy(),
                'FullCov': r_fc[0, 2].cpu().numpy(),
                'Flow': r_flow[0, 2].cpu().numpy(),
                'WFullCov': r_wb[0, 2].cpu().numpy(),
            }

            waves = {
                name: reconstruct_signal(pred_spec, mag_min=mag_min, mag_max=mag_max, fs=100.0)
                for name, pred_spec in specs.items()
            }

            for model_name in specs:
                min_len = min(len(gt_wav), len(waves[model_name]))
                metrics = calculate_metrics(
                    gt_wav[:min_len],
                    waves[model_name][:min_len],
                    orig_spec,
                    specs[model_name],
                    fs=100.0,
                )
                for k in metric_keys:
                    results[model_name][k].append(metrics[k])

            print(f'[INFO] Processed sample {i+1}/{len(dataset)}: {meta.get("file_name", "unknown")}')

    summary = {}
    for model_name, model_metrics in results.items():
        summary[model_name] = {}
        for k, vals in model_metrics.items():
            summary[model_name][k] = float(np.nanmean(vals)) if vals else None

    out_file = os.path.join(output_dir, 'diverse_wfullcov_comparison_metrics.json')
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print('[INFO] Summary metrics saved:', out_file)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
