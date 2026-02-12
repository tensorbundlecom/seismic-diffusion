import argparse
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

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder
from ML.autoencoder.experiments.FullCovariance.core.model_full_cov import FullCovCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE
from ML.autoencoder.experiments.WBaseline.core.model_wbaseline import WBaselineCVAE
from ML.autoencoder.experiments.WFullCov.core.model_wfullcov import WFullCovCVAE
from ML.autoencoder.experiments.NormalizingW.core.model_wflow import WFlowCVAE


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


def calculate_seismic_metrics(target_wav, pred_wav, target_spec, pred_spec, fs=100.0):
    metrics = {}

    s1 = (target_spec - np.min(target_spec)) / (np.max(target_spec) - np.min(target_spec) + 1e-8)
    s2 = (pred_spec - np.min(pred_spec)) / (np.max(pred_spec) - np.min(pred_spec) + 1e-8)
    metrics['ssim'] = float(ssim(s1, s2, data_range=1.0))

    metrics['lsd'] = float(np.sqrt(np.mean((np.log(target_spec + 1e-8) - np.log(pred_spec + 1e-8)) ** 2)))

    metrics['sc'] = float(np.linalg.norm(target_spec - pred_spec) / (np.linalg.norm(target_spec) + 1e-8))

    metrics['s_corr'] = float(np.corrcoef(target_spec.flatten(), pred_spec.flatten())[0, 1])

    spec_power_target = np.sum(target_spec, axis=0)
    spec_power_pred = np.sum(pred_spec, axis=0)

    def get_spectral_sta_lta(power, sta_len=5, lta_len=40):
        sta = np.convolve(power, np.ones(sta_len) / sta_len, mode='same')
        lta = np.convolve(power, np.ones(lta_len) / lta_len, mode='same')
        return sta / (lta + 1e-8)

    sl_target = get_spectral_sta_lta(spec_power_target)
    sl_pred = get_spectral_sta_lta(spec_power_pred)
    metrics['sta_lta_err'] = float(np.abs(np.max(sl_target) - np.max(sl_pred)) / (np.max(sl_target) + 1e-8))

    mr_lsd = []
    for n_fft in [64, 128, 512]:
        hop = n_fft // 4
        _, _, z1 = signal.stft(target_wav, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
        _, _, z2 = signal.stft(pred_wav, fs=fs, nperseg=n_fft, noverlap=n_fft - hop)
        t_spec = np.abs(z1)
        p_spec = np.abs(z2)
        min_f = min(t_spec.shape[0], p_spec.shape[0])
        min_t = min(t_spec.shape[1], p_spec.shape[1])
        mr_lsd.append(np.sqrt(np.mean((np.log(t_spec[:min_f, :min_t] + 1e-8) - np.log(p_spec[:min_f, :min_t] + 1e-8)) ** 2)))
    metrics['mr_lsd'] = float(np.mean(mr_lsd))

    try:
        from scipy.integrate import trapezoid
        a_target = (np.pi / (2 * 9.81)) * trapezoid(target_wav ** 2, dx=1 / fs)
        a_pred = (np.pi / (2 * 9.81)) * trapezoid(pred_wav ** 2, dx=1 / fs)
    except Exception:
        a_target = (np.pi / (2 * 9.81)) * np.trapz(target_wav ** 2, dx=1 / fs)
        a_pred = (np.pi / (2 * 9.81)) * np.trapz(pred_wav ** 2, dx=1 / fs)
    metrics['arias_err'] = float(np.abs(a_target - a_pred) / (np.abs(a_target) + 1e-8))

    env1 = np.abs(hilbert(target_wav))
    env2 = np.abs(hilbert(pred_wav))
    min_len = min(len(env1), len(env2))
    metrics['env_corr'] = float(np.corrcoef(env1[:min_len], env2[:min_len])[0, 1])

    factor = max(1, len(target_wav) // 500)
    s_target = target_wav[::factor].reshape(-1, 1)
    s_pred = pred_wav[::factor].reshape(-1, 1)
    dtw_dist, _ = fastdtw(s_target, s_pred, dist=euclidean)
    metrics['dtw'] = float(dtw_dist / len(s_target))

    x1 = (target_wav - np.mean(target_wav)) / (np.std(target_wav) + 1e-8)
    x2 = (pred_wav - np.mean(pred_wav)) / (np.std(pred_wav) + 1e-8)
    min_len = min(len(x1), len(x2))
    xcorr = np.correlate(x1[:min_len], x2[:min_len], mode='full')
    metrics['xcorr'] = float(np.max(np.abs(xcorr)) / len(x1[:min_len]))

    return metrics


def mean_metrics(metrics_dict):
    out = {}
    for k, v in metrics_dict.items():
        if len(v) == 0:
            out[k] = None
        else:
            out[k] = float(np.nanmean(v))
    return out


def format_md_table(agg):
    headers = [
        'Model', 'SSIM', 'LSD', 'SC', 'S-Corr', 'STA/LTA Err', 'MR-LSD',
        'Arias Err', 'Env Corr', 'DTW', 'XCorr'
    ]
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + ' | '.join([':---'] + [':---:' for _ in headers[1:]]) + ' |')
    for model, metrics in agg.items():
        row = [model]
        for key in ['ssim', 'lsd', 'sc', 's_corr', 'sta_lta_err', 'mr_lsd', 'arias_err', 'env_corr', 'dtw', 'xcorr']:
            val = metrics.get(key)
            if val is None:
                row.append('--')
            else:
                if key in ['dtw']:
                    row.append(f'{val:.2f}')
                else:
                    row.append(f'{val:.4f}')
        lines.append('| ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


def interpret_results(agg):
    # Determine best per metric (higher or lower)
    higher_better = {'ssim', 's_corr', 'env_corr', 'xcorr'}
    lower_better = {'lsd', 'sc', 'sta_lta_err', 'mr_lsd', 'arias_err', 'dtw'}

    best = {}
    for metric in list(higher_better | lower_better):
        values = {m: v.get(metric) for m, v in agg.items() if v.get(metric) is not None}
        if not values:
            continue
        if metric in higher_better:
            best_model = max(values, key=values.get)
        else:
            best_model = min(values, key=values.get)
        best[metric] = best_model

    lines = []
    if best:
        lines.append('Best-per-metric highlights:')
        for metric in ['ssim', 'lsd', 'sc', 's_corr', 'sta_lta_err', 'mr_lsd', 'arias_err', 'env_corr', 'dtw', 'xcorr']:
            if metric in best:
                lines.append(f'- `{metric}`: **{best[metric]}**')

    # Coarse summary
    lines.append('General interpretation:')
    lines.append('The summary shows trade-offs between structural similarity (SSIM/S-Corr), spectral fidelity (LSD/MR-LSD), energy matching (Arias Err), and temporal alignment (DTW/XCorr).')
    lines.append('Models that improve LSD or MR-LSD may not lead on SSIM, indicating sharper spectral detail at some cost to global structure.' )
    lines.append('Use DTW and XCorr together to judge timing vs phase alignment, and prioritize metrics based on the downstream objective (onset accuracy vs spectral texture vs waveform alignment).')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Evaluate custom post-training OOD across all models.')
    parser.add_argument('--ood_data_dir', default='data/ood_waveforms/post_training_custom/filtered', help='OOD data root.')
    parser.add_argument('--ood_catalog', default='data/events/ood_catalog_post_training.txt', help='OOD catalog path.')
    parser.add_argument('--station_list_file', default='data/station_list_external_full.json', help='Station list JSON.')
    parser.add_argument('--station_subset_file', default='data/station_list_post_custom.json', help='Subset stations JSON.')
    parser.add_argument('--output_metrics', default='ML/autoencoder/experiments/General/results/post_training_custom_ood_all_models_metrics.json', help='Output JSON metrics path.')
    parser.add_argument('--output_md', default='ML/autoencoder/experiments/General/setup/docs/post_training_custom_ood_metrics_summary.md', help='Output markdown summary path.')
    parser.add_argument('--mrloss_checkpoint', default='ML/autoencoder/experiments/CVAE_MRLoss/checkpoints/cvae_mrloss_pilot_b0p1_l0p5_fix_20260211_155549_best.pt', help='MRLoss checkpoint path.')
    parser.add_argument('--baseline_checkpoint', default='ML/autoencoder/experiments/General/checkpoints/baseline_external_best.pt', help='Baseline checkpoint path.')
    parser.add_argument('--fullcov_checkpoint', default='ML/autoencoder/experiments/FullCovariance/checkpoints/full_cov_external_best.pt', help='FullCov checkpoint path.')
    parser.add_argument('--flow_checkpoint', default='ML/autoencoder/experiments/NormalizingFlow/checkpoints/flow_external_best.pt', help='Flow checkpoint path.')
    parser.add_argument('--wbaseline_checkpoint', default='ML/autoencoder/experiments/WBaseline/checkpoints/wbaseline_external_best.pt', help='WBaseline checkpoint path.')
    parser.add_argument('--wfullcov_checkpoint', default='ML/autoencoder/experiments/WFullCov/checkpoints/wfullcov_external_best.pt', help='WFullCov checkpoint path.')
    parser.add_argument('--wflow_checkpoint', default='ML/autoencoder/experiments/NormalizingW/checkpoints/wflow_external_best.pt', help='WFlow checkpoint path.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.station_list_file, 'r') as f:
        station_list = json.load(f)
    station_subset = None
    if args.station_subset_file:
        with open(args.station_subset_file, 'r') as f:
            station_subset = set(json.load(f))

    num_stations = len(station_list)

    # Load models
    base_model = ConditionalVariationalAutoencoder(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    base_model.load_state_dict(torch.load(args.baseline_checkpoint, map_location=device)['model_state_dict'])
    base_model.eval()

    fc_model = FullCovCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    fc_model.load_state_dict(torch.load(args.fullcov_checkpoint, map_location=device)['model_state_dict'])
    fc_model.eval()

    flow_model = FlowCVAE(in_channels=3, latent_dim=128, num_stations=num_stations, flow_layers=8).to(device)
    flow_model.load_state_dict(torch.load(args.flow_checkpoint, map_location=device)['model_state_dict'])
    flow_model.eval()

    wbaseline = WBaselineCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    wbaseline.load_state_dict(torch.load(args.wbaseline_checkpoint, map_location=device)['model_state_dict'])
    wbaseline.eval()

    wfullcov = WFullCovCVAE(in_channels=3, latent_dim=128, num_stations=num_stations).to(device)
    wfullcov.load_state_dict(torch.load(args.wfullcov_checkpoint, map_location=device)['model_state_dict'])
    wfullcov.eval()

    wflow = WFlowCVAE(in_channels=3, latent_dim=128, num_stations=num_stations, flow_layers=8).to(device)
    wflow.load_state_dict(torch.load(args.wflow_checkpoint, map_location=device)['model_state_dict'])
    wflow.eval()

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

    metric_keys = ['ssim', 'lsd', 'sc', 's_corr', 'sta_lta_err', 'mr_lsd', 'arias_err', 'env_corr', 'dtw', 'xcorr']
    results = {m: {k: [] for k in metric_keys} for m in ['Baseline', 'FullCov', 'Flow', 'WBaseline', 'WFullCov', 'WFlow', 'CVAE_MRLoss']}

    target_fs = 100.0

    with torch.no_grad():
        for i in range(len(dataset)):
            spec, mag, loc, station_idx, meta = dataset[i]
            if 'error' in meta:
                continue

            file_path = meta['file_path']
            file_name = os.path.basename(file_path)
            parts = file_name.split('_')
            if len(parts) >= 4:
                station_name = parts[3]
            else:
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

            r_base, _, _ = base_model(spec_in, mag_in, loc_in, sta_in)
            r_fc, _, _ = fc_model(spec_in, mag_in, loc_in, sta_in)
            r_flow, _, _, _, _ = flow_model(spec_in, mag_in, loc_in, sta_in)

            r_wbase, _, _ = wbaseline(spec_in, mag_in, loc_in, sta_in)
            r_wfc, _, _ = wfullcov(spec_in, mag_in, loc_in, sta_in)
            r_wflow, _, _, _, _ = wflow(spec_in, mag_in, loc_in, sta_in)

            r_mr, _, _ = mrloss_model(spec_in, mag_in, loc_in, sta_in)

            orig_spec = spec_in[0, 2].cpu().numpy()
            base_spec = r_base[0, 2].cpu().numpy()
            fc_spec = r_fc[0, 2].cpu().numpy()
            flow_spec = r_flow[0, 2].cpu().numpy()
            wbase_spec = r_wbase[0, 2].cpu().numpy()
            wfc_spec = r_wfc[0, 2].cpu().numpy()
            wflow_spec = r_wflow[0, 2].cpu().numpy()
            mr_spec = r_mr[0, 2].cpu().numpy()

            wav_base = reconstruct_signal(base_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=target_fs)
            wav_fc = reconstruct_signal(fc_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=target_fs)
            wav_flow = reconstruct_signal(flow_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=target_fs)
            wav_wbase = reconstruct_signal(wbase_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=target_fs)
            wav_wfc = reconstruct_signal(wfc_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=target_fs)
            wav_wflow = reconstruct_signal(wflow_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=target_fs)
            wav_mr = reconstruct_signal(mr_spec, mag_min, mag_max, nperseg=256, noverlap=192, fs=target_fs)

            min_len = min(len(gt_wav), len(wav_base), len(wav_fc), len(wav_flow), len(wav_wbase), len(wav_wfc), len(wav_wflow), len(wav_mr))
            gt_wav = gt_wav[:min_len]

            def add_metrics(name, pred_wav, pred_spec):
                m = calculate_seismic_metrics(gt_wav, pred_wav[:min_len], orig_spec, pred_spec, fs=target_fs)
                for k in metric_keys:
                    results[name][k].append(m[k])

            add_metrics('Baseline', wav_base, base_spec)
            add_metrics('FullCov', wav_fc, fc_spec)
            add_metrics('Flow', wav_flow, flow_spec)
            add_metrics('WBaseline', wav_wbase, wbase_spec)
            add_metrics('WFullCov', wav_wfc, wfc_spec)
            add_metrics('WFlow', wav_wflow, wflow_spec)
            add_metrics('CVAE_MRLoss', wav_mr, mr_spec)

    aggregated = {m: mean_metrics(metrics) for m, metrics in results.items()}

    os.makedirs(os.path.dirname(args.output_metrics), exist_ok=True)
    with open(args.output_metrics, 'w') as f:
        json.dump(aggregated, f, indent=2)

    md_lines = []
    md_lines.append('# Post-Training Custom OOD (HH-only) Metrics Summary')
    md_lines.append('')
    md_lines.append('This summary compares all models on the custom post-training OOD set (HH-only, station subset).')
    md_lines.append('')
    md_lines.append('## Metrics Table')
    md_lines.append('')
    md_lines.append(format_md_table(aggregated))
    md_lines.append('')
    md_lines.append('## Metric Definitions')
    md_lines.append('')
    md_lines.append('- `SSIM`: Spectrogram structural similarity (higher is better).')
    md_lines.append('- `LSD`: Log-spectral distance (lower is better).')
    md_lines.append('- `SC`: Spectral convergence (lower is better).')
    md_lines.append('- `S-Corr`: Spectral correlation (higher is better).')
    md_lines.append('- `STA/LTA Err`: Onset energy ratio error (lower is better).')
    md_lines.append('- `MR-LSD`: Multi-resolution log-spectral distance (lower is better).')
    md_lines.append('- `Arias Err`: Arias intensity error (lower is better).')
    md_lines.append('- `Env Corr`: Envelope correlation (higher is better).')
    md_lines.append('- `DTW`: Dynamic time warping distance (lower is better).')
    md_lines.append('- `XCorr`: Maximum cross-correlation (higher is better).')
    md_lines.append('')
    md_lines.append('## Overall Interpretation')
    md_lines.append('')
    md_lines.append(interpret_results(aggregated))

    os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
    with open(args.output_md, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')

    print(f'[INFO] Metrics JSON saved to: {args.output_metrics}')
    print(f'[INFO] Markdown summary saved to: {args.output_md}')


if __name__ == '__main__':
    main()
