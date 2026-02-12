import torch
import torch.nn.functional as F


def reconstruct_waveform_with_gt_phase(
    pred_spec_z,
    gt_waveform,
    mag_min,
    mag_max,
    n_fft=256,
    hop_length=64,
    win_length=256,
    eps=1e-8,
):
    """
    Build differentiable waveform prediction by combining predicted magnitude
    with GT phase from waveform STFT.

    Args:
        pred_spec_z: (B, F, T) normalized/log-domain predicted spectrogram (Z channel)
        gt_waveform: (B, L)
        mag_min: (B,)
        mag_max: (B,)
    """
    device = pred_spec_z.device
    window = torch.hann_window(win_length, device=device)

    # Inverse normalization + inverse log1p.
    mag_min = mag_min.view(-1, 1, 1)
    mag_max = mag_max.view(-1, 1, 1)
    pred_log = pred_spec_z * (mag_max - mag_min) + mag_min
    pred_mag = torch.expm1(pred_log).clamp_min(eps)

    gt_stft = torch.stft(
        gt_waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )

    phase = torch.angle(gt_stft)

    min_f = min(pred_mag.size(1), phase.size(1))
    pred_mag = pred_mag[:, :min_f, :]
    phase = phase[:, :min_f, :]

    # Align time frames while preserving as much predicted magnitude as possible.
    if phase.size(2) > pred_mag.size(2):
        phase = phase[:, :, :pred_mag.size(2)]
    elif pred_mag.size(2) > phase.size(2):
        pred_mag = pred_mag[:, :, :phase.size(2)]

    pred_complex = torch.polar(pred_mag, phase)

    pred_wave = torch.istft(
        pred_complex,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
    )
    target_len = gt_waveform.size(1)
    if pred_wave.size(1) > target_len:
        pred_wave = pred_wave[:, :target_len]
    elif pred_wave.size(1) < target_len:
        pred_wave = F.pad(pred_wave, (0, target_len - pred_wave.size(1)), mode='constant', value=0.0)
    return pred_wave


def _stft_mag(x, n_fft, hop_length, win_length):
    window = torch.hann_window(win_length, device=x.device)
    stft = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    return torch.abs(stft).clamp_min(1e-8)


def multi_resolution_stft_loss(
    pred_wave,
    gt_wave,
    fft_sizes=(64, 256, 1024),
    hop_sizes=(16, 64, 256),
    win_lengths=(64, 256, 1024),
    scale_weights=(1.0, 0.7, 0.5),
    alpha_sc=1.0,
    alpha_mag=1.0,
    eps=1e-8,
):
    """
    Multi-resolution STFT loss.
    Returns total loss and per-scale diagnostics.
    """
    if not (len(fft_sizes) == len(hop_sizes) == len(win_lengths) == len(scale_weights)):
        raise ValueError('fft_sizes, hop_sizes, win_lengths, scale_weights must have same length')

    total = pred_wave.new_tensor(0.0)
    details = {}

    for i, (n_fft, hop, win, w) in enumerate(zip(fft_sizes, hop_sizes, win_lengths, scale_weights)):
        mag_pred = _stft_mag(pred_wave, n_fft=n_fft, hop_length=hop, win_length=win)
        mag_gt = _stft_mag(gt_wave, n_fft=n_fft, hop_length=hop, win_length=win)

        # Spectral convergence.
        sc_num = torch.norm(mag_gt - mag_pred, p='fro')
        sc_den = torch.norm(mag_gt, p='fro') + eps
        l_sc = sc_num / sc_den

        # Log-magnitude L1.
        l_mag = F.l1_loss(torch.log(mag_pred + eps), torch.log(mag_gt + eps))

        comp = w * (alpha_sc * l_sc + alpha_mag * l_mag)
        total = total + comp

        details[f'scale{i}_nfft'] = int(n_fft)
        details[f'scale{i}_sc'] = float(l_sc.detach().cpu().item())
        details[f'scale{i}_mag'] = float(l_mag.detach().cpu().item())
        details[f'scale{i}_weighted'] = float(comp.detach().cpu().item())

    return total, details
