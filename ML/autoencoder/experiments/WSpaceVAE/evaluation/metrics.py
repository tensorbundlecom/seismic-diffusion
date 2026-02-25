import numpy as np
from scipy import signal
from scipy.signal import hilbert

from fastdtw import fastdtw
from skimage.metrics import structural_similarity as ssim


def reconstruct_signal_griffin_lim(magnitude_spec_norm, mag_min=0.0, mag_max=1.0, fs=100.0, n_iter=64):
    spec = magnitude_spec_norm.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)

    phase = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    for _ in range(n_iter):
        z = spec * phase
        _, wav = signal.istft(z, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
        _, _, znew = signal.stft(wav, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
        mf = min(znew.shape[0], spec.shape[0])
        mt = min(znew.shape[1], spec.shape[1])
        phase = np.zeros_like(spec, dtype=np.complex128)
        phase[:mf, :mt] = np.exp(1j * np.angle(znew[:mf, :mt]))
        phase[np.abs(phase) == 0] = 1.0
    _, wav = signal.istft(spec * phase, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
    return wav.astype(np.float32)


def _safe_corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def calculate_all_metrics(target_wav, pred_wav, target_spec, pred_spec, fs=100.0):
    out = {}
    s1 = (target_spec - np.min(target_spec)) / (np.max(target_spec) - np.min(target_spec) + 1e-8)
    s2 = (pred_spec - np.min(pred_spec)) / (np.max(pred_spec) - np.min(pred_spec) + 1e-8)
    out["ssim"] = float(ssim(s1, s2, data_range=1.0))
    out["lsd"] = float(np.sqrt(np.mean((np.log(target_spec + 1e-8) - np.log(pred_spec + 1e-8)) ** 2)))
    out["sc"] = float(np.linalg.norm(target_spec - pred_spec) / (np.linalg.norm(target_spec) + 1e-8))
    out["s_corr"] = _safe_corr(target_spec.flatten(), pred_spec.flatten())

    arias = lambda x: (np.pi / (2 * 9.81)) * np.trapz(x ** 2, dx=1.0 / fs)
    out["arias_err"] = float(np.abs(arias(target_wav) - arias(pred_wav)) / (np.abs(arias(target_wav)) + 1e-8))

    env_t = np.abs(hilbert(target_wav))
    env_p = np.abs(hilbert(pred_wav))
    m = min(len(env_t), len(env_p))
    out["env_corr"] = _safe_corr(env_t[:m], env_p[:m])

    factor = max(1, len(target_wav) // 500)
    dtw_dist, _ = fastdtw(target_wav[::factor].reshape(-1, 1), pred_wav[::factor].reshape(-1, 1))
    out["dtw"] = float(dtw_dist / max(1, len(target_wav[::factor])))

    x1 = (target_wav - np.mean(target_wav)) / (np.std(target_wav) + 1e-8)
    x2 = (pred_wav - np.mean(pred_wav)) / (np.std(pred_wav) + 1e-8)
    m = min(len(x1), len(x2))
    xc = np.correlate(x1[:m], x2[:m], mode="full")
    out["xcorr"] = float(np.max(np.abs(xc)) / max(1, m))

    wins = [64, 256, 512]
    mrlsd = []
    for w in wins:
        no = int(0.75 * w)
        _, _, z1 = signal.stft(target_wav, fs=fs, nperseg=w, noverlap=no, nfft=w, boundary="zeros")
        _, _, z2 = signal.stft(pred_wav, fs=fs, nperseg=w, noverlap=no, nfft=w, boundary="zeros")
        a = np.log1p(np.abs(z1))
        b = np.log1p(np.abs(z2))
        mf = min(a.shape[0], b.shape[0])
        mt = min(a.shape[1], b.shape[1])
        mrlsd.append(float(np.sqrt(np.mean((a[:mf, :mt] - b[:mf, :mt]) ** 2))))
    out["mr_lsd"] = float(np.mean(mrlsd))

    def sta_lta_peak(sig):
        x = np.abs(sig).astype(np.float64)
        sta = np.convolve(x, np.ones(50) / 50, mode="same")
        lta = np.convolve(x, np.ones(500) / 500, mode="same")
        return float(np.max(sta / (lta + 1e-8)))

    st = sta_lta_peak(target_wav)
    sp = sta_lta_peak(pred_wav)
    out["sta_lta_err"] = float(np.abs(st - sp) / (np.abs(st) + 1e-8))
    return out

