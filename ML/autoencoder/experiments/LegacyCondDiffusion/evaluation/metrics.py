import numpy as np
from scipy import signal
from scipy.signal import hilbert

try:
    from fastdtw import fastdtw
except Exception:
    fastdtw = None

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None


def reconstruct_signal_griffin_lim(
    magnitude_spec_norm: np.ndarray,
    mag_min: float,
    mag_max: float,
    fs: float = 100.0,
    n_iter: int = 64,
):
    spec = magnitude_spec_norm.copy()
    if mag_max > mag_min:
        spec = spec * (mag_max - mag_min) + mag_min
    spec = np.expm1(spec)

    phase = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    for _ in range(n_iter):
        z = spec * phase
        _, wav = signal.istft(z, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
        _, _, z_new = signal.stft(wav, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
        min_f = min(z_new.shape[0], spec.shape[0])
        min_t = min(z_new.shape[1], spec.shape[1])
        phase = np.zeros_like(spec, dtype=np.complex128)
        phase[:min_f, :min_t] = np.exp(1j * np.angle(z_new[:min_f, :min_t]))
        phase[np.abs(phase) == 0] = 1.0

    _, wav = signal.istft(spec * phase, fs=fs, nperseg=256, noverlap=192, nfft=256, boundary="zeros")
    return wav.astype(np.float32)


def _safe_corr(a: np.ndarray, b: np.ndarray):
    if a.size == 0 or b.size == 0:
        return 0.0
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spectral_convergence(target_spec: np.ndarray, pred_spec: np.ndarray):
    return float(np.linalg.norm(target_spec - pred_spec) / (np.linalg.norm(target_spec) + 1e-8))


def _log_spectral_distance(target_spec: np.ndarray, pred_spec: np.ndarray):
    return float(np.sqrt(np.mean((np.log(target_spec + 1e-8) - np.log(pred_spec + 1e-8)) ** 2)))


def _multires_lsd(target_wav: np.ndarray, pred_wav: np.ndarray, fs: float = 100.0):
    wins = [64, 256, 512]
    vals = []
    for w in wins:
        noverlap = int(0.75 * w)
        _, _, z1 = signal.stft(target_wav, fs=fs, nperseg=w, noverlap=noverlap, nfft=w, boundary="zeros")
        _, _, z2 = signal.stft(pred_wav, fs=fs, nperseg=w, noverlap=noverlap, nfft=w, boundary="zeros")
        a = np.log1p(np.abs(z1))
        b = np.log1p(np.abs(z2))
        mf = min(a.shape[0], b.shape[0])
        mt = min(a.shape[1], b.shape[1])
        vals.append(_log_spectral_distance(a[:mf, :mt], b[:mf, :mt]))
    return float(np.mean(vals))


def _arias_intensity(sig: np.ndarray, fs: float = 100.0):
    return float((np.pi / (2 * 9.81)) * np.trapz(sig ** 2, dx=1.0 / fs))


def _sta_lta_peak(sig: np.ndarray, fs: float = 100.0, sta_sec: float = 0.5, lta_sec: float = 5.0):
    x = np.abs(sig).astype(np.float64)
    n_sta = max(1, int(fs * sta_sec))
    n_lta = max(n_sta + 1, int(fs * lta_sec))
    sta = np.convolve(x, np.ones(n_sta) / n_sta, mode="same")
    lta = np.convolve(x, np.ones(n_lta) / n_lta, mode="same")
    ratio = sta / (lta + 1e-8)
    return float(np.max(ratio))


def _dtw_distance(a: np.ndarray, b: np.ndarray):
    if fastdtw is None:
        # Fallback: L1 aligned distance after subsampling.
        m = min(len(a), len(b))
        if m == 0:
            return 0.0
        return float(np.mean(np.abs(a[:m] - b[:m])))
    factor = max(1, len(a) // 500)
    sa = a[::factor].reshape(-1, 1)
    sb = b[::factor].reshape(-1, 1)
    dist, _ = fastdtw(sa, sb)
    return float(dist / max(1, len(sa)))


def calculate_all_metrics(
    target_wav: np.ndarray,
    pred_wav: np.ndarray,
    target_spec: np.ndarray,
    pred_spec: np.ndarray,
    fs: float = 100.0,
):
    out = {}

    if ssim is not None:
        s1 = (target_spec - np.min(target_spec)) / (np.max(target_spec) - np.min(target_spec) + 1e-8)
        s2 = (pred_spec - np.min(pred_spec)) / (np.max(pred_spec) - np.min(pred_spec) + 1e-8)
        out["ssim"] = float(ssim(s1, s2, data_range=1.0))
    else:
        out["ssim"] = 0.0

    out["lsd"] = _log_spectral_distance(target_spec, pred_spec)
    out["sc"] = _spectral_convergence(target_spec, pred_spec)
    out["s_corr"] = _safe_corr(target_spec.flatten(), pred_spec.flatten())

    at = _arias_intensity(target_wav, fs=fs)
    ap = _arias_intensity(pred_wav, fs=fs)
    out["arias_err"] = float(np.abs(at - ap) / (np.abs(at) + 1e-8))

    env_t = np.abs(hilbert(target_wav))
    env_p = np.abs(hilbert(pred_wav))
    m = min(len(env_t), len(env_p))
    out["env_corr"] = _safe_corr(env_t[:m], env_p[:m])

    out["dtw"] = _dtw_distance(target_wav, pred_wav)

    x1 = (target_wav - np.mean(target_wav)) / (np.std(target_wav) + 1e-8)
    x2 = (pred_wav - np.mean(pred_wav)) / (np.std(pred_wav) + 1e-8)
    m = min(len(x1), len(x2))
    xc = np.correlate(x1[:m], x2[:m], mode="full")
    out["xcorr"] = float(np.max(np.abs(xc)) / max(1, m))

    out["mr_lsd"] = _multires_lsd(target_wav, pred_wav, fs=fs)
    sta_t = _sta_lta_peak(target_wav, fs=fs)
    sta_p = _sta_lta_peak(pred_wav, fs=fs)
    out["sta_lta_err"] = float(np.abs(sta_t - sta_p) / (np.abs(sta_t) + 1e-8))
    return out

