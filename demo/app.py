"""
Seismic Diffusion Demo
======================
Interactive GUI for generating synthetic seismic waveforms conditioned on
event parameters (magnitude, location, depth) and recording station.

Run from the repo root:
    python demo/app.py
"""

import sys
import json
import threading
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import librosa
from scipy import signal as sp_signal

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ML   = ROOT / "ML"
DIFF = ML / "diffusion"
AE   = ML / "autoencoder"
sys.path.insert(0, str(ROOT))

from ML.diffusion.model import DiffusionUNet2D, create_conditioning_vector
from ML.autoencoder.inference import load_model as _load_ae
from diffusers import DDPMScheduler

# ── Constants ─────────────────────────────────────────────────────────────────
STATION_NAMES = [
    "ADVT", "ARMT", "BALB", "BGKT", "CAVI", "CRLT", "CTKS", "CTYL", "DKL",  "EDC",
    "EDRB", "ENEZ", "ERIK", "EZN",  "GADA", "GAZ",  "GAZK", "GEDZ", "GELI", "GEML",
    "GEMT", "GULT", "HRTX", "ISK",  "KAVV", "KCTX", "KLYT", "KMRS", "KOZT", "KRBG",
    "LAP",  "MDNY", "MDUB", "MRMT", "PHSR", "RKY",  "SARI", "SAUV", "SILT", "SIMA",
    "SLVT", "SPNC", "TVSB", "UVEZ", "YAYO", "YLV",
]

NUM_TRAIN_TIMESTEPS = 1000
FS      = 100.0   # Hz  (HH-channel sampling rate)
WAVEFORM_DISPLAY_SECONDS = 70.0
NPERSEG = 256
NOVERLAP= 192
NFFT    = 256
FREQ_BINS = NFFT // 2 + 1
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
GRIFFIN_LIM_ITERS = 400
GRIFFIN_LIM_MOMENTUM = 0.9
GRIFFIN_LIM_WINDOW = "hann"
GRIFFIN_LIM_CENTER = True
GRIFFIN_LIM_RANDOM_STATE = 0
_GRIFFIN_LIM_WARNED = False
# User-specified Marmara plotting bounds
MARMARA_LAT_MIN = 39.77  # South
MARMARA_LAT_MAX = 41.62  # North
MARMARA_LON_MIN = 26.52  # West
MARMARA_LON_MAX = 30.17  # East

# Dark-theme colour palette (Catppuccin Mocha)
BG      = "#1e1e2e"
SURFACE = "#181825"
OVERLAY = "#313244"
TEXT    = "#cdd6f4"
SUBTEXT = "#585b70"
BLUE    = "#89b4fa"
GREEN   = "#a6e3a1"
RED     = "#f38ba8"
YELLOW  = "#f9e2af"
TEAL    = "#89dceb"
CH_COLS = [BLUE, GREEN, RED]   # E / N / Z


# ── Scale helpers (loaded lazily from scale.json) ────────────────────────────
def _load_scale():
    path = DIFF / "embeddings" / "scale.json"
    if path.exists():
        return json.load(open(path))
    return {}


def _load_station_locations():
    """Load station coordinates used by diffusion conditioning."""
    path = DIFF / "embeddings" / "station_locations.json"
    if not path.exists():
        raise FileNotFoundError(
            "station_locations.json not found.\n"
            "Run: python ML/diffusion/fetch_station_locations.py"
        )
    return json.load(open(path))


def _get_embeddings_source_checkpoint():
    """Return AE checkpoint used to create embeddings, if recorded."""
    path = DIFF / "embeddings" / "source.json"
    if not path.exists():
        return None
    try:
        source = json.load(open(path))
    except Exception:
        return None
    ckpt = source.get("ae_checkpoint")
    if not ckpt:
        return None
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (DIFF / ckpt_path).resolve()
    return ckpt_path if ckpt_path.exists() else None


def _find_latest_timestamped_ae_checkpoint():
    """
    Return latest AE best_model.pt from timestamped dirs only (YYYYMMDD_HHMMSS),
    ignoring temporary folders such as _tmp_*.
    """
    timestamp_pat = re.compile(r"^\d{8}_\d{6}$")
    ckpts = sorted(
        p for p in (AE / "checkpoints").glob("*/best_model.pt")
        if timestamp_pat.match(p.parent.name)
    )
    return ckpts[-1] if ckpts else None


def _make_normalise_cond(scale: dict):
    """Return a closure that normalises continuous cond dims in-place."""
    if "cond_mean" not in scale:
        return lambda v: v
    mean = torch.tensor(scale["cond_mean"])
    std  = torch.tensor(scale["cond_std"]).clamp(min=1e-8)
    n    = len(mean)           # NUM_CONTINUOUS = 4

    def normalise(vec: torch.Tensor) -> torch.Tensor:
        out = vec.clone()
        out[:n] = (out[:n] - mean) / std
        return out
    return normalise

def _load_diffusion_scheduler(diff_ckpt: Path) -> DDPMScheduler:
    """
    Load scheduler settings from a diffusion checkpoint directory.
    Falls back to legacy defaults for older checkpoints that predate
    scheduler_config.json export.
    """
    scheduler_cfg = diff_ckpt / "scheduler_config.json"
    if scheduler_cfg.exists():
        return DDPMScheduler.from_pretrained(str(diff_ckpt))

    # Legacy fallback matching training defaults in ML/diffusion/train.py
    print(
        f"[demo] scheduler_config.json not found in {diff_ckpt.name}; "
        "using legacy DDPM defaults."
    )
    return DDPMScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS,
        beta_start=1e-4,
        beta_end=0.02,
        prediction_type="epsilon",
        clip_sample=False,
    )


def _read_diffusion_unet_config(diff_ckpt: Path):
    """Read UNet config.json for a diffusion checkpoint dir."""
    cfg_path = diff_ckpt / "config.json"
    if not cfg_path.exists():
        return None
    try:
        return json.load(open(cfg_path))
    except Exception:
        return None


def _read_diffusion_training_config(diff_ckpt: Path):
    """Read optional training_config.json saved with newer checkpoints."""
    cfg_path = diff_ckpt / "training_config.json"
    if not cfg_path.exists():
        return {}
    try:
        return json.load(open(cfg_path))
    except Exception:
        return {}


def _find_latest_diffusion_checkpoint():
    """
    Return latest diffusion checkpoint dir by mtime.
    Searches type-specific subdirs (ddpm/, flow_matching/) then legacy top-level dirs.
    """
    ckpt_root = DIFF / "checkpoints"
    if not ckpt_root.exists():
        return None, None

    candidates = []
    for type_name in ("ddpm", "flow_matching"):
        type_root = ckpt_root / type_name
        if type_root.exists():
            for p in type_root.iterdir():
                if p.is_dir() and (p / "config.json").exists():
                    candidates.append(p)
    for p in ckpt_root.iterdir():
        if not p.is_dir() or p.name in ("ddpm", "flow_matching"):
            continue
        if (p / "config.json").exists():
            candidates.append(p)

    if not candidates:
        return None, None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest, _read_diffusion_unet_config(latest)


def _find_latest_checkpoint_by_type(training_type: str):
    """Find latest checkpoint in checkpoints/{training_type}/ subfolder."""
    ckpt_root = DIFF / "checkpoints" / training_type
    if not ckpt_root.exists():
        return None, None
    candidates = [
        p for p in ckpt_root.iterdir()
        if p.is_dir() and (p / "config.json").exists()
    ]
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest, _read_diffusion_unet_config(latest)


def _find_compatible_diffusion_checkpoint(latent_channels: int):
    """
    Pick latest diffusion checkpoint whose in/out channels match AE latent size.
    Returns (ckpt_path, config_dict) or (None, None).
    """
    diff_ckpts = sorted((DIFF / "checkpoints").glob("epoch_*"))
    for ckpt in reversed(diff_ckpts):
        cfg = _read_diffusion_unet_config(ckpt)
        if cfg is None:
            continue
        in_ch = int(cfg.get("in_channels", -1))
        out_ch = int(cfg.get("out_channels", -1))
        if in_ch == latent_channels and out_ch == latent_channels:
            return ckpt, cfg
    return None, None


def _get_embeddings_source_stft_config():
    """Return source STFT params recorded during embedding creation, if available."""
    path = DIFF / "embeddings" / "source.json"
    defaults = {"nperseg": NPERSEG, "noverlap": NOVERLAP, "nfft": NFFT}
    if not path.exists():
        return defaults
    try:
        source = json.load(open(path))
        stft = source.get("stft", {})
        return {
            "nperseg": int(stft.get("nperseg", defaults["nperseg"])),
            "noverlap": int(stft.get("noverlap", defaults["noverlap"])),
            "nfft": int(stft.get("nfft", defaults["nfft"])),
        }
    except Exception:
        return defaults


def _apply_ae_stft_config(ae_config: dict) -> dict:
    """
    Update demo STFT/Griffin-Lim params from the loaded AE checkpoint config.

    This keeps generation/reconstruction settings aligned with whatever STFT
    resolution the autoencoder was trained on.
    """
    global NPERSEG, NOVERLAP, NFFT, FREQ_BINS

    nperseg = int(ae_config.get("nperseg", NPERSEG))
    noverlap = int(ae_config.get("noverlap", NOVERLAP))
    nfft = int(ae_config.get("nfft", NFFT))

    if nperseg <= 0:
        raise ValueError(f"Invalid nperseg from AE config: {nperseg}")
    if noverlap < 0 or noverlap >= nperseg:
        raise ValueError(
            f"Invalid noverlap from AE config: noverlap={noverlap}, nperseg={nperseg}. "
            "Expected 0 <= noverlap < nperseg."
        )
    if nfft < nperseg:
        raise ValueError(
            f"Invalid nfft from AE config: nfft={nfft}, nperseg={nperseg}. "
            "Expected nfft >= nperseg."
        )

    NPERSEG = nperseg
    NOVERLAP = noverlap
    NFFT = nfft
    FREQ_BINS = NFFT // 2 + 1
    return {
        "nperseg": NPERSEG,
        "noverlap": NOVERLAP,
        "nfft": NFFT,
        "freq_bins": FREQ_BINS,
    }


def _create_legacy_conditioning_vector(cond_meta: dict) -> torch.Tensor:
    """
    Legacy conditioning used by older diffusion checkpoints:
    [magnitude, latitude, longitude, depth, one_hot_station(46)].
    """
    continuous = torch.tensor(
        [
            float(cond_meta["magnitude"]),
            float(cond_meta["latitude"]),
            float(cond_meta["longitude"]),
            float(cond_meta["depth"]),
        ],
        dtype=torch.float32,
    )
    one_hot = torch.zeros(len(STATION_NAMES), dtype=torch.float32)
    one_hot[int(cond_meta["station_idx"])] = 1.0
    return torch.cat([continuous, one_hot], dim=0)


# ── Griffin-Lim reconstruction ───────────────────────────────────────────────
def griffin_lim_channel(magnitude: np.ndarray, n_iter: int = GRIFFIN_LIM_ITERS, params=None) -> np.ndarray:
    """
    Reconstruct one channel waveform from a magnitude STFT via Griffin-Lim.

    Parameters
    ----------
    magnitude : (freq_bins, time_bins) float32
        Normalised log-magnitude spectrogram (the raw VAE decoder output).
    """
    cfg = {
        "n_iter": n_iter,
        "hop_length": NPERSEG - NOVERLAP,
        "win_length": NPERSEG,
        "n_fft": NFFT,
        "window": GRIFFIN_LIM_WINDOW,
        "center": GRIFFIN_LIM_CENTER,
        "momentum": GRIFFIN_LIM_MOMENTUM,
        "random_state": GRIFFIN_LIM_RANDOM_STATE,
        "length": None,
        "inv_log_gain": 1.0,
    }
    if params is not None:
        cfg.update(params)

    cfg["n_iter"] = max(1, int(cfg["n_iter"]))
    cfg["hop_length"] = max(1, int(cfg["hop_length"]))
    cfg["n_fft"] = max(2, int(cfg["n_fft"]))
    cfg["win_length"] = max(1, min(int(cfg["win_length"]), cfg["n_fft"]))
    cfg["window"] = str(cfg["window"])
    cfg["center"] = bool(cfg["center"])
    cfg["momentum"] = float(np.clip(float(cfg["momentum"]), 0.0, 0.999))
    cfg["inv_log_gain"] = float(np.clip(float(cfg["inv_log_gain"]), 0.1, 20.0))
    if cfg.get("random_state", None) is not None:
        cfg["random_state"] = int(cfg["random_state"])

    # Invert the per-sample log1p normalization that SeismicSTFTDataset applies.
    # Exact per-sample min/max are unavailable at inference, so expose a scalar
    # gain in log-domain before expm1 as a practical calibration knob.
    mag = np.expm1(np.clip(magnitude, 0.0, None).astype(np.float64) * cfg["inv_log_gain"])
    # Make Griffin-Lim robust to custom STFT freq-bin counts from stft-mode training.
    expected_n_fft = max(2, int((mag.shape[0] - 1) * 2))
    if cfg["n_fft"] != expected_n_fft:
        cfg["n_fft"] = expected_n_fft
        cfg["win_length"] = min(cfg["win_length"], cfg["n_fft"])
        cfg["hop_length"] = min(cfg["hop_length"], cfg["n_fft"] - 1)
        cfg["hop_length"] = max(1, cfg["hop_length"])

    gl_kwargs = dict(
        n_iter=cfg["n_iter"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        n_fft=cfg["n_fft"],
        window=cfg["window"],
        center=cfg["center"],
        momentum=cfg["momentum"],
        random_state=cfg["random_state"],
    )

    length = cfg.get("length", None)
    if length is not None:
        length = max(1, int(length))
        try:
            return librosa.griffinlim(mag, length=length, **gl_kwargs).astype(np.float32)
        except Exception as exc:
            global _GRIFFIN_LIM_WARNED
            if not _GRIFFIN_LIM_WARNED:
                print(
                    "[demo] Griffin-Lim explicit length caused a frame mismatch; "
                    f"retrying without length. Details: {exc}"
                )
                _GRIFFIN_LIM_WARNED = True

    return librosa.griffinlim(mag, **gl_kwargs).astype(np.float32)


# ── Generation pipeline ──────────────────────────────────────────────────────
def _decode_to_spec(ae_model, x: torch.Tensor, emb_std: float, emb_mean: float = 0.0) -> np.ndarray:
    """Decode a latent batch tensor to a (3, FREQ_BINS, W) numpy spectrogram."""
    embedding = (x * emb_std + emb_mean).squeeze(0)
    with torch.no_grad():
        spec_t = ae_model.decode(embedding.unsqueeze(0))[0].cpu()
    if spec_t.shape[1] < FREQ_BINS:
        raise ValueError(
            f"Decoded spectrogram has too few frequency bins ({spec_t.shape[1]}), "
            f"but current STFT expects {FREQ_BINS}. "
            "Check AE checkpoint STFT params vs demo reconstruction params."
        )
    spec_t = spec_t.clamp(min=0.0)[:, :FREQ_BINS, :]
    return spec_t.numpy()


def _sample_to_spec(
    ae_model,
    x: torch.Tensor,
    emb_std: float,
    emb_mean: float,
    data_mode: str,
) -> np.ndarray:
    """Convert diffusion sample tensor to displayable spectrogram."""
    if data_mode == "stft":
        spec_t = (x * emb_std + emb_mean).squeeze(0).detach().cpu()
        if spec_t.shape[0] < 3:
            raise ValueError(
                f"STFT-mode sample has {spec_t.shape[0]} channels; expected at least 3."
            )
        spec_t = spec_t[:3].clamp(min=0.0)
        return spec_t.numpy()

    if ae_model is None:
        raise ValueError("AE model is required for latent-mode diffusion samples.")
    return _decode_to_spec(ae_model, x, emb_std=emb_std, emb_mean=emb_mean)


@torch.no_grad()
def generate(diff_unet, ae_model, scheduler, emb_shape,
             normalise_cond, emb_std, station_locations: dict, cond_meta: dict,
             step_callback=None, update_every: int = 50,
             griffin_lim_params=None,
             data_mode: str = "latent",
             emb_mean: float = 0.0,
             guidance_scale: float = 1.0,
             training_type: str = "ddpm"):
    """
    Full pipeline: conditioning → diffusion sampling → VAE decode → Griffin-Lim.

    step_callback(step, total, spec, waves_or_None) is called every `update_every`
    steps and always on the very last step (waves is only provided then).
    guidance_scale > 1 enables classifier-free guidance.
    training_type: 'ddpm' uses DDPM scheduler; 'flow_matching' uses Euler ODE (100 steps).
    """
    if hasattr(diff_unet, "station_embedding"):
        cond_vec = create_conditioning_vector(cond_meta, station_locations)
        expected_width = diff_unet.num_continuous + 1
        if cond_vec.shape[0] > expected_width:
            cond_vec = torch.cat([cond_vec[:diff_unet.num_continuous], cond_vec[-1:]])
    else:
        cond_vec = _create_legacy_conditioning_vector(cond_meta)
    cond_norm = normalise_cond(cond_vec).unsqueeze(0).unsqueeze(0).to(DEVICE)
    null_cond = torch.zeros_like(cond_norm)

    x = torch.randn(1, *emb_shape, device=DEVICE)
    diff_unet.eval()

    def _pred(x_, t_):
        if guidance_scale != 1.0:
            x_in = torch.cat([x_, x_])
            t_in = torch.cat([t_, t_])
            c_in = torch.cat([null_cond, cond_norm])
            if hasattr(diff_unet, "station_embedding"):
                preds = diff_unet.forward(x_in, t_in, c_in).sample
            else:
                preds = diff_unet(x_in, t_in, encoder_hidden_states=c_in).sample
            u, c = preds.chunk(2)
            return u + guidance_scale * (c - u)
        if hasattr(diff_unet, "station_embedding"):
            return diff_unet.forward(x_, t_, cond_norm).sample
        return diff_unet(x_, t_, encoder_hidden_states=cond_norm).sample

    if training_type == "flow_matching":
        num_steps = min(100, NUM_TRAIN_TIMESTEPS)
        dt = 1.0 / num_steps
        total = num_steps
        for i in range(num_steps):
            t_cont = 1.0 - i * dt
            t_b = torch.tensor([round(t_cont * NUM_TRAIN_TIMESTEPS)], device=DEVICE).long()
            x = x - dt * _pred(x, t_b)

            if step_callback is not None:
                is_last = (i == total - 1)
                if i % update_every == 0 or is_last:
                    spec  = _sample_to_spec(ae_model, x, emb_std=emb_std, emb_mean=emb_mean, data_mode=data_mode)
                    waves = (np.stack([griffin_lim_channel(spec[ch], params=griffin_lim_params)
                                       for ch in range(3)], axis=0)
                             if is_last else None)
                    step_callback(i + 1, total, spec, waves)
    else:
        num_steps = int(getattr(scheduler.config, "num_train_timesteps", NUM_TRAIN_TIMESTEPS))
        scheduler.set_timesteps(num_steps)
        timesteps = scheduler.timesteps
        total     = len(timesteps)

        for i, t in enumerate(timesteps):
            t_b = t.unsqueeze(0).to(DEVICE)
            noise_pred = _pred(x, t_b)
            x = scheduler.step(noise_pred, t, x).prev_sample

            if step_callback is not None:
                is_last = (i == total - 1)
                if i % update_every == 0 or is_last:
                    spec  = _sample_to_spec(ae_model, x, emb_std=emb_std, emb_mean=emb_mean, data_mode=data_mode)
                    waves = (np.stack([griffin_lim_channel(spec[ch], params=griffin_lim_params)
                                       for ch in range(3)], axis=0)
                             if is_last else None)
                    step_callback(i + 1, total, spec, waves)

    spec  = _sample_to_spec(ae_model, x, emb_std=emb_std, emb_mean=emb_mean, data_mode=data_mode)
    waves = np.stack([griffin_lim_channel(spec[ch], params=griffin_lim_params) for ch in range(3)], axis=0)
    return spec, waves


# ── GUI ───────────────────────────────────────────────────────────────────────
class SeismicDemoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Seismic Diffusion Demo")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._diff_unet     = None
        self._ae_model      = None
        self._scheduler     = None
        self._emb_shape     = None
        self._emb_std       = 1.0
        self._emb_mean      = 0.0
        self._data_mode     = "latent"
        self._training_type = "ddpm"
        self._normalise_cond = lambda v: v
        self._station_locations = {}
        self._train_metadatas   = []
        self._generating    = False
        self._latest_spec   = None
        self._latest_title  = ""
        self._gl_params_at_generate = None
        self._gl_recompute_token = 0
        self._cond_meta_at_generate = None
        self._title_at_generate = ""
        self._map_fig = None
        self._map_ax = None
        self._map_canvas = None

        self._apply_theme()
        self._build_ui()
        threading.Thread(target=self._load_models, daemon=True).start()

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _apply_theme(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure(".",                background=BG,      foreground=TEXT,  font=("Helvetica", 10))
        s.configure("TFrame",           background=BG)
        s.configure("TLabel",           background=BG,      foreground=TEXT)
        s.configure("TLabelframe",      background=BG,      foreground=BLUE)
        s.configure("TLabelframe.Label",background=BG,      foreground=BLUE,  font=("Helvetica", 10, "bold"))
        s.configure("TButton",          background=BLUE,    foreground=BG,    font=("Helvetica", 11, "bold"))
        s.map("TButton", background=[("disabled", OVERLAY), ("active", TEXT)])
        s.configure("TCombobox",        fieldbackground=OVERLAY, foreground=TEXT, background=OVERLAY)
        s.configure("Horizontal.TScale",background=BG,     troughcolor=OVERLAY, sliderlength=16)
        s.configure("TSeparator",       background=OVERLAY)

    # ── UI layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=12, pady=12)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        self._build_controls(root)
        self._build_plots(root)

    def _build_controls(self, parent):
        ctrl = ttk.LabelFrame(parent, text="Conditioning Parameters", padding=12)
        ctrl.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        # Model Type dropdown
        ttk.Label(ctrl, text="Model Type").pack(anchor="w")
        self._model_type_var = tk.StringVar(value="DDPM")
        ttk.Combobox(
            ctrl,
            textvariable=self._model_type_var,
            values=["DDPM", "Flow Matching"],
            state="readonly",
            width=16,
        ).pack(fill="x", pady=(2, 12))
        self._model_type_var.trace_add("write", lambda *_: self.after(0, self._on_model_type_changed))

        # Station dropdown
        ttk.Label(ctrl, text="Station").pack(anchor="w")
        self._station_var = tk.StringVar(value="EDC")
        self._station_cb = ttk.Combobox(
            ctrl,
            textvariable=self._station_var,
            values=STATION_NAMES,
            state="readonly",
            width=16,
        )
        self._station_cb.pack(fill="x", pady=(2, 12))
        self._station_var.trace_add("write", lambda *_: self.after(0, self._update_station_map))

        # Sliders: (label, key, lo, hi, default, fmt)
        sliders = [
            ("Magnitude",        "mag",      1.0,   5.7,   3.0,  ".1f"),
            ("Latitude",         "lat", MARMARA_LAT_MIN, MARMARA_LAT_MAX, 40.70, ".3f"),
            ("Longitude",        "lon", MARMARA_LON_MIN, MARMARA_LON_MAX, 28.00, ".3f"),
            ("Depth  (km)",      "dep",      0.0,  29.7,  10.0,  ".1f"),
            ("SNR",              "snr",      1.0,  30.0,   5.0,  ".1f"),
            ("Guidance Scale",   "cfg",      1.0,  10.0,   3.0,  ".1f"),
        ]
        self._svars = {}
        for label, key, lo, hi, default, fmt in sliders:
            ttk.Label(ctrl, text=label).pack(anchor="w", pady=(8, 0))
            var = tk.DoubleVar(value=default)
            self._svars[key] = var
            row = ttk.Frame(ctrl)
            row.pack(fill="x")
            ttk.Scale(row, from_=lo, to=hi, variable=var,
                      orient="horizontal", length=190).pack(side="left", fill="x", expand=True)
            val_lbl = ttk.Label(row, text=format(default, fmt), width=7)
            val_lbl.pack(side="left")
            var.trace_add("write",
                lambda *_, v=var, l=val_lbl, f=fmt: l.config(text=format(v.get(), f)))
            if key in {"lat", "lon"}:
                var.trace_add("write", lambda *_: self.after(0, self._update_station_map))

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=10)
        self._build_griffin_lim_controls(ctrl)
        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=10)

        self._rand_btn = ttk.Button(ctrl, text="🎲  Random EQ",
                                    command=self._on_random_eq, state="disabled")
        self._rand_btn.pack(fill="x", pady=(0, 4))

        self._gen_btn = ttk.Button(ctrl, text="⚡  Generate",
                                   command=self._on_generate, state="disabled")
        self._gen_btn.pack(fill="x", ipady=8)

        self._status_lbl = ttk.Label(ctrl, text="Loading models…",
                                     foreground=YELLOW, wraplength=210, justify="left")
        self._status_lbl.pack(pady=10)

    def _build_griffin_lim_controls(self, parent):
        gl_frame = ttk.LabelFrame(parent, text="Griffin-Lim Params", padding=8)
        gl_frame.pack(fill="x")
        gl_frame.columnconfigure(1, weight=1)

        self._gl_vars = {
            "n_iter": tk.IntVar(value=900),
            "momentum": tk.DoubleVar(value=0.9),
            "inv_log_gain": tk.DoubleVar(value=3.0),
            "hop_length": tk.IntVar(value=32),
            "win_length": tk.IntVar(value=1024),
            "n_fft": tk.IntVar(value=1024),
            "window": tk.StringVar(value="bartlett"),
            "center": tk.BooleanVar(value=GRIFFIN_LIM_CENTER),
            "random_state": tk.IntVar(value=GRIFFIN_LIM_RANDOM_STATE),
            "use_length": tk.BooleanVar(value=False),
            "length": tk.IntVar(value=0),
        }

        row = 0
        ttk.Label(gl_frame, text="Iterations").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Spinbox(gl_frame, from_=1, to=4000, increment=1, width=10,
                    textvariable=self._gl_vars["n_iter"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(gl_frame, text="Momentum").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Spinbox(gl_frame, from_=0.0, to=0.999, increment=0.01, width=10,
                    textvariable=self._gl_vars["momentum"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(gl_frame, text="Inv-Log Gain").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Spinbox(gl_frame, from_=0.1, to=20.0, increment=0.1, width=10,
                    textvariable=self._gl_vars["inv_log_gain"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(gl_frame, text="Hop Length").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Spinbox(gl_frame, from_=1, to=4096, increment=1, width=10,
                    textvariable=self._gl_vars["hop_length"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(gl_frame, text="Win Length").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Spinbox(gl_frame, from_=1, to=4096, increment=1, width=10,
                    textvariable=self._gl_vars["win_length"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(gl_frame, text="N FFT").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Spinbox(gl_frame, from_=2, to=8192, increment=1, width=10,
                    textvariable=self._gl_vars["n_fft"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(gl_frame, text="Window").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(
            gl_frame,
            textvariable=self._gl_vars["window"],
            values=["hann", "hamming", "blackman", "bartlett", "boxcar"],
            state="readonly",
            width=10,
        ).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Label(gl_frame, text="Random State").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Spinbox(gl_frame, from_=-1, to=100000, increment=1, width=10,
                    textvariable=self._gl_vars["random_state"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1

        ttk.Checkbutton(gl_frame, text="Center", variable=self._gl_vars["center"]).grid(
            row=row, column=0, sticky="w", pady=2
        )
        ttk.Checkbutton(gl_frame, text="Use Length", variable=self._gl_vars["use_length"]).grid(
            row=row, column=1, sticky="w", pady=2
        )
        row += 1

        ttk.Label(gl_frame, text="Length").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Spinbox(gl_frame, from_=0, to=200000, increment=1, width=10,
                    textvariable=self._gl_vars["length"]).grid(row=row, column=1, sticky="ew", pady=2)
        row += 1
        self._gl_update_btn = ttk.Button(
            gl_frame,
            text="Update Waveform",
            command=self._start_griffin_lim_refresh,
        )
        self._gl_update_btn.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _sync_griffin_lim_defaults_from_stft(self):
        pass

    def _get_griffin_lim_params_from_ui(self) -> dict:
        if not hasattr(self, "_gl_vars"):
            return {}

        n_iter = max(1, int(self._gl_vars["n_iter"].get()))
        momentum = float(np.clip(float(self._gl_vars["momentum"].get()), 0.0, 0.999))
        inv_log_gain = float(np.clip(float(self._gl_vars["inv_log_gain"].get()), 0.1, 20.0))
        hop_length = max(1, int(self._gl_vars["hop_length"].get()))
        win_length = max(1, int(self._gl_vars["win_length"].get()))
        n_fft = max(2, int(self._gl_vars["n_fft"].get()))
        window = str(self._gl_vars["window"].get())
        center = bool(self._gl_vars["center"].get())
        random_state_val = int(self._gl_vars["random_state"].get())
        random_state = None if random_state_val < 0 else random_state_val

        if win_length > n_fft:
            win_length = n_fft

        length = None
        if bool(self._gl_vars["use_length"].get()):
            length = int(self._gl_vars["length"].get())
            if length <= 0:
                length = None

        return {
            "n_iter": n_iter,
            "momentum": momentum,
            "inv_log_gain": inv_log_gain,
            "hop_length": hop_length,
            "win_length": win_length,
            "n_fft": n_fft,
            "window": window,
            "center": center,
            "random_state": random_state,
            "length": length,
        }

    def _start_griffin_lim_refresh(self):
        if self._latest_spec is None:
            self._set_status("Generate a sample first, then click Update Waveform.", YELLOW)
            return
        try:
            params = self._get_griffin_lim_params_from_ui()
        except Exception as exc:
            self._set_status(f"Invalid Griffin-Lim params:\n{exc}", RED)
            return

        spec = np.array(self._latest_spec, copy=True)
        self._gl_recompute_token += 1
        token = self._gl_recompute_token
        threading.Thread(
            target=self._recompute_waveforms_worker,
            args=(spec, params, token),
            daemon=True,
        ).start()

    def _recompute_waveforms_worker(self, spec: np.ndarray, params: dict, token: int):
        try:
            waves = np.stack(
                [griffin_lim_channel(spec[ch], params=params) for ch in range(3)],
                axis=0,
            )
            self.after(0, lambda w=waves, t=token: self._apply_recomputed_waveforms(w, t))
        except Exception as exc:
            self.after(0, lambda: self._set_status(f"Griffin-Lim error:\n{exc}", RED))

    def _apply_recomputed_waveforms(self, waves: np.ndarray, token: int):
        if token != self._gl_recompute_token:
            return
        self._draw_waveforms(waves)
        if not self._generating:
            self._set_status("Waveform updated from current Griffin-Lim params", GREEN)

    def _build_plots(self, parent):
        plots = ttk.Frame(parent)
        plots.grid(row=0, column=1, sticky="nsew")
        plots.rowconfigure(0, weight=1)   # map
        plots.rowconfigure(1, weight=2)   # reference row
        plots.rowconfigure(2, weight=2)   # generated row
        plots.columnconfigure(0, weight=1)  # STFT column
        plots.columnconfigure(1, weight=2)  # waveform column

        # ── Map ────────────────────────────────────────────────────────────────
        map_frame = ttk.LabelFrame(plots, text="Marmara Stations & Earthquake", padding=4)
        map_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", pady=(0, 6))

        self._map_fig, self._map_ax = plt.subplots(figsize=(11, 2.8), facecolor=BG)
        self._style_ax(self._map_ax)
        self._map_ax.set_xlabel("Longitude", color=SUBTEXT, fontsize=8)
        self._map_ax.set_ylabel("Latitude", color=SUBTEXT, fontsize=8)
        self._map_ax.set_title("Marmara Region", color=TEXT, fontsize=10)
        self._map_canvas = FigureCanvasTkAgg(self._map_fig, map_frame)
        self._map_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Reference row ─────────────────────────────────────────────────────
        ref_frame = ttk.LabelFrame(plots, text="Reference STFT  (E / N / Z)", padding=4)
        ref_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 6), padx=(0, 6))

        self._ref_fig, self._ref_axes = plt.subplots(1, 3, figsize=(5.5, 2.8), facecolor=BG)
        self._ref_fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.14, wspace=0.3)
        for ax, ch in zip(self._ref_axes, ["E", "N", "Z"]):
            self._style_ax(ax)
            ax.set_title(ch, color=SUBTEXT, fontsize=10)
            ax.set_xlabel("Time bins", color=SUBTEXT, fontsize=8)
            ax.set_ylabel("Freq bins", color=SUBTEXT, fontsize=8)
        self._ref_canvas = FigureCanvasTkAgg(self._ref_fig, ref_frame)
        self._ref_canvas.get_tk_widget().pack(fill="both", expand=True)

        ref_wav_frame = ttk.LabelFrame(plots, text="Reference Waveform  (E / N / Z)", padding=4)
        ref_wav_frame.grid(row=1, column=1, sticky="nsew", pady=(0, 6))

        self._ref_wav_fig, self._ref_wav_axes = plt.subplots(3, 1, figsize=(6, 2.8), facecolor=BG, sharex=True)
        self._ref_wav_fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.10, hspace=0.08)
        for ax, ch in zip(self._ref_wav_axes, ["E", "N", "Z"]):
            self._style_ax(ax)
            ax.set_ylabel(ch, color=TEXT, fontsize=9)
        self._ref_wav_axes[-1].set_xlabel("Time (s)", color=TEXT, fontsize=9)
        self._ref_wav_canvas = FigureCanvasTkAgg(self._ref_wav_fig, ref_wav_frame)
        self._ref_wav_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Generated row ─────────────────────────────────────────────────────
        spec_frame = ttk.LabelFrame(plots, text="Generated STFT  (E / N / Z)", padding=4)
        spec_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 6))

        self._spec_fig, self._spec_axes = plt.subplots(1, 3, figsize=(5.5, 2.8), facecolor=BG)
        self._spec_fig.subplots_adjust(left=0.04, right=0.99, top=0.84, bottom=0.14, wspace=0.3)
        for ax, ch in zip(self._spec_axes, ["E", "N", "Z"]):
            self._style_ax(ax)
            ax.set_title(ch, color=TEXT, fontsize=10)
            ax.set_xlabel("Time bins", color=SUBTEXT, fontsize=8)
            ax.set_ylabel("Freq bins", color=SUBTEXT, fontsize=8)
        self._spec_canvas = FigureCanvasTkAgg(self._spec_fig, spec_frame)
        self._spec_canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Waveforms ─────────────────────────────────────────────────────────
        wav_frame = ttk.LabelFrame(plots, text="Reconstructed Waveform via Griffin-Lim  (E / N / Z)", padding=4)
        wav_frame.grid(row=2, column=1, sticky="nsew")

        self._wav_fig, self._wav_axes = plt.subplots(
            3, 1, figsize=(11, 3.8), facecolor=BG, sharex=True)
        self._wav_fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.10, hspace=0.08)
        for ax, ch in zip(self._wav_axes, ["E", "N", "Z"]):
            self._style_ax(ax)
            ax.set_ylabel(ch, color=TEXT, fontsize=9)
        self._wav_axes[-1].set_xlabel("Time (s)", color=TEXT, fontsize=9)
        self._wav_canvas = FigureCanvasTkAgg(self._wav_fig, wav_frame)
        self._wav_canvas.get_tk_widget().pack(fill="both", expand=True)
        self._update_station_map()

    @staticmethod
    def _style_ax(ax):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=SUBTEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(SUBTEXT)

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_models(self):
        try:
            scale = _load_scale()
            self._emb_std        = float(scale.get("emb_std", 1.0))
            self._emb_mean       = float(scale.get("emb_mean", 0.0))
            self._data_mode      = str(scale.get("data_mode", "latent")).lower()
            self._normalise_cond = _make_normalise_cond(scale)
            self._station_locations = _load_station_locations()

            meta_path = DIFF / "embeddings" / "metadata.json"
            if meta_path.exists():
                self._train_metadatas = json.load(open(meta_path))
                print(f"[demo] Loaded {len(self._train_metadatas):,} training metadata entries")

            available_stations = [s for s in STATION_NAMES if s in self._station_locations]
            missing_stations = [s for s in STATION_NAMES if s not in self._station_locations]
            if missing_stations:
                print(
                    "[demo] Warning: station locations missing for: "
                    + ", ".join(missing_stations)
                )
            if available_stations:
                def _update_station_choices():
                    self._station_cb.configure(values=available_stations)
                    if self._station_var.get() not in available_stations:
                        self._station_var.set(available_stations[0])
                    self._update_station_map()
                self.after(0, _update_station_choices)

            # ── Diffusion model (latest checkpoint) ──────────────────────────
            diff_ckpt, diff_cfg = _find_latest_diffusion_checkpoint()
            if diff_ckpt is None:
                raise FileNotFoundError(
                    "No diffusion checkpoint found.\n"
                    "Train it first with ML/diffusion/train.py")
            self._set_status(f"Loading diffusion  ({diff_ckpt.name})…", YELLOW)
            station_emb_path = diff_ckpt / "station_embedding.pt"
            if station_emb_path.exists():
                self._diff_unet = DiffusionUNet2D.load_pretrained(diff_ckpt).to(DEVICE)
                print(f"[demo] Loaded diffusion wrapper with station embedding: {station_emb_path.name}")
            else:
                # Legacy checkpoints without station embedding metadata.
                from diffusers import UNet2DConditionModel
                self._diff_unet = UNet2DConditionModel.from_pretrained(str(diff_ckpt)).to(DEVICE)
                print("[demo] Loaded legacy UNet checkpoint (no station_embedding.pt).")
            self._scheduler = _load_diffusion_scheduler(diff_ckpt)

            train_cfg = _read_diffusion_training_config(diff_ckpt)
            self._data_mode     = str(train_cfg.get("data_mode",     self._data_mode)).lower()
            self._training_type = str(train_cfg.get("training_type", self._training_type)).lower()
            if self._data_mode not in {"latent", "stft"}:
                self._data_mode = "latent"
            if self._training_type not in {"ddpm", "flow_matching"}:
                self._training_type = "ddpm"
            _display = "Flow Matching" if self._training_type == "flow_matching" else "DDPM"
            self.after(0, lambda d=_display: self._model_type_var.set(d))

            # ── Data-mode specific setup ──────────────────────────────────────
            if self._data_mode == "latent":
                ae_ckpt = _get_embeddings_source_checkpoint() or _find_latest_timestamped_ae_checkpoint()
                if ae_ckpt is None:
                    raise FileNotFoundError(
                        "No autoencoder checkpoint found.\n"
                        "Train it first with ML/autoencoder/train.py"
                    )
                self._set_status(f"Loading AE  ({ae_ckpt.parent.name})…", YELLOW)
                self._ae_model, ae_config = _load_ae(str(ae_ckpt), device=DEVICE)
                self._ae_model.eval()
                ae_latent_channels = int(getattr(self._ae_model, "latent_channels", -1))
                if ae_latent_channels <= 0:
                    raise ValueError("Could not determine AE latent_channels from checkpoint.")

                stft_cfg = _apply_ae_stft_config(ae_config)
                print(
                    "[demo] Using STFT params from AE checkpoint: "
                    f"nperseg={stft_cfg['nperseg']}, "
                    f"noverlap={stft_cfg['noverlap']}, "
                    f"nfft={stft_cfg['nfft']} "
                    f"(freq_bins={stft_cfg['freq_bins']})"
                )
                print(f"[demo] AE latent_channels={ae_latent_channels}")
                self.after(0, self._sync_griffin_lim_defaults_from_stft)

                if "data_shape" in train_cfg:
                    self._emb_shape = tuple(int(v) for v in train_cfg["data_shape"])
                else:
                    emb_path = DIFF / "embeddings" / "embeddings.pt"
                    if emb_path.exists():
                        emb = torch.load(emb_path, map_location="cpu")
                        self._emb_shape = tuple(emb.shape[1:])
                    else:
                        self._emb_shape = (ae_latent_channels, 17, 14)

                if self._emb_shape[0] != ae_latent_channels:
                    raise FileNotFoundError(
                        "Diffusion latent channel count is incompatible with AE.\n"
                        f"Diffusion expects C={self._emb_shape[0]} but AE has C={ae_latent_channels}."
                    )
            else:
                self._ae_model = None
                stft_cfg = _get_embeddings_source_stft_config()
                _apply_ae_stft_config(stft_cfg)
                print(
                    "[demo] Using STFT-mode diffusion data shape + source STFT params: "
                    f"nperseg={stft_cfg['nperseg']}, "
                    f"noverlap={stft_cfg['noverlap']}, "
                    f"nfft={stft_cfg['nfft']}"
                )
                self.after(0, self._sync_griffin_lim_defaults_from_stft)

                if "data_shape" in train_cfg:
                    self._emb_shape = tuple(int(v) for v in train_cfg["data_shape"])
                elif "data_shape" in scale:
                    self._emb_shape = tuple(int(v) for v in scale["data_shape"])
                else:
                    in_ch = int((diff_cfg or {}).get("in_channels", 3))
                    self._emb_shape = (in_ch, FREQ_BINS, 111)

            diff_cfg = diff_cfg or {}
            print(
                "[demo] Using diffusion UNet config: "
                f"in_channels={diff_cfg.get('in_channels')}, "
                f"out_channels={diff_cfg.get('out_channels')}, "
                f"cross_attention_dim={diff_cfg.get('cross_attention_dim')}"
            )
            print(
                f"[demo] data_mode={self._data_mode}, "
                f"sample_shape={self._emb_shape}, emb_mean={self._emb_mean:.5f}, emb_std={self._emb_std:.5f}"
            )

            self._set_status("Models ready ✓", GREEN)
            self.after(0, lambda: self._gen_btn.config(state="normal"))
            self.after(0, lambda: self._rand_btn.config(state="normal"))

        except FileNotFoundError as exc:
            self._set_status(str(exc), RED)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._set_status(f"Load error:\n{exc}", RED)

    def _set_status(self, msg: str, colour: str = TEXT):
        self.after(0, lambda: self._status_lbl.config(text=msg, foreground=colour))

    def _draw_reference_stft(self, spec: np.ndarray, title: str = ""):
        for ax, channel, col in zip(self._ref_axes, spec, CH_COLS):
            ax.cla()
            self._style_ax(ax)
            ax.imshow(channel, origin="lower", aspect="auto", cmap="inferno")
            ax.set_xlabel("Time bins", color=SUBTEXT, fontsize=8)
            ax.set_ylabel("Freq bins", color=SUBTEXT, fontsize=8)
        self._ref_fig.suptitle(title, color=TEXT, fontsize=9)
        self._ref_canvas.draw()

    def _load_reference_data(self, file_path: str):
        """Returns (spec (3,F,T), waves (3,N), sampling_rate)."""
        from obspy import read as obspy_read
        from scipy import signal as sp_signal

        path = (DIFF / file_path).resolve()
        stream = obspy_read(str(path))
        stream.sort(keys=["channel"])

        stft_channels = []
        wave_channels = []
        sr = stream[0].stats.sampling_rate
        for trace in stream:
            data = trace.data.astype(np.float32)
            wave_channels.append(data)
            _, _, zxx = sp_signal.stft(
                data,
                fs=trace.stats.sampling_rate,
                nperseg=NPERSEG,
                noverlap=NOVERLAP,
                nfft=NFFT,
                return_onesided=True,
                boundary="zeros",
                padded=True,
            )
            mag = np.log1p(np.abs(zxx))
            mag_min, mag_max = mag.min(), mag.max()
            if mag_max > mag_min:
                mag = (mag - mag_min) / (mag_max - mag_min)
            else:
                mag = np.zeros_like(mag)
            stft_channels.append(mag)

        spec  = np.stack(stft_channels, axis=0)   # (3, F, T)
        waves = np.stack(wave_channels,  axis=0)   # (3, N)
        return spec, waves, sr

    def _draw_reference_waveform(self, waves: np.ndarray, sr: float):
        n = waves.shape[1]
        t = np.linspace(0.0, n / sr, num=n, endpoint=False)
        for ax, wave, ch, col in zip(self._ref_wav_axes, waves, ["E", "N", "Z"], CH_COLS):
            ax.cla()
            self._style_ax(ax)
            ax.plot(t, wave, color=col, lw=0.8)
            ax.axhline(0, color=SUBTEXT, lw=0.5, ls="--")
            ax.set_ylabel(ch, color=TEXT, fontsize=9)
            ax.set_xlim(t[0], t[-1])
        self._ref_wav_axes[-1].set_xlabel("Time (s)", color=TEXT, fontsize=9)
        self._ref_wav_canvas.draw()

    def _on_random_eq(self):
        import random
        if not self._train_metadatas:
            self._set_status("No training metadata loaded.", RED)
            return
        m = random.choice(self._train_metadatas)
        self._svars["mag"].set(float(m["magnitude"]))
        self._svars["lat"].set(float(m["latitude"]))
        self._svars["lon"].set(float(m["longitude"]))
        self._svars["dep"].set(float(m["depth"]))
        if "snr" in m and "snr" in self._svars:
            self._svars["snr"].set(float(m["snr"]))
        station = m.get("station_name", "")
        if station in STATION_NAMES and station in self._station_locations:
            self._station_var.set(station)

        try:
            spec, waves, sr = self._load_reference_data(m["file_path"])
            title = (f"M{m['magnitude']:.1f}  {m.get('location_name', '')}  "
                     f"station={station}  SNR={m.get('snr', 0):.1f}")
            self._draw_reference_stft(spec, title)
            self._draw_reference_waveform(waves, sr)
        except Exception as exc:
            self._set_status(f"Could not load reference data:\n{exc}", RED)

    # ── Model type switching ──────────────────────────────────────────────────
    def _on_model_type_changed(self):
        selected = self._model_type_var.get()
        training_type = "flow_matching" if selected == "Flow Matching" else "ddpm"
        if training_type == self._training_type:
            return
        self._set_status(f"Loading {selected} model…", YELLOW)
        self.after(0, lambda: self._gen_btn.config(state="disabled"))
        self.after(0, lambda: self._rand_btn.config(state="disabled"))
        threading.Thread(
            target=self._reload_diffusion_model,
            args=(training_type,),
            daemon=True,
        ).start()

    def _reload_diffusion_model(self, training_type: str):
        try:
            diff_ckpt, _ = _find_latest_checkpoint_by_type(training_type)
            if diff_ckpt is None:
                self._set_status(
                    f"No {training_type} checkpoint found.\n"
                    f"Train with: python train.py --training_type {training_type}",
                    RED,
                )
                _restore = "Flow Matching" if self._training_type == "flow_matching" else "DDPM"
                self.after(0, lambda d=_restore: self._model_type_var.set(d))
                self.after(0, lambda: self._gen_btn.config(state="normal"))
                self.after(0, lambda: self._rand_btn.config(state="normal"))
                return

            self._set_status(f"Loading {training_type} ({diff_ckpt.name})…", YELLOW)
            station_emb_path = diff_ckpt / "station_embedding.pt"
            if station_emb_path.exists():
                self._diff_unet = DiffusionUNet2D.load_pretrained(diff_ckpt).to(DEVICE)
            else:
                from diffusers import UNet2DConditionModel
                self._diff_unet = UNet2DConditionModel.from_pretrained(str(diff_ckpt)).to(DEVICE)
            self._scheduler = _load_diffusion_scheduler(diff_ckpt)

            train_cfg = _read_diffusion_training_config(diff_ckpt)
            self._training_type = str(train_cfg.get("training_type", training_type)).lower()
            new_data_mode = str(train_cfg.get("data_mode", self._data_mode)).lower()
            if new_data_mode not in {"latent", "stft"}:
                new_data_mode = self._data_mode
            if "data_shape" in train_cfg:
                self._emb_shape = tuple(int(v) for v in train_cfg["data_shape"])

            # Load AE if the new checkpoint needs latent decoding and we don't have one
            if new_data_mode == "latent" and self._ae_model is None:
                ae_ckpt = _get_embeddings_source_checkpoint() or _find_latest_timestamped_ae_checkpoint()
                if ae_ckpt is None:
                    raise FileNotFoundError(
                        "No autoencoder checkpoint found.\n"
                        "Train it first with ML/autoencoder/train.py"
                    )
                self._set_status(f"Loading AE ({ae_ckpt.parent.name})…", YELLOW)
                self._ae_model, _ = _load_ae(str(ae_ckpt), device=DEVICE)
                self._ae_model.eval()
            elif new_data_mode == "stft":
                self._ae_model = None

            self._data_mode = new_data_mode
            print(f"[demo] Switched to {self._training_type} model: {diff_ckpt.name}")
            self._set_status(
                f"{self._model_type_var.get()} model loaded ✓  ({diff_ckpt.name})", GREEN
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._set_status(f"Load error:\n{exc}", RED)
        finally:
            self.after(0, lambda: self._gen_btn.config(state="normal"))
            self.after(0, lambda: self._rand_btn.config(state="normal"))

    # ── Generate callback ─────────────────────────────────────────────────────
    def _on_generate(self):
        if self._generating:
            return
        station_name = self._station_var.get()
        if station_name not in self._station_locations:
            self._set_status(
                f"Station '{station_name}' has no coordinates in station_locations.json.\n"
                "Run fetch_station_locations.py and reload demo.",
                RED,
            )
            return
        station_idx  = STATION_NAMES.index(station_name)
        self._cond_meta_at_generate = {
            "magnitude":   self._svars["mag"].get(),
            "latitude":    self._svars["lat"].get(),
            "longitude":   self._svars["lon"].get(),
            "depth":       self._svars["dep"].get(),
            "snr":         self._svars["snr"].get(),
            "station_name": station_name,
            "station_idx": station_idx,
        }
        self._title_at_generate = (
            f"M{self._cond_meta_at_generate['magnitude']:.1f}  "
            f"lat={self._cond_meta_at_generate['latitude']:.3f}  "
            f"lon={self._cond_meta_at_generate['longitude']:.3f}  "
            f"depth={self._cond_meta_at_generate['depth']:.1f} km  "
            f"SNR={self._cond_meta_at_generate['snr']:.1f}  "
            f"station={station_name}"
        )
        self._gl_params_at_generate = self._get_griffin_lim_params_from_ui()
        self._generating = True
        self._gen_btn.config(state="disabled")
        self._set_status("Running diffusion…", TEAL)
        threading.Thread(target=self._run_generation, daemon=True).start()

    def _run_generation(self):
        try:
            cond_meta = dict(self._cond_meta_at_generate) if self._cond_meta_at_generate is not None else {}
            suptitle = self._title_at_generate
            gl_params = dict(self._gl_params_at_generate) if self._gl_params_at_generate is not None else {}

            def on_step(step, total, spec, waves):
                pct = int(100 * step / total)
                self._set_status(f"Diffusing\u2026 {pct}%", TEAL)
                # Schedule spec update on the main thread.
                # Capture spec/waves by value to avoid data-race.
                self.after(0, lambda s=spec, w=waves, title=suptitle:
                           self._live_update(s, w, title))

            generate(
                self._diff_unet, self._ae_model, self._scheduler,
                self._emb_shape, self._normalise_cond, self._emb_std,
                self._station_locations,
                cond_meta,
                step_callback=on_step,
                update_every=50,
                griffin_lim_params=gl_params,
                data_mode=self._data_mode,
                emb_mean=self._emb_mean,
                guidance_scale=float(self._svars["cfg"].get()),
                training_type=self._training_type,
            )
            self._set_status("Done ✓", GREEN)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._set_status(f"Error:\n{exc}", RED)
        finally:
            self._generating = False
            self.after(0, lambda: self._gen_btn.config(state="normal"))

    # ── Plot updates ──────────────────────────────────────────────────────────
    def _draw_waveforms(self, waves: np.ndarray):
        # Keep the displayed time axis fixed to 70 s regardless of reconstruction params/length.
        t = np.linspace(0.0, WAVEFORM_DISPLAY_SECONDS, num=waves.shape[1], endpoint=False)
        for ax, wave, ch, col in zip(self._wav_axes, waves, ["E", "N", "Z"], CH_COLS):
            ax.cla()
            self._style_ax(ax)
            ax.plot(t, wave, color=col, lw=0.8)
            ax.axhline(0, color=SUBTEXT, lw=0.5, ls="--")
            ax.set_ylabel(ch, color=TEXT, fontsize=9)
            ax.set_xlim(0.0, WAVEFORM_DISPLAY_SECONDS)
        self._wav_axes[-1].set_xlabel("Time (s)", color=TEXT, fontsize=9)
        self._wav_canvas.draw()

    def _update_station_map(self):
        if self._map_ax is None or self._map_canvas is None:
            return

        ax = self._map_ax
        ax.cla()
        self._style_ax(ax)
        ax.set_title("Marmara Region", color=TEXT, fontsize=10)
        ax.set_xlabel("Longitude", color=SUBTEXT, fontsize=8)
        ax.set_ylabel("Latitude", color=SUBTEXT, fontsize=8)
        ax.set_xlim(MARMARA_LON_MIN, MARMARA_LON_MAX)
        ax.set_ylim(MARMARA_LAT_MIN, MARMARA_LAT_MAX)
        ax.grid(True, color=OVERLAY, alpha=0.35, lw=0.6)
        ax.set_aspect("equal", adjustable="box")

        selected_station = self._station_var.get() if hasattr(self, "_station_var") else None
        eq_lat = self._svars["lat"].get() if hasattr(self, "_svars") else None
        eq_lon = self._svars["lon"].get() if hasattr(self, "_svars") else None

        station_pts = []
        for name in STATION_NAMES:
            meta = self._station_locations.get(name)
            if not meta:
                continue
            lat = float(meta["latitude"])
            lon = float(meta["longitude"])
            if MARMARA_LAT_MIN <= lat <= MARMARA_LAT_MAX and MARMARA_LON_MIN <= lon <= MARMARA_LON_MAX:
                station_pts.append((name, lat, lon))

        if station_pts:
            lats = [p[1] for p in station_pts]
            lons = [p[2] for p in station_pts]
            ax.scatter(
                lons,
                lats,
                s=28,
                c=TEAL,
                alpha=0.9,
                edgecolors=BG,
                linewidths=0.6,
                label=f"Stations ({len(station_pts)})",
                zorder=2,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "Loading station map…",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=SUBTEXT,
                fontsize=9,
            )

        sel_lat = None
        sel_lon = None
        if selected_station is not None:
            for name, lat, lon in station_pts:
                if name == selected_station:
                    sel_lat, sel_lon = lat, lon
                    break

        if sel_lat is not None and sel_lon is not None:
            ax.scatter(
                [sel_lon],
                [sel_lat],
                s=120,
                marker="D",
                c=YELLOW,
                edgecolors=BG,
                linewidths=1.1,
                label=f"Selected: {selected_station}",
                zorder=4,
            )
            ax.text(
                sel_lon + 0.02,
                sel_lat + 0.02,
                selected_station,
                color=YELLOW,
                fontsize=8,
                zorder=5,
            )

        if eq_lat is not None and eq_lon is not None:
            ax.scatter(
                [eq_lon],
                [eq_lat],
                s=180,
                marker="*",
                c=RED,
                edgecolors=TEXT,
                linewidths=0.8,
                label="EQ location",
                zorder=6,
            )
            if sel_lat is not None and sel_lon is not None:
                ax.plot(
                    [sel_lon, eq_lon],
                    [sel_lat, eq_lat],
                    color=TEXT,
                    lw=0.9,
                    ls="--",
                    alpha=0.8,
                    zorder=3,
                )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc="lower left",
                fontsize=7,
                facecolor=BG,
                edgecolor=OVERLAY,
                labelcolor=TEXT,
            )
        self._map_canvas.draw_idle()

    def _live_update(self, spec: np.ndarray, waves, suptitle: str):
        """Called on the main thread for every preview update."""
        self._latest_spec = np.array(spec, copy=True)
        self._latest_title = suptitle

        # Always refresh the spectrogram
        for ax, img, ch in zip(self._spec_axes, spec, ["E", "N", "Z"]):
            ax.cla()
            self._style_ax(ax)
            ax.imshow(img, aspect="auto", origin="lower",
                      cmap="inferno", interpolation="nearest")
            ax.set_title(ch, color=TEXT, fontsize=10)
            ax.set_xlabel("Time bins", color=SUBTEXT, fontsize=8)
            ax.set_ylabel("Freq bins", color=SUBTEXT, fontsize=8)
        self._spec_fig.suptitle(suptitle, color=TEXT, fontsize=9)
        self._spec_canvas.draw()

        # Only draw waveforms on the final step (when waves is provided)
        if waves is not None:
            self._draw_waveforms(waves)



# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SeismicDemoApp()
    app.mainloop()
