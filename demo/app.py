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

from ML.diffusion.model import create_conditioning_vector, NUM_STATIONS
from ML.autoencoder.inference import load_model as _load_ae
from diffusers import DDPMScheduler, UNet2DConditionModel

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
NPERSEG = 256
NOVERLAP= 192
NFFT    = 256
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

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


# ── Griffin-Lim reconstruction ───────────────────────────────────────────────
def griffin_lim_channel(magnitude: np.ndarray, n_iter: int = 60) -> np.ndarray:
    """
    Reconstruct one channel waveform from a magnitude STFT via Griffin-Lim.

    Parameters
    ----------
    magnitude : (freq_bins, time_bins) float32
        Normalised log-magnitude spectrogram (the raw VAE decoder output).
    """
    # Invert the per-sample log1p normalisation that SeismicSTFTDataset applies.
    # The original per-sample min/max is not stored, so we rescale the [0,1]
    # output to a reasonable amplitude range then apply expm1.
    mag = np.expm1(np.clip(magnitude, 0.0, None).astype(np.float64))

    return librosa.griffinlim(
        mag,
        n_iter=n_iter,
        hop_length=NPERSEG - NOVERLAP,
        win_length=NPERSEG,
        n_fft=NFFT,
        window="hann",
        center=True,
        momentum=0.99,
        random_state=0,
    ).astype(np.float32)


# ── Generation pipeline ──────────────────────────────────────────────────────
FREQ_BINS = NFFT // 2 + 1   # 129 — required by Griffin-Lim


def _decode_to_spec(ae_model, x: torch.Tensor, emb_std: float) -> np.ndarray:
    """Decode a latent batch tensor to a (3, FREQ_BINS, W) numpy spectrogram."""
    embedding = (x * emb_std).squeeze(0)
    with torch.no_grad():
        spec_t = ae_model.decode(embedding.unsqueeze(0))[0].cpu()
    spec_t = spec_t.clamp(min=0.0)[:, :FREQ_BINS, :]
    return spec_t.numpy()


@torch.no_grad()
def generate(diff_unet, ae_model, scheduler, emb_shape,
             normalise_cond, emb_std, cond_meta: dict,
             step_callback=None, update_every: int = 50):
    """
    Full pipeline: conditioning → diffusion sampling → VAE decode → Griffin-Lim.

    step_callback(step, total, spec, waves_or_None) is called every `update_every`
    diffusion steps and always on the very last step (waves is only provided then).
    """
    # Build conditioning vector (raw, then normalise)
    cond_vec  = create_conditioning_vector(cond_meta)
    cond_norm = normalise_cond(cond_vec).unsqueeze(0).unsqueeze(0).to(DEVICE)

    x = torch.randn(1, *emb_shape, device=DEVICE)
    scheduler.set_timesteps(NUM_TRAIN_TIMESTEPS)
    diff_unet.eval()

    timesteps = scheduler.timesteps
    total     = len(timesteps)

    for i, t in enumerate(timesteps):
        t_b        = t.unsqueeze(0).to(DEVICE)
        noise_pred = diff_unet(x, t_b, encoder_hidden_states=cond_norm).sample
        x          = scheduler.step(noise_pred, t, x).prev_sample

        if step_callback is not None:
            is_last = (i == total - 1)
            if i % update_every == 0 or is_last:
                spec  = _decode_to_spec(ae_model, x, emb_std)
                waves = (np.stack([griffin_lim_channel(spec[ch])
                                   for ch in range(3)], axis=0)
                         if is_last else None)
                step_callback(i + 1, total, spec, waves)

    # Also return the final result for non-callback callers
    spec  = _decode_to_spec(ae_model, x, emb_std)
    waves = np.stack([griffin_lim_channel(spec[ch]) for ch in range(3)], axis=0)
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
        self._normalise_cond = lambda v: v
        self._generating    = False

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

        # Station dropdown
        ttk.Label(ctrl, text="Station").pack(anchor="w")
        self._station_var = tk.StringVar(value="EDC")
        cb = ttk.Combobox(ctrl, textvariable=self._station_var,
                          values=STATION_NAMES, state="readonly", width=16)
        cb.pack(fill="x", pady=(2, 12))

        # Sliders: (label, key, lo, hi, default, fmt)
        sliders = [
            ("Magnitude",   "mag", 1.0,   5.7,   3.0,  ".1f"),
            ("Latitude",    "lat", 39.75, 41.53, 40.70, ".3f"),
            ("Longitude",   "lon", 25.80, 29.99, 28.00, ".3f"),
            ("Depth  (km)", "dep", 0.0,   29.7,  10.0,  ".1f"),
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

        ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=16)

        self._gen_btn = ttk.Button(ctrl, text="⚡  Generate",
                                   command=self._on_generate, state="disabled")
        self._gen_btn.pack(fill="x", ipady=8)

        self._status_lbl = ttk.Label(ctrl, text="Loading models…",
                                     foreground=YELLOW, wraplength=210, justify="left")
        self._status_lbl.pack(pady=10)

    def _build_plots(self, parent):
        plots = ttk.Frame(parent)
        plots.grid(row=0, column=1, sticky="nsew")
        plots.rowconfigure(0, weight=2)
        plots.rowconfigure(1, weight=3)
        plots.columnconfigure(0, weight=1)

        # ── Spectrogram ───────────────────────────────────────────────────────
        spec_frame = ttk.LabelFrame(plots, text="Generated Spectrogram  (E / N / Z channels)", padding=4)
        spec_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 6))

        self._spec_fig, self._spec_axes = plt.subplots(
            1, 3, figsize=(10, 2.8), facecolor=BG)
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
        wav_frame.grid(row=1, column=0, sticky="nsew")

        self._wav_fig, self._wav_axes = plt.subplots(
            3, 1, figsize=(10, 3.8), facecolor=BG, sharex=True)
        self._wav_fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.10, hspace=0.08)
        for ax, ch in zip(self._wav_axes, ["E", "N", "Z"]):
            self._style_ax(ax)
            ax.set_ylabel(ch, color=TEXT, fontsize=9)
        self._wav_axes[-1].set_xlabel("Time (s)", color=TEXT, fontsize=9)
        self._wav_canvas = FigureCanvasTkAgg(self._wav_fig, wav_frame)
        self._wav_canvas.get_tk_widget().pack(fill="both", expand=True)

    @staticmethod
    def _style_ax(ax):
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=SUBTEXT, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(SUBTEXT)

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_models(self):
        try:
            # ── Autoencoder ──────────────────────────────────────────────────
            ae_ckpts = sorted((AE / "checkpoints").glob("*/best_model.pt"))
            if not ae_ckpts:
                raise FileNotFoundError(
                    "No autoencoder checkpoint found.\n"
                    "Train it first with ML/autoencoder/train.py")
            ae_ckpt = ae_ckpts[-1]
            self._set_status(f"Loading AE  ({ae_ckpt.parent.name})…", YELLOW)
            self._ae_model, _ = _load_ae(str(ae_ckpt), device=DEVICE)
            self._ae_model.eval()

            # ── Embedding shape ───────────────────────────────────────────────
            emb_path = DIFF / "embeddings" / "embeddings.pt"
            if emb_path.exists():
                emb = torch.load(emb_path, map_location="cpu")
                self._emb_shape = tuple(emb.shape[1:])
            else:
                self._emb_shape = (4, 17, 14)     # default after new AE training

            # ── Scale / normalisation ─────────────────────────────────────────
            scale = _load_scale()
            self._emb_std        = scale.get("emb_std", 1.0)
            self._normalise_cond = _make_normalise_cond(scale)

            # ── Diffusion model ───────────────────────────────────────────────
            diff_ckpts = sorted((DIFF / "checkpoints").glob("epoch_*"))
            if not diff_ckpts:
                raise FileNotFoundError(
                    "No diffusion checkpoint found.\n"
                    "Train it first with ML/diffusion/train.py")
            diff_ckpt = diff_ckpts[-1]
            self._set_status(f"Loading diffusion  ({diff_ckpt.name})…", YELLOW)
            self._diff_unet = UNet2DConditionModel.from_pretrained(str(diff_ckpt)).to(DEVICE)

            self._scheduler = DDPMScheduler(
                num_train_timesteps=NUM_TRAIN_TIMESTEPS,
                beta_start=1e-4, beta_end=0.02)

            self._set_status("Models ready ✓", GREEN)
            self.after(0, lambda: self._gen_btn.config(state="normal"))

        except FileNotFoundError as exc:
            self._set_status(str(exc), RED)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._set_status(f"Load error:\n{exc}", RED)

    def _set_status(self, msg: str, colour: str = TEXT):
        self.after(0, lambda: self._status_lbl.config(text=msg, foreground=colour))

    # ── Generate callback ─────────────────────────────────────────────────────
    def _on_generate(self):
        if self._generating:
            return
        self._generating = True
        self._gen_btn.config(state="disabled")
        self._set_status("Running diffusion…", TEAL)
        threading.Thread(target=self._run_generation, daemon=True).start()

    def _run_generation(self):
        try:
            station_name = self._station_var.get()
            station_idx  = STATION_NAMES.index(station_name)
            cond_meta = {
                "magnitude":   self._svars["mag"].get(),
                "latitude":    self._svars["lat"].get(),
                "longitude":   self._svars["lon"].get(),
                "depth":       self._svars["dep"].get(),
                "station_idx": station_idx,
            }
            suptitle = (
                f"M{cond_meta['magnitude']:.1f}  "
                f"lat={cond_meta['latitude']:.3f}  lon={cond_meta['longitude']:.3f}  "
                f"depth={cond_meta['depth']:.1f} km  "
                f"station={station_name}"
            )

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
                cond_meta,
                step_callback=on_step,
                update_every=50,
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
    def _live_update(self, spec: np.ndarray, waves, suptitle: str):
        """Called on the main thread for every preview update."""
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
            t = np.arange(waves.shape[1]) / FS
            for ax, wave, ch, col in zip(self._wav_axes, waves, ["E", "N", "Z"], CH_COLS):
                ax.cla()
                self._style_ax(ax)
                ax.plot(t, wave, color=col, lw=0.8)
                ax.axhline(0, color=SUBTEXT, lw=0.5, ls="--")
                ax.set_ylabel(ch, color=TEXT, fontsize=9)
            self._wav_axes[-1].set_xlabel("Time (s)", color=TEXT, fontsize=9)
            self._wav_canvas.draw()



# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SeismicDemoApp()
    app.mainloop()
