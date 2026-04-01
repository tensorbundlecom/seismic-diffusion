import torch
import json
import matplotlib.pyplot as plt
import re
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ML.autoencoder.inference import load_model

_scale_path = Path(__file__).resolve().parent / "embeddings" / "scale.json"
_scale = json.load(open(_scale_path)) if _scale_path.exists() else {}
EMB_STD    = _scale.get("emb_std", 1.0)
EMB_MEAN   = _scale.get("emb_mean", 0.0)
_cond_mean = torch.tensor(_scale["cond_mean"]) if "cond_mean" in _scale else None
_cond_std  = torch.tensor(_scale["cond_std"])  if "cond_std"  in _scale else None
NUM_CONTINUOUS = int(_scale.get("num_continuous", 4))
_source_path = Path(__file__).resolve().parent / "embeddings" / "source.json"

def normalise_cond(
    cond_vec: torch.Tensor,
    cond_mean: torch.Tensor = None,
    cond_std: torch.Tensor = None,
    num_continuous: int = None,
) -> torch.Tensor:
    """Normalise only leading continuous features; keep station_idx tail unchanged."""
    if cond_mean is None:
        cond_mean = _cond_mean
    if cond_std is None:
        cond_std = _cond_std
    if num_continuous is None:
        num_continuous = NUM_CONTINUOUS

    if cond_mean is None or cond_std is None:
        return cond_vec

    if not torch.is_tensor(cond_mean):
        cond_mean = torch.tensor(cond_mean, dtype=cond_vec.dtype, device=cond_vec.device)
    else:
        cond_mean = cond_mean.to(device=cond_vec.device, dtype=cond_vec.dtype)
    if not torch.is_tensor(cond_std):
        cond_std = torch.tensor(cond_std, dtype=cond_vec.dtype, device=cond_vec.device)
    else:
        cond_std = cond_std.to(device=cond_vec.device, dtype=cond_vec.dtype)

    out = cond_vec.clone()
    n = int(num_continuous)
    out[:n] = (out[:n] - cond_mean[:n]) / cond_std[:n]
    return out

@torch.no_grad()
def generate(
    cond_vec,
    embedding_shape,
    scheduler,
    num_train_timesteps,
    model,
    device,
    cond_mean=None,
    cond_std=None,
    num_continuous=None,
    data_mean=None,
    data_std=None,
):
    """Run the full reverse diffusion loop given a conditioning vector (1, cond_dim).
    cond_vec should be the *raw* (un-normalised) output of create_conditioning_vector.
    """
    model.eval()
    x = torch.randn(1, *embedding_shape, device=device)
    cond = normalise_cond(
        cond_vec,
        cond_mean=cond_mean,
        cond_std=cond_std,
        num_continuous=num_continuous,
    ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, cond_dim)
    scheduler.set_timesteps(num_train_timesteps)
    for t in scheduler.timesteps:
        t_batch = t.unsqueeze(0).to(device)
        noise_pred = model.forward(x, t_batch, cond).sample
        if not torch.isfinite(noise_pred).all():
            raise RuntimeError(f"Non-finite noise prediction encountered at timestep {int(t.item())}.")
        x = scheduler.step(noise_pred, t, x).prev_sample
        if not torch.isfinite(x).all():
            raise RuntimeError(f"Non-finite latent encountered after scheduler step at timestep {int(t.item())}.")
    if data_mean is None:
        data_mean = EMB_MEAN
    if data_std is None:
        data_std = EMB_STD
    # Undo data normalization applied during training.
    return (x * float(data_std) + float(data_mean)).squeeze(0).cpu()  # (C, H, W)

def embedding_to_figure(tensor, title):
    """
    Return a matplotlib figure with per-channel views plus an energy map.

    This avoids mean-channel cancellation that can make non-zero outputs
    look uniformly gray/white in TensorBoard.
    """
    t = tensor.detach().cpu()
    n_ch = min(3, t.shape[0])
    ch = t[:n_ch].numpy()

    # Robust channel range so outliers don't flatten the plot.
    vmin, vmax = np.percentile(ch, [2, 98])
    if abs(vmax - vmin) < 1e-8:
        vmin, vmax = float(ch.min()), float(ch.max())
        if abs(vmax - vmin) < 1e-8:
            center = float(ch.mean())
            vmin, vmax = center - 1e-3, center + 1e-3

    energy = t[:n_ch].abs().mean(dim=0).numpy()
    e_vmin, e_vmax = np.percentile(energy, [2, 98])
    if abs(e_vmax - e_vmin) < 1e-8:
        e_vmin, e_vmax = float(energy.min()), float(energy.max())
        if abs(e_vmax - e_vmin) < 1e-8:
            center = float(energy.mean())
            e_vmin, e_vmax = center - 1e-3, center + 1e-3

    fig, axes = plt.subplots(1, n_ch + 1, figsize=(3.2 * (n_ch + 1), 3))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i in range(n_ch):
        ax = axes[i]
        im = ax.imshow(ch[i], cmap="seismic", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(f"Ch {i}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[n_ch]
    im = ax.imshow(energy, cmap="magma", aspect="auto", vmin=e_vmin, vmax=e_vmax)
    ax.set_title("Abs mean")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _find_latest_ae_checkpoint():
    """
    Return the path to the most recent timestamped best_model.pt under
    ML/autoencoder/checkpoints/.

    Ignores temporary folders such as _tmp_*.
    """
    ckpt_root = Path(__file__).resolve().parent.parent.joinpath("autoencoder", "checkpoints")
    timestamp_pat = re.compile(r"^\d{8}_\d{6}$")
    ckpts = sorted(
        p for p in ckpt_root.glob("*/best_model.pt")
        if timestamp_pat.match(p.parent.name)
    )
    if not ckpts:
        raise FileNotFoundError("No autoencoder checkpoint found. Train it first.")
    return str(ckpts[-1])


def _get_embeddings_source_checkpoint():
    """Return AE checkpoint path recorded at embedding creation time, if available."""
    if not _source_path.exists():
        return None
    try:
        source = json.load(open(_source_path))
    except Exception:
        return None
    ckpt = source.get("ae_checkpoint")
    if not ckpt:
        return None
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = (Path(__file__).resolve().parent / ckpt_path).resolve()
    return str(ckpt_path) if ckpt_path.exists() else None

_ae_model = None

def _get_ae_model():
    """Lazy-load the autoencoder (so importing utils.py has no side-effects)."""
    global _ae_model
    if _ae_model is None:
        import torch
        ckpt = _get_embeddings_source_checkpoint() or _find_latest_ae_checkpoint()
        print(f"[diffusion.utils] Loading AE checkpoint for decode: {ckpt}")
        _ae_model, _ = load_model(ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
        _ae_model.eval()
    return _ae_model

def decode_embedding(embedding):
    ae = _get_ae_model()
    device = next(ae.parameters()).device
    decoded = ae.decode(embedding.unsqueeze(0).to(device))[0].cpu()
    return decoded
