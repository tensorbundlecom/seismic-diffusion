import torch
import json
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ML.autoencoder.inference import load_model

_scale_path = Path(__file__).resolve().parent / "embeddings" / "scale.json"
_scale = json.load(open(_scale_path)) if _scale_path.exists() else {}
EMB_STD    = _scale.get("emb_std", 1.0)
_cond_mean = torch.tensor(_scale["cond_mean"]) if "cond_mean" in _scale else None
_cond_std  = torch.tensor(_scale["cond_std"])  if "cond_std"  in _scale else None
NUM_CONTINUOUS = 4

def normalise_cond(cond_vec: torch.Tensor) -> torch.Tensor:
    """Normalise only the continuous leading features; leave one-hot tail as-is."""
    if _cond_mean is None:
        return cond_vec
    out = cond_vec.clone()
    out[:NUM_CONTINUOUS] = (out[:NUM_CONTINUOUS] - _cond_mean) / _cond_std
    return out

@torch.no_grad()
def generate(cond_vec, embedding_shape, scheduler, num_train_timesteps, model, device):
    """Run the full reverse diffusion loop given a conditioning vector (1, cond_dim).
    cond_vec should be the *raw* (un-normalised) output of create_conditioning_vector.
    """
    model.model.eval()
    x = torch.randn(1, *embedding_shape, device=device)
    cond = normalise_cond(cond_vec).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, cond_dim)
    scheduler.set_timesteps(num_train_timesteps)
    for t in scheduler.timesteps:
        t_batch = t.unsqueeze(0).to(device)
        noise_pred = model.forward(x, t_batch, cond).sample
        x = scheduler.step(noise_pred, t, x).prev_sample
    # Undo the variance-normalisation applied during training
    return (x * EMB_STD).squeeze(0).cpu()  # (C, H, W)

def embedding_to_figure(tensor, title):
    """Return a matplotlib figure of the mean-channel heatmap."""
    img = tensor.mean(dim=0).detach().numpy()
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(img, cmap="seismic", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


model, config = load_model("../autoencoder/checkpoints/20260227_010846/best_model.pt", device='cuda')
def decode_embedding(embedding):
    decoded_image = model.decode(embedding.unsqueeze(0).to('cuda'))[0].cpu()
    return decoded_image