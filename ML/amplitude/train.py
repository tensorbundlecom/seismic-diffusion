"""
Train the amplitude model.

Run from ML/amplitude/:
    python train.py

Target: per-channel std of the filtered (not normalized) waveforms.
At inference the generated waveform is first normalized to unit variance,
then scaled by the predicted std, so the units match directly.

Saves:
  checkpoints/amplitude_mlp.pt   — model weights + config
  checkpoints/amp_stats.json     — log-std normalization stats needed at inference
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

DIFF_DIR = Path(__file__).resolve().parent.parent / "diffusion"
AMP_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ML.diffusion.model import create_conditioning_vector  # noqa: E402
from ML.amplitude.model import AmplitudeMLP               # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_EPOCHS       = 200
BATCH_SIZE       = 64
LR               = 1e-3
NUM_CONTINUOUS   = 6
TEST_FRACTION    = 0.1
CHECKPOINT_EVERY = 20
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
CHANNEL_NAMES    = ["E", "N", "Z"]

# ── Load metadata and conditioning normalization stats ─────────────────────────
metadatas         = json.load(open(DIFF_DIR / "embeddings/metadata.json"))
station_locations = json.load(open(DIFF_DIR / "embeddings/station_locations.json"))
scale             = json.load(open(DIFF_DIR / "embeddings/scale.json"))

cond_mean = torch.tensor(scale["cond_mean"], dtype=torch.float32)
cond_std  = torch.tensor(scale["cond_std"],  dtype=torch.float32).clamp(min=1e-8)

raw_cond = torch.stack([create_conditioning_vector(m, station_locations) for m in metadatas])
cond = raw_cond.clone()
cond[:, :NUM_CONTINUOUS] = (cond[:, :NUM_CONTINUOUS] - cond_mean) / cond_std

# ── Compute per-channel stds from filtered waveforms ──────────────────────────
try:
    from obspy import read as obspy_read
except ImportError:
    raise RuntimeError("obspy is required: pip install obspy")

print(f"Computing per-channel stds from {len(metadatas)} waveforms…")
raw_stds = []
for m in tqdm(metadatas):
    file_path = Path(m["file_path"])
    if not file_path.is_absolute():
        candidates = [
            (DIFF_DIR / file_path).resolve(),
            (Path.cwd() / file_path).resolve(),
        ]
        file_path = next((c for c in candidates if c.exists()), candidates[0])

    stream = obspy_read(str(file_path))
    stream.sort(keys=["channel"])
    raw_stds.append([float(np.std(tr.data.astype(np.float64))) for tr in stream[:3]])

stds_tensor = torch.tensor(raw_stds, dtype=torch.float32)   # (N, 3)
log_stds    = torch.log(stds_tensor.clamp(min=1e-10))        # (N, 3)

# Normalize targets for stable training.
log_std_mean  = log_stds.mean(dim=0)
log_std_scale = log_stds.std(dim=0).clamp(min=1e-8)
log_stds_norm = (log_stds - log_std_mean) / log_std_scale

ckpt_dir = AMP_DIR / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
json.dump(
    {
        "log_std_mean":  log_std_mean.tolist(),
        "log_std_scale": log_std_scale.tolist(),
    },
    open(ckpt_dir / "amp_stats.json", "w"),
)
print(f"Saved amp_stats.json  (log_std_mean={log_std_mean.tolist()})")

# ── Model ──────────────────────────────────────────────────────────────────────
num_stations = max(int(m["station_idx"]) for m in metadatas) + 1
model        = AmplitudeMLP(num_stations=num_stations, num_continuous=NUM_CONTINUOUS).to(DEVICE)
optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}  device={DEVICE}")

dataset  = TensorDataset(cond, log_stds_norm)
n_test   = max(1, int(len(dataset) * TEST_FRACTION))
n_train  = len(dataset) - n_test
train_set, test_set = random_split(
    dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)
print(f"Train: {n_train}  Test: {n_test}")

writer = SummaryWriter(log_dir=str(AMP_DIR / "runs/amplitude"))

# ── Training loop ──────────────────────────────────────────────────────────────


def _eval_loss(loader):
    model.eval()
    total_loss = 0.0
    total_ch   = torch.zeros(3)
    with torch.no_grad():
        for batch_cond, batch_targets in loader:
            batch_cond    = batch_cond.to(DEVICE)
            batch_targets = batch_targets.to(DEVICE)
            pred       = model(batch_cond)
            total_loss += nn.functional.mse_loss(pred, batch_targets).item()
            total_ch   += ((pred - batch_targets) ** 2).mean(dim=0).cpu()
    return total_loss / len(loader), total_ch / len(loader)


for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    epoch_ch   = torch.zeros(3)

    for batch_cond, batch_targets in train_loader:
        batch_cond    = batch_cond.to(DEVICE)
        batch_targets = batch_targets.to(DEVICE)

        pred = model(batch_cond)
        loss = nn.functional.mse_loss(pred, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        with torch.no_grad():
            epoch_ch += ((pred - batch_targets) ** 2).mean(dim=0).cpu()

    train_avg = epoch_loss / len(train_loader)
    train_ch  = epoch_ch   / len(train_loader)
    test_avg, test_ch = _eval_loss(test_loader)

    writer.add_scalars("Loss/total", {"train": train_avg, "test": test_avg}, epoch)
    for i, ch in enumerate(CHANNEL_NAMES):
        writer.add_scalars(
            f"Loss/channel_{ch}",
            {"train": train_ch[i].item(), "test": test_ch[i].item()},
            epoch,
        )

    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        model.save(ckpt_dir / f"amplitude_mlp_epoch{epoch + 1}.pt")
        model.save(ckpt_dir / "amplitude_mlp.pt")
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}  "
            f"train={train_avg:.6f}  test={test_avg:.6f}  "
            + "  ".join(f"{ch}={test_ch[i]:.6f}" for i, ch in enumerate(CHANNEL_NAMES))
        )

writer.close()
model.save(ckpt_dir / "amplitude_mlp.pt")
print(f"Saved model to {ckpt_dir / 'amplitude_mlp.pt'}")
