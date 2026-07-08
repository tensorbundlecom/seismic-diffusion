"""
Train the amplitude model.

Run from ML/amplitude/:
    python train.py

Targets (6 per sample):
  0-2  per-channel AMP_METRIC (max |amplitude| or std) of the processed
       waveform — resampled to the dataset rate and cropped/padded to the
       training window, so the units match the Griffin-Lim reconstruction
       that gets rescaled at inference.
  3-5  per-channel STFT log-magnitude range log1p|S|.max() - log1p|S|.min(),
       i.e. the inv-log gain destroyed by the diffusion dataset's per-sample
       min-max normalization, needed to invert it at inference.

Saves:
  checkpoints/amplitude_mlp.pt   — model weights + config
  checkpoints/amp_stats.json     — target normalization stats needed at inference
  checkpoints/amp_targets.pt     — cached target tensor (waveform pass is slow)
"""

import argparse
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

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train amplitude model")
parser.add_argument(
    "--use_vs30",
    action="store_true",
    help=(
        "Add the per-station Vs30 (site condition, m/s) as an extra continuous "
        "feature, matching a diffusion model trained with --use_vs30. Requires the "
        "station Vs30 lookup and a scale.json that already includes Vs30 stats."
    ),
)
parser.add_argument(
    "--station_vs30",
    type=str,
    default=str(DIFF_DIR / "embeddings" / "station_vs30.json"),
    help="Path to the station -> Vs30 JSON lookup used when --use_vs30 is set.",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="Parallel workers for the waveform/STFT target computation pass.",
)
parser.add_argument(
    "--recompute_targets",
    action="store_true",
    help="Ignore the cached target tensor and recompute from waveforms.",
)
args = parser.parse_args()

# ── Config ─────────────────────────────────────────────────────────────────────
NUM_EPOCHS       = 200
BATCH_SIZE       = 64
LR               = 1e-3
# Base continuous features (magnitude, distance, sin/cos azimuth, depth, snr);
# Vs30 is appended as a 7th continuous feature when --use_vs30 is set.
NUM_CONTINUOUS   = 7 if args.use_vs30 else 6
TEST_FRACTION    = 0.1
CHECKPOINT_EVERY = 20
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
CHANNEL_NAMES    = ["E", "N", "Z"]
# AMP_METRIC       = "std"   # per-channel amplitude target: "std" or "max" (max |amplitude|)
AMP_METRIC       = "max"   # per-channel amplitude target: "std" or "max" (max |amplitude|)
assert AMP_METRIC in ("std", "max"), f"AMP_METRIC must be 'std' or 'max', got {AMP_METRIC!r}"

# ── Load metadata and conditioning normalization stats ─────────────────────────
metadatas         = json.load(open(DIFF_DIR / "embeddings/metadata.json"))
station_locations = json.load(open(DIFF_DIR / "embeddings/station_locations.json"))
scale             = json.load(open(DIFF_DIR / "embeddings/scale.json"))

cond_mean = torch.tensor(scale["cond_mean"], dtype=torch.float32)
cond_std  = torch.tensor(scale["cond_std"],  dtype=torch.float32).clamp(min=1e-8)

station_vs30 = None
if args.use_vs30:
    if len(cond_mean) < NUM_CONTINUOUS:
        raise ValueError(
            f"--use_vs30 needs Vs30 normalization stats, but scale.json only has "
            f"{len(cond_mean)} continuous dims. Train the diffusion model with "
            "--use_vs30 first so scale.json includes the Vs30 stats."
        )
    vs30_path = Path(args.station_vs30)
    if not vs30_path.exists():
        raise FileNotFoundError(
            f"Missing {vs30_path}. Run ML/diffusion/compute_station_vs30.py first."
        )
    station_vs30 = json.load(open(vs30_path))
    print(f"[amplitude] Vs30 conditioning enabled ({len(station_vs30)} stations).")

# Use only the leading NUM_CONTINUOUS stats so a longer (Vs30) scale.json stays
# compatible with a base run and vice versa.
cond_mean = cond_mean[:NUM_CONTINUOUS]
cond_std  = cond_std[:NUM_CONTINUOUS]

raw_cond = torch.stack(
    [create_conditioning_vector(m, station_locations, station_vs30) for m in metadatas]
)
cond = raw_cond.clone()
cond[:, :NUM_CONTINUOUS] = (cond[:, :NUM_CONTINUOUS] - cond_mean) / cond_std

# ── Compute per-channel targets from waveforms ────────────────────────────────
# Waveform processing (resample, crop/pad, STFT params) must match
# STFTDataWithMetadataConditionDataset._compute_raw_stft so the gain targets
# invert exactly the normalization the diffusion data was trained with.
try:
    from obspy import read as obspy_read
    from scipy import signal as sp_signal
except ImportError:
    raise RuntimeError("obspy and scipy are required: pip install obspy scipy")


def _load_source_stft_config():
    source_path = DIFF_DIR / "embeddings" / "source.json"
    defaults = {"nperseg": 256, "noverlap": 192, "nfft": 256,
                "resample_hz": 100.0, "target_seconds": 70.0}
    if not source_path.exists():
        print("[amplitude] embeddings/source.json not found; using default STFT params.")
        return defaults
    stft = json.load(open(source_path)).get("stft", {})
    return {k: type(v)(stft.get(k, v)) for k, v in defaults.items()}


STFT_CFG       = _load_source_stft_config()
TARGET_SAMPLES = int(round(STFT_CFG["resample_hz"] * STFT_CFG["target_seconds"]))


def _channel_metric(data):
    """Per-channel amplitude statistic selected by AMP_METRIC."""
    if AMP_METRIC == "max":
        return float(np.max(np.abs(data)))
    return float(np.std(data))


def _compute_targets_for_file(path_str):
    """Return [amp_E, amp_N, amp_Z, gain_E, gain_N, gain_Z] for one waveform file."""
    stream = obspy_read(path_str)
    if len(stream) != 3:
        raise ValueError(f"Expected 3 traces, got {len(stream)} in {path_str}")
    stream.sort(keys=["channel"])
    amps, gains = [], []
    for trace in stream:
        if abs(trace.stats.sampling_rate - STFT_CFG["resample_hz"]) > 1e-6:
            trace.resample(STFT_CFG["resample_hz"])
        data = trace.data.astype(np.float32)
        n = TARGET_SAMPLES
        data = data[:n] if data.shape[0] >= n else np.pad(data, (0, n - data.shape[0]), mode="constant")
        amps.append(_channel_metric(data))
        _, _, zxx = sp_signal.stft(
            data,
            fs=trace.stats.sampling_rate,
            nperseg=STFT_CFG["nperseg"],
            noverlap=STFT_CFG["noverlap"],
            nfft=STFT_CFG["nfft"],
            return_onesided=True,
            boundary="zeros",
            padded=True,
        )
        log_mag = np.log1p(np.abs(zxx))
        gains.append(float(log_mag.max() - log_mag.min()))
    return amps + gains


def _resolve_path(m):
    file_path = Path(m["file_path"])
    if not file_path.is_absolute():
        candidates = [
            (DIFF_DIR / file_path).resolve(),
            (Path.cwd() / file_path).resolve(),
        ]
        file_path = next((c for c in candidates if c.exists()), candidates[0])
    return str(file_path)


ckpt_dir = AMP_DIR / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

cache_path = ckpt_dir / "amp_targets.pt"
cache_key  = {"n": len(metadatas), "metric": AMP_METRIC, "stft": STFT_CFG}
targets    = None
if cache_path.exists() and not args.recompute_targets:
    cached = torch.load(cache_path)
    if cached.get("key") == cache_key:
        targets = cached["targets"]
        print(f"Loaded cached targets from {cache_path}")
    else:
        print("Target cache is stale; recomputing.")

if targets is None:
    from concurrent.futures import ProcessPoolExecutor

    file_paths = [_resolve_path(m) for m in metadatas]
    print(
        f"Computing per-channel {AMP_METRIC} + STFT log-gain from "
        f"{len(file_paths)} waveforms ({args.num_workers} workers)…"
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        rows = list(tqdm(
            pool.map(_compute_targets_for_file, file_paths, chunksize=64),
            total=len(file_paths),
        ))
    targets = torch.tensor(rows, dtype=torch.float32)  # (N, 6)
    torch.save({"key": cache_key, "targets": targets}, cache_path)
    print(f"Cached targets to {cache_path}")

amps_tensor  = targets[:, :3]
gains_tensor = targets[:, 3:]
log_amps     = torch.log(amps_tensor.clamp(min=1e-10))   # (N, 3)

# The gain is ~log(amplitude) + a spectral-concentration term; report the
# correlation so we know how much independent signal the gain head carries.
for i, ch in enumerate(CHANNEL_NAMES):
    la = log_amps[:, i] - log_amps[:, i].mean()
    g  = gains_tensor[:, i] - gains_tensor[:, i].mean()
    corr = (la * g).sum() / (la.norm() * g.norm()).clamp(min=1e-12)
    print(f"corr(log {AMP_METRIC}, stft log-gain)  channel {ch}: {corr.item():.4f}")

# Normalize targets for stable training. Amplitudes are z-scored in log space;
# gains are already log-domain quantities, so they are z-scored directly.
log_std_mean  = log_amps.mean(dim=0)
log_std_scale = log_amps.std(dim=0).clamp(min=1e-8)
gain_mean     = gains_tensor.mean(dim=0)
gain_scale    = gains_tensor.std(dim=0).clamp(min=1e-8)
targets_norm  = torch.cat(
    [(log_amps - log_std_mean) / log_std_scale,
     (gains_tensor - gain_mean) / gain_scale],
    dim=1,
)  # (N, 6)

json.dump(
    {
        "metric":         AMP_METRIC,
        "out_dim":        6,
        "log_std_mean":   log_std_mean.tolist(),
        "log_std_scale":  log_std_scale.tolist(),
        "gain_mean":      gain_mean.tolist(),
        "gain_scale":     gain_scale.tolist(),
        "use_vs30":       bool(args.use_vs30),
        "num_continuous": NUM_CONTINUOUS,
        "stft":           STFT_CFG,
    },
    open(ckpt_dir / "amp_stats.json", "w"),
)
print(f"Saved amp_stats.json  (metric={AMP_METRIC}  gain_mean={gain_mean.tolist()})")

# ── Model ──────────────────────────────────────────────────────────────────────
TARGET_NAMES = CHANNEL_NAMES + [f"gain_{ch}" for ch in CHANNEL_NAMES]
num_stations = max(int(m["station_idx"]) for m in metadatas) + 1
model        = AmplitudeMLP(
    num_stations=num_stations, num_continuous=NUM_CONTINUOUS, out_dim=len(TARGET_NAMES)
).to(DEVICE)
optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}  device={DEVICE}")

dataset  = TensorDataset(cond, targets_norm)
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
    total_ch   = torch.zeros(len(TARGET_NAMES))
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
    epoch_ch   = torch.zeros(len(TARGET_NAMES))

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
    for i, name in enumerate(TARGET_NAMES):
        writer.add_scalars(
            f"Loss/target_{name}",
            {"train": train_ch[i].item(), "test": test_ch[i].item()},
            epoch,
        )

    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        model.save(ckpt_dir / f"amplitude_mlp_epoch{epoch + 1}.pt")
        model.save(ckpt_dir / "amplitude_mlp.pt")
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}  "
            f"train={train_avg:.6f}  test={test_avg:.6f}  "
            + "  ".join(f"{name}={test_ch[i]:.6f}" for i, name in enumerate(TARGET_NAMES))
        )

writer.close()
model.save(ckpt_dir / "amplitude_mlp.pt")
print(f"Saved model to {ckpt_dir / 'amplitude_mlp.pt'}")
