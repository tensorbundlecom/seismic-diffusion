from model import DiffusionUNet2D, create_conditioning_vector
import torch
import json
import shutil
import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from diffusers import DDPMScheduler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import generate, decode_embedding


class STFTDataWithMetadataConditionDataset(Dataset):
    """
    Dataset for diffusion *data* in STFT space while conditioning stays metadata.

    Returns:
      x_data: normalized STFT tensor (C, F, T)
      cond:   normalized metadata conditioning vector (5,)
    """

    def __init__(
        self,
        metadatas: List[Dict],
        cond_vectors: torch.Tensor,
        nperseg: int,
        noverlap: int,
        nfft: int,
        base_dir: Path,
        target_freq_bins: int = None,
        target_time_bins: int = None,
    ):
        if len(metadatas) != len(cond_vectors):
            raise ValueError(
                f"Metadata length ({len(metadatas)}) != cond_vectors length ({len(cond_vectors)})."
            )

        self.metadatas = metadatas
        self.cond_vectors = cond_vectors
        self.nperseg = int(nperseg)
        self.noverlap = int(noverlap)
        self.nfft = int(nfft)
        self.base_dir = base_dir

        if self.nperseg <= 0:
            raise ValueError(f"Invalid nperseg: {self.nperseg}")
        if self.noverlap < 0 or self.noverlap >= self.nperseg:
            raise ValueError(
                f"Invalid noverlap={self.noverlap}; expected 0 <= noverlap < nperseg({self.nperseg})."
            )
        if self.nfft < self.nperseg:
            raise ValueError(
                f"Invalid nfft={self.nfft}; expected nfft >= nperseg({self.nperseg})."
            )

        shape0 = metadatas[0].get("shape", [3, self.nfft // 2 + 1, 1])
        c0 = int(shape0[0])
        native_f = int(shape0[1])
        native_t = int(shape0[2])
        f = int(target_freq_bins) if target_freq_bins is not None else native_f
        t = int(target_time_bins) if target_time_bins is not None else native_t
        if f <= 0 or t <= 0:
            raise ValueError(
                f"Invalid target STFT shape requested: freq_bins={f}, time_bins={t}."
            )
        self.expected_shape = (c0, f, t)
        self.file_paths = [self._resolve_file_path(m["file_path"]) for m in metadatas]

        # Data normalization values for diffusion training.
        self.data_mean = 0.0
        self.data_std = 1.0

        # Lazy imports per process.
        self._obspy_read = None
        self._sp_signal = None

    def _resolve_file_path(self, path_str: str) -> Path:
        raw = Path(path_str)
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.extend(
                [
                    (self.base_dir / raw).resolve(),
                    (Path.cwd() / raw).resolve(),
                ]
            )

        for cand in candidates:
            if cand.exists():
                return cand

        raise FileNotFoundError(
            f"Could not resolve waveform path '{path_str}'. Tried: {', '.join(str(c) for c in candidates)}"
        )

    def _lazy_imports(self):
        if self._obspy_read is None or self._sp_signal is None:
            try:
                from obspy import read as obspy_read
                from scipy import signal as sp_signal
            except Exception as exc:
                raise RuntimeError(
                    "--data_mode stft requires obspy and scipy in the active environment."
                ) from exc
            self._obspy_read = obspy_read
            self._sp_signal = sp_signal

    def _compute_raw_stft(self, idx: int) -> torch.Tensor:
        self._lazy_imports()
        file_path = self.file_paths[idx]
        stream = self._obspy_read(str(file_path))
        if len(stream) != 3:
            raise ValueError(f"Expected 3 traces, got {len(stream)} in {file_path}")
        stream.sort(keys=["channel"])

        channels = []
        for trace in stream:
            data = trace.data.astype(np.float32)
            _, _, zxx = self._sp_signal.stft(
                data,
                fs=trace.stats.sampling_rate,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                return_onesided=True,
                boundary="zeros",
                padded=True,
            )
            mag = np.log1p(np.abs(zxx))
            mag_min = float(mag.min())
            mag_max = float(mag.max())
            if mag_max > mag_min:
                mag = (mag - mag_min) / (mag_max - mag_min)
            else:
                mag = np.zeros_like(mag)
            channels.append(mag.astype(np.float32))

        stft = torch.from_numpy(np.stack(channels, axis=0))  # (3, F, T)

        # Keep shape stable for batching; crop/pad only if needed.
        c_exp, f_exp, t_exp = self.expected_shape
        if tuple(stft.shape) != (c_exp, f_exp, t_exp):
            aligned = torch.zeros((c_exp, f_exp, t_exp), dtype=stft.dtype)
            c = min(c_exp, stft.shape[0])
            f = min(f_exp, stft.shape[1])
            t = min(t_exp, stft.shape[2])
            aligned[:c, :f, :t] = stft[:c, :f, :t]
            stft = aligned

        return stft

    def estimate_stats(self, num_samples: int = 2048):
        n = len(self)
        if n == 0:
            raise RuntimeError("Empty STFT dataset")

        if num_samples <= 0 or num_samples >= n:
            indices = list(range(n))
        else:
            indices = np.linspace(0, n - 1, num=num_samples, dtype=int).tolist()

        total_sum = 0.0
        total_sq = 0.0
        total_count = 0

        for idx in tqdm(indices, desc="Estimating STFT scale"):
            x = self._compute_raw_stft(idx)
            total_sum += float(x.sum().item())
            total_sq += float((x * x).sum().item())
            total_count += int(x.numel())

        mean = total_sum / max(1, total_count)
        var = max(total_sq / max(1, total_count) - mean * mean, 1e-8)
        std = float(math.sqrt(var))
        return float(mean), std

    def set_normalization(self, data_mean: float, data_std: float):
        self.data_mean = float(data_mean)
        self.data_std = max(float(data_std), 1e-8)

    def __len__(self):
        return len(self.metadatas)

    def __getitem__(self, idx):
        raw_stft = self._compute_raw_stft(idx)
        x_data = (raw_stft - self.data_mean) / self.data_std
        cond = self.cond_vectors[idx]
        return x_data, cond


def _load_source_stft_config() -> Dict[str, int]:
    source_path = Path("embeddings/source.json")
    defaults = {"nperseg": 256, "noverlap": 192, "nfft": 256}
    if not source_path.exists():
        print("[train] embeddings/source.json not found; using default STFT params.")
        return defaults

    try:
        payload = json.load(open(source_path, "r"))
        stft = payload.get("stft", {})
        return {
            "nperseg": int(stft.get("nperseg", defaults["nperseg"])),
            "noverlap": int(stft.get("noverlap", defaults["noverlap"])),
            "nfft": int(stft.get("nfft", defaults["nfft"])),
        }
    except Exception as exc:
        print(f"[train] Failed reading source STFT config ({exc}); using defaults.")
        return defaults


# --- CLI ---
parser = argparse.ArgumentParser(description="Train diffusion model")
parser.add_argument(
    "--training_type",
    type=str,
    default="ddpm",
    choices=["ddpm", "flow_matching"],
    help="ddpm = standard DDPM training; flow_matching = linear flow matching (velocity target).",
)
parser.add_argument(
    "--prediction_target",
    type=str,
    default="epsilon",
    choices=["epsilon", "x0", "v_prediction"],
    help=(
        "Training target: epsilon (predict noise), "
        "x0 (predict clean sample), or v_prediction (predict velocity)."
    ),
)
parser.add_argument(
    "--data_mode",
    type=str,
    default="latent",
    choices=["latent", "stft"],
    help=(
        "Diffusion data representation. "
        "latent=AE embeddings (default), stft=STFT tensors. "
        "Conditioning stays metadata in both modes."
    ),
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="DataLoader workers (mainly relevant for --data_mode stft).",
)
parser.add_argument(
    "--stft_stats_samples",
    type=int,
    default=2048,
    help=(
        "Number of STFT samples used to estimate mean/std for data normalization. "
        "Use <=0 to use the full dataset."
    ),
)
parser.add_argument(
    "--stft_freq_bins",
    type=int,
    default=0,
    help=(
        "Target STFT frequency bins for --data_mode stft. "
        "Set <=0 to keep native bins from metadata shape."
    ),
)
parser.add_argument(
    "--stft_time_bins",
    type=int,
    default=0,
    help=(
        "Target STFT time bins for --data_mode stft. "
        "Set <=0 to keep native bins from metadata shape."
    ),
)
parser.add_argument(
    "--log_images_every_n_batches",
    type=int,
    default=200,
    help=(
        "Log generated preview images every N optimizer batches. "
        "Set <=0 to disable image logging during training."
    ),
)
parser.add_argument(
    "--checkpoint_every_n_batches",
    type=int,
    default=0,
    help=(
        "Save an additional checkpoint every N optimizer batches (step checkpoints). "
        "Set <=0 to disable batch checkpointing."
    ),
)
parser.add_argument(
    "--keep_last_batch_checkpoints",
    type=int,
    default=3,
    help="How many step_* checkpoints to keep when batch checkpointing is enabled.",
)
args = parser.parse_args()

# --- Config ---
NUM_EPOCHS = 500
BATCH_SIZE = 32
LR = 1e-4
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
CHECKPOINT_EVERY_N_EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREDICTION_TARGET = "sample" if args.prediction_target == "x0" else args.prediction_target  # HF scheduler name
STATION_EMB_DIM = 64
NUM_CONTINUOUS = 6
CFG_DROPOUT = 0.15       # fraction of samples per batch trained unconditionally
CFG_GUIDANCE_SCALE = 3.0 # guidance scale used when logging preview images
TRAINING_TYPE = args.training_type

writer = SummaryWriter(log_dir=f"runs/diffusion_{args.data_mode}_{TRAINING_TYPE}")

# --- Shared metadata conditioning ---
metadatas = json.load(open("embeddings/metadata.json", "r"))
station_locations_path = Path("embeddings/station_locations.json")
if not station_locations_path.exists():
    raise FileNotFoundError(
        f"Missing {station_locations_path}. Run fetch_station_locations.py first."
    )
station_locations = json.load(open(station_locations_path, "r"))

raw_cond_vectors = torch.stack([create_conditioning_vector(m, station_locations) for m in metadatas])
cond_mean = raw_cond_vectors[:, :NUM_CONTINUOUS].mean(dim=0)
cond_std = raw_cond_vectors[:, :NUM_CONTINUOUS].std(dim=0).clamp(min=1e-8)
cond_vectors = raw_cond_vectors.clone()
cond_vectors[:, :NUM_CONTINUOUS] = (cond_vectors[:, :NUM_CONTINUOUS] - cond_mean) / cond_std

fixed_real_idx = 0
fixed_real_cond = raw_cond_vectors[fixed_real_idx]
fixed_rand_idx = torch.randint(len(raw_cond_vectors), (1,)).item()
fixed_rand_cond = raw_cond_vectors[fixed_rand_idx]
fixed_real_stft = None

# --- Data mode specific setup ---
if args.data_mode == "latent":
    data_tensor = torch.load("embeddings/embeddings.pt", map_location="cpu").float()
    if len(data_tensor) != len(cond_vectors):
        raise ValueError(
            f"Embeddings count ({len(data_tensor)}) != metadata count ({len(cond_vectors)})."
        )

    data_mean = 0.0
    data_std = float(data_tensor.std().item())
    train_data = data_tensor / data_std
    fixed_real_stft = decode_embedding(data_tensor[fixed_real_idx])
    dataset = TensorDataset(train_data, cond_vectors)
    data_shape = tuple(train_data.shape[1:])
    num_workers = 0

    print(f"[train] data_mode=latent, shape={data_shape}, std={data_std:.6f}")

else:
    stft_cfg = _load_source_stft_config()
    target_freq_bins = int(args.stft_freq_bins) if int(args.stft_freq_bins) > 0 else None
    target_time_bins = int(args.stft_time_bins) if int(args.stft_time_bins) > 0 else None
    stft_dataset = STFTDataWithMetadataConditionDataset(
        metadatas=metadatas,
        cond_vectors=cond_vectors,
        nperseg=stft_cfg["nperseg"],
        noverlap=stft_cfg["noverlap"],
        nfft=stft_cfg["nfft"],
        base_dir=Path(__file__).resolve().parent,
        target_freq_bins=target_freq_bins,
        target_time_bins=target_time_bins,
    )

    data_mean, data_std = stft_dataset.estimate_stats(args.stft_stats_samples)
    stft_dataset.set_normalization(data_mean, data_std)
    fixed_real_stft = stft_dataset._compute_raw_stft(fixed_real_idx)
    x0, _ = stft_dataset[0]
    data_shape = tuple(x0.shape)
    dataset = stft_dataset
    num_workers = max(0, int(args.num_workers))

    print(
        f"[train] data_mode=stft, shape={data_shape}, "
        f"mean={data_mean:.6f}, std={data_std:.6f}, "
        f"target_override=(freq={target_freq_bins}, time={target_time_bins})"
    )

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(num_workers > 0),
)

# --- Model & Scheduler ---
num_stations = max(int(m["station_idx"]) for m in metadatas) + 1
model = DiffusionUNet2D(
    in_channels=int(data_shape[0]),
    out_channels=int(data_shape[0]),
    num_stations=num_stations,
    station_emb_dim=STATION_EMB_DIM,
)
model.to(DEVICE)
print(f"Conditioning: metadata (unchanged)")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

noise_scheduler = DDPMScheduler(
    num_train_timesteps=NUM_TRAIN_TIMESTEPS,
    beta_start=BETA_START,
    beta_end=BETA_END,
    prediction_type=PREDICTION_TARGET,
    clip_sample=False,  # latents are ~N(0,1); default clip to [-1,1] corrupts inference
)
print(f"Prediction target: {args.prediction_target} (scheduler prediction_type={PREDICTION_TARGET})")

scale_payload = {
    "emb_mean": float(data_mean),
    "emb_std": float(data_std),
    "cond_mean": cond_mean.tolist(),
    "cond_std": cond_std.tolist(),
    "num_continuous": NUM_CONTINUOUS,
    "station_emb_dim": STATION_EMB_DIM,
    "data_mode": args.data_mode,
    "data_shape": [int(data_shape[0]), int(data_shape[1]), int(data_shape[2])],
    "stft_freq_bins": int(data_shape[1]) if args.data_mode == "stft" else None,
    "stft_time_bins": int(data_shape[2]) if args.data_mode == "stft" else None,
}
json.dump(scale_payload, open("embeddings/scale.json", "w"))

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

# Per-step LR schedule — warmup is capped at 10% of total steps so short runs aren't hurt
MIN_LR_RATIO = 0.1
STEPS_PER_EPOCH = max(1, len(dataloader))
TOTAL_TRAIN_STEPS = NUM_EPOCHS * STEPS_PER_EPOCH
WARMUP_STEPS = min(200, max(TOTAL_TRAIN_STEPS // 10, 0))


def _lr_for_step(step_idx: int) -> float:
    if WARMUP_STEPS > 0 and step_idx < WARMUP_STEPS:
        return LR * float(step_idx + 1) / float(WARMUP_STEPS)
    if TOTAL_TRAIN_STEPS <= WARMUP_STEPS + 1:
        return LR
    cosine_progress = (step_idx - WARMUP_STEPS) / float(TOTAL_TRAIN_STEPS - WARMUP_STEPS - 1)
    cosine_progress = min(max(cosine_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return LR * (MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine)


print(
    "LR schedule: "
    f"warmup_steps={WARMUP_STEPS}, total_steps={TOTAL_TRAIN_STEPS}, "
    f"min_lr={LR * MIN_LR_RATIO:.2e}"
)
if args.log_images_every_n_batches > 0:
    print(f"Preview image logging: every {args.log_images_every_n_batches} batches")
else:
    print("Preview image logging: disabled")
if args.checkpoint_every_n_batches > 0:
    print(f"Batch checkpointing: every {args.checkpoint_every_n_batches} batches")
else:
    print("Batch checkpointing: disabled")


def _print_data_stats(tag: str, tensor: torch.Tensor, epoch: int):
    t = tensor.detach().cpu()
    mean_map = t.mean(dim=0)
    print(
        f"[Epoch {epoch + 1}] {tag} stats  "
        f"tensor_min={t.min().item():.6f} tensor_max={t.max().item():.6f}  "
        f"mean_map_min={mean_map.min().item():.6f} mean_map_max={mean_map.max().item():.6f}"
    )


def _normalise_for_tb_image(tensor: torch.Tensor) -> torch.Tensor:
    t = tensor.detach().cpu().float()
    lo = torch.quantile(t, 0.01)
    hi = torch.quantile(t, 0.99)
    if not torch.isfinite(lo) or not torch.isfinite(hi) or (hi - lo).abs().item() < 1e-8:
        lo = t.min()
        hi = t.max()
    if (hi - lo).abs().item() < 1e-8:
        return torch.zeros_like(t)
    return ((t - lo) / (hi - lo)).clamp(0.0, 1.0)


def _log_preview_images(log_step: int, epoch: int):
    gen_real = generate(
        fixed_real_cond,
        embedding_shape,
        noise_scheduler,
        NUM_TRAIN_TIMESTEPS,
        model,
        DEVICE,
        cond_mean=cond_mean,
        cond_std=cond_std,
        num_continuous=NUM_CONTINUOUS,
        data_mean=data_mean,
        data_std=data_std,
        guidance_scale=CFG_GUIDANCE_SCALE,
        training_type=TRAINING_TYPE,
    )
    if args.data_mode == "latent":
        vis_real = decode_embedding(gen_real)
    else:
        vis_real = gen_real
    _print_data_stats(f"real_cond step={log_step}", vis_real, epoch)
    writer.add_image("Generation/real_cond", _normalise_for_tb_image(vis_real), log_step)
    if fixed_real_stft is not None:
        _print_data_stats(f"real_stft step={log_step}", fixed_real_stft, epoch)
        writer.add_image("Generation/real_stft", _normalise_for_tb_image(fixed_real_stft), log_step)

    gen_rand = generate(
        fixed_rand_cond,
        embedding_shape,
        noise_scheduler,
        NUM_TRAIN_TIMESTEPS,
        model,
        DEVICE,
        cond_mean=cond_mean,
        cond_std=cond_std,
        num_continuous=NUM_CONTINUOUS,
        data_mean=data_mean,
        data_std=data_std,
        guidance_scale=CFG_GUIDANCE_SCALE,
        training_type=TRAINING_TYPE,
    )
    if args.data_mode == "latent":
        vis_rand = decode_embedding(gen_rand)
    else:
        vis_rand = gen_rand
    _print_data_stats(f"rand_cond step={log_step}", vis_rand, epoch)
    writer.add_image("Generation/rand_cond", _normalise_for_tb_image(vis_rand), log_step)


def _save_checkpoint(ckpt_name: str):
    ckpt_path = Path("checkpoints") / TRAINING_TYPE / ckpt_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_path))
    noise_scheduler.save_pretrained(str(ckpt_path))
    json.dump(
        {
            "data_mode": args.data_mode,
            "training_type": TRAINING_TYPE,
            "data_shape": [int(data_shape[0]), int(data_shape[1]), int(data_shape[2])],
            "emb_mean": float(data_mean),
            "emb_std": float(data_std),
            "num_continuous": NUM_CONTINUOUS,
            "station_emb_dim": STATION_EMB_DIM,
            "stft_freq_bins": int(data_shape[1]) if args.data_mode == "stft" else None,
            "stft_time_bins": int(data_shape[2]) if args.data_mode == "stft" else None,
        },
        open(ckpt_path / "training_config.json", "w"),
    )
    print(f"Checkpoint saved to {ckpt_path}")


def _cleanup_checkpoints(pattern: str, keep: int):
    keep = max(1, int(keep))
    all_ckpts = sorted((Path("checkpoints") / TRAINING_TYPE).glob(pattern), key=lambda p: p.stat().st_mtime)
    for old in all_ckpts[:-keep]:
        shutil.rmtree(old)


# --- Training Loop ---
embedding_shape = data_shape
global_step = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_data, batch_cond in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        batch_data = batch_data.to(DEVICE)
        batch_cond = batch_cond.to(DEVICE)

        # CFG: randomly zero-out each sample's conditioning independently
        if CFG_DROPOUT > 0:
            drop = torch.rand(batch_data.shape[0], 1, device=DEVICE) < CFG_DROPOUT
            batch_cond = batch_cond.masked_fill(drop, 0.0)

        step_lr = _lr_for_step(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = step_lr

        noise = torch.randn_like(batch_data)
        if TRAINING_TYPE == "flow_matching":
            t_cont = torch.rand(batch_data.shape[0], device=DEVICE)
            timesteps = (t_cont * NUM_TRAIN_TIMESTEPS).long().clamp(0, NUM_TRAIN_TIMESTEPS - 1)
            noisy_data = (1 - t_cont[:, None, None, None]) * batch_data + t_cont[:, None, None, None] * noise
            target = noise - batch_data
        else:
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_data.shape[0],), device=DEVICE,
            ).long()
            noisy_data = noise_scheduler.add_noise(batch_data, noise, timesteps)
            if args.prediction_target == "epsilon":
                target = noise
            elif args.prediction_target == "x0":
                target = batch_data
            elif args.prediction_target == "v_prediction":
                target = noise_scheduler.get_velocity(batch_data, noise, timesteps)
            else:
                raise ValueError(f"Unsupported prediction_target: {args.prediction_target}")

        # Metadata conditioning for cross-attention.
        cond = batch_cond.unsqueeze(1)

        model_pred = model.forward(noisy_data, timesteps, cond).sample

        loss = torch.nn.functional.mse_loss(model_pred, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1

        epoch_loss += loss.item()
        if args.log_images_every_n_batches > 0 and (global_step % args.log_images_every_n_batches == 0):
            _log_preview_images(global_step, epoch)
            model.train()
            _save_checkpoint(f"step_{global_step}")
            _cleanup_checkpoints("step_*", args.keep_last_batch_checkpoints)
        if args.checkpoint_every_n_batches > 0 and (global_step % args.checkpoint_every_n_batches == 0):
            _save_checkpoint(f"step_{global_step}")
            _cleanup_checkpoints("step_*", args.keep_last_batch_checkpoints)

    avg_loss = epoch_loss / len(dataloader)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Loss: {avg_loss:.6f}  LR: {current_lr:.2e}")
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("LR", current_lr, epoch)

    if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
        _save_checkpoint(f"epoch_{epoch + 1}")
        _cleanup_checkpoints("epoch_*", 3)

# --- Save model ---
writer.close()
model.save_pretrained(f"checkpoints/{TRAINING_TYPE}/unet2d")
noise_scheduler.save_pretrained(f"checkpoints/{TRAINING_TYPE}/unet2d")
print(f"Model saved to checkpoints/{TRAINING_TYPE}/unet2d")
