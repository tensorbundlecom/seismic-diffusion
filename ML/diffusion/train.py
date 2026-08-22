from model import DiffusionUNet2D, create_conditioning_vector
import torch
import json
import shutil
import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from diffusers import DDPMScheduler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import generate, decode_embedding
import wandb


EMBEDDINGS_DIR = Path("embeddings")
SOURCE_PATH = EMBEDDINGS_DIR / "source.json"
METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.pt"


def _read_json(path: Path) -> Any:
    """Read a JSON artifact with a useful error message for training inputs."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing required training artifact: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in training artifact {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_embedding_source(source: Any) -> Dict[str, Any]:
    """Validate the export contract needed to train safely from latent embeddings."""
    if not isinstance(source, dict):
        raise ValueError(f"{SOURCE_PATH} must contain a JSON object.")

    required = ("ae_checkpoint", "stft", "normalization_mode", "num_embeddings", "embedding_shape")
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"{SOURCE_PATH} is missing required fields: {', '.join(missing)}.")
    if not isinstance(source["stft"], dict):
        raise ValueError(f"{SOURCE_PATH}: 'stft' must be an object.")
    if not isinstance(source["ae_checkpoint"], str) or not source["ae_checkpoint"].strip():
        raise ValueError(f"{SOURCE_PATH}: 'ae_checkpoint' must be a non-empty string.")

    mode = str(source["normalization_mode"]).strip().lower()
    if mode not in {"global", "per_event"}:
        raise ValueError(
            f"{SOURCE_PATH}: unsupported normalization_mode={source['normalization_mode']!r}."
        )
    if mode == "global":
        try:
            global_min = float(source["global_min"])
            global_max = float(source["global_max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{SOURCE_PATH}: global normalization requires numeric global_min/global_max."
            ) from exc
        if not math.isfinite(global_min) or not math.isfinite(global_max) or global_max < global_min:
            raise ValueError(
                f"{SOURCE_PATH}: invalid global normalization bounds "
                f"({global_min}, {global_max})."
            )

    try:
        count = int(source["num_embeddings"])
        shape = [int(dim) for dim in source["embedding_shape"]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{SOURCE_PATH}: invalid num_embeddings or embedding_shape.") from exc
    if count <= 0 or len(shape) != 3 or any(dim <= 0 for dim in shape):
        raise ValueError(
            f"{SOURCE_PATH}: expected positive num_embeddings and a 3D positive embedding_shape; "
            f"got count={count}, shape={shape}."
        )
    channels = source.get("channels")
    if channels is not None and (not isinstance(channels, list) or not all(isinstance(x, str) for x in channels)):
        raise ValueError(f"{SOURCE_PATH}: 'channels' must be a list of strings when present.")
    return source


def _build_index_mappings(metadatas: List[Dict]) -> Dict[str, Dict[str, str]]:
    """Create immutable station/channel mappings and reject ambiguous exports."""
    station_index_to_name: Dict[str, str] = {}
    channel_index_to_type: Dict[str, str] = {}
    for i, metadata in enumerate(metadatas):
        try:
            station_idx = str(int(metadata["station_idx"]))
            station_name = str(metadata["station_name"])
            channel_idx = str(int(metadata.get("channel_idx", 0)))
            channel_type = str(metadata.get("channel_type", ""))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid station/channel metadata at row {i}: {metadata!r}") from exc
        if not station_name or not channel_type:
            raise ValueError(f"Missing station_name or channel_type in metadata row {i}.")
        if station_idx in station_index_to_name and station_index_to_name[station_idx] != station_name:
            raise ValueError(
                f"station_idx {station_idx} maps to both {station_index_to_name[station_idx]!r} "
                f"and {station_name!r}."
            )
        if channel_idx in channel_index_to_type and channel_index_to_type[channel_idx] != channel_type:
            raise ValueError(
                f"channel_idx {channel_idx} maps to both {channel_index_to_type[channel_idx]!r} "
                f"and {channel_type!r}."
            )
        station_index_to_name[station_idx] = station_name
        channel_index_to_type[channel_idx] = channel_type
    return {
        "station_index_to_name": dict(sorted(station_index_to_name.items(), key=lambda item: int(item[0]))),
        "channel_index_to_type": dict(sorted(channel_index_to_type.items(), key=lambda item: int(item[0]))),
    }


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
        resample_hz: float = 100.0,
        target_seconds: float = 70.0,
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
        self.resample_hz = float(resample_hz) if resample_hz else None
        self.target_seconds = float(target_seconds) if target_seconds else None
        self.target_samples = (
            int(round(self.resample_hz * self.target_seconds))
            if (self.resample_hz and self.target_seconds)
            else None
        )

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
            if self.resample_hz is not None and abs(trace.stats.sampling_rate - self.resample_hz) > 1e-6:
                trace.resample(self.resample_hz)
            data = trace.data.astype(np.float32)
            if self.target_samples is not None:
                n = self.target_samples
                data = data[:n] if data.shape[0] >= n else np.pad(data, (0, n - data.shape[0]), mode="constant")
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

    def estimate_stats(self, num_samples: int = 2048, indices: List[int] = None):
        n = len(self)
        if n == 0:
            raise RuntimeError("Empty STFT dataset")

        # Restrict statistics to the provided indices (e.g. train split only) to
        # avoid leaking validation data into the normalization constants.
        pool = list(indices) if indices is not None else list(range(n))
        if len(pool) == 0:
            raise RuntimeError("estimate_stats received an empty index pool")

        if num_samples <= 0 or num_samples >= len(pool):
            indices = pool
        else:
            picks = np.linspace(0, len(pool) - 1, num=num_samples, dtype=int).tolist()
            indices = [pool[i] for i in picks]

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
    defaults = {"nperseg": 256, "noverlap": 192, "nfft": 256, "resample_hz": 100.0, "target_seconds": 70.0}
    if not SOURCE_PATH.exists():
        print("[train] embeddings/source.json not found; using default STFT params.")
        return defaults

    try:
        payload = _read_json(SOURCE_PATH)
        stft = payload.get("stft", {})
        return {
            "nperseg": int(stft.get("nperseg", defaults["nperseg"])),
            "noverlap": int(stft.get("noverlap", defaults["noverlap"])),
            "nfft": int(stft.get("nfft", defaults["nfft"])),
            "resample_hz": stft.get("resample_hz", defaults["resample_hz"]),
            "target_seconds": stft.get("target_seconds", defaults["target_seconds"]),
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
    "--experiment_name",
    type=str,
    default=None,
    help=(
        "Optional name for this experiment. When set, checkpoints are saved under "
        "checkpoints/<training_type>/<experiment_name>/ and TensorBoard logs under "
        "the corresponding named run."
    ),
)
parser.add_argument(
    "--use_wandb",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Log metrics, config, and generated previews to Weights & Biases "
        "(default: enabled). Use --no-use_wandb to disable."
    ),
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="seismic-diffusion",
    help="wandb project name to log runs under.",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default=None,
    help="wandb entity/username (defaults to your logged-in account).",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="DataLoader workers (mainly relevant for --data_mode stft).",
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=500,
    help="Number of training epochs (default: 500).",
)
parser.add_argument(
    "--use_vs30",
    type=str,
    default="false",
    help=(
        "Add the per-station Vs30 (site condition, m/s) as an extra continuous "
        "conditioning feature. Use --use_vs30 true or --use_vs30 false "
        "(default: false). Requires the station Vs30 lookup "
        "(see compute_station_vs30.py) when true."
    ),
)
parser.add_argument(
    "--include_station_id",
    type=str,
    default="true",
    help=(
        "Condition the diffusion model on a learned per-station ID embedding. "
        "Use --include_station_id true or --include_station_id false "
        "(default: true). Any case accepted."
    ),
)
parser.add_argument(
    "--num_held_out_stations",
    type=int,
    default=5,
    help=(
        "When >0, hold out this many random stations (chosen by --split_seed) "
        "entirely from training and validation so their loss can be monitored as "
        "an unseen-station generalization signal. Most useful with "
        "--no-include_station_id. Set <=0 to disable."
    ),
)
parser.add_argument(
    "--station_vs30",
    type=str,
    default="embeddings/station_vs30.json",
    help="Path to the station -> Vs30 JSON lookup used when --use_vs30 is set.",
)
parser.add_argument(
    "--val_fraction",
    type=float,
    default=0.1,
    help="Fraction of the dataset held out for validation. Set <=0 to disable the split.",
)
parser.add_argument(
    "--split_seed",
    type=int,
    default=42,
    help="Seed for the train/val split (also seeds the validation noise for comparable loss).",
)
parser.add_argument(
    "--val_every_n_epochs",
    type=int,
    default=1,
    help="Run a validation pass every N epochs. Set <=0 to disable validation logging.",
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
args.include_station_id = str(args.include_station_id).strip().lower() == "true"
args.use_vs30 = str(args.use_vs30).strip().lower() == "true"

# --- Config ---
NUM_EPOCHS = int(args.num_epochs)
if NUM_EPOCHS <= 0:
    raise ValueError(f"--num_epochs must be positive; got {NUM_EPOCHS}.")
BATCH_SIZE = 32
LR = 1e-4
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
CHECKPOINT_EVERY_N_EPOCHS = 1
MIN_LR_RATIO = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREDICTION_TARGET = "sample" if args.prediction_target == "x0" else args.prediction_target  # HF scheduler name
STATION_EMB_DIM = 64
CHANNEL_EMB_DIM = 16
# Base continuous features: magnitude, 2D distance, sin/cos azimuth, depth, snr.
# Vs30 (site condition) is appended as a 7th continuous feature when --use_vs30 is set.
NUM_CONTINUOUS = 7 if args.use_vs30 else 6
TRAINING_TYPE = args.training_type
VAL_EVERY_N_EPOCHS = int(args.val_every_n_epochs)
NUM_HELD_OUT_STATIONS = int(args.num_held_out_stations)

if args.experiment_name is not None:
    EXPERIMENT_NAME = args.experiment_name.strip()
    if not EXPERIMENT_NAME:
        raise ValueError("--experiment_name must not be empty.")
    if Path(EXPERIMENT_NAME).name != EXPERIMENT_NAME or EXPERIMENT_NAME in {".", ".."}:
        raise ValueError("--experiment_name must be a single directory name (no path separators).")
else:
    EXPERIMENT_NAME = None

CHECKPOINT_ROOT = Path("checkpoints") / TRAINING_TYPE
TENSORBOARD_LOG_DIR = Path("runs") / f"diffusion_{args.data_mode}_{TRAINING_TYPE}"
if EXPERIMENT_NAME is not None:
    CHECKPOINT_ROOT /= EXPERIMENT_NAME
    TENSORBOARD_LOG_DIR /= EXPERIMENT_NAME

writer = SummaryWriter(log_dir=str(TENSORBOARD_LOG_DIR))
print(f"[train] experiment={EXPERIMENT_NAME or 'default'}, tensorboard={TENSORBOARD_LOG_DIR}")

wb = None
if args.use_wandb:
    run_name = EXPERIMENT_NAME or f"diffusion_{args.data_mode}_{TRAINING_TYPE}"
    wb = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            "training_type": TRAINING_TYPE,
            "diffusion_mode": args.training_type,
            "prediction_target": args.prediction_target,
            "hf_prediction_type": PREDICTION_TARGET,
            "data_mode": args.data_mode,
            "experiment_name": EXPERIMENT_NAME,
            "all_cli_args": vars(args),
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": 1e-2,
            "min_lr_ratio": MIN_LR_RATIO,
            "num_train_timesteps": NUM_TRAIN_TIMESTEPS,
            "beta_start": BETA_START,
            "beta_end": BETA_END,
            "num_continuous": NUM_CONTINUOUS,
            "station_emb_dim": STATION_EMB_DIM,
            "channel_emb_dim": CHANNEL_EMB_DIM,
            "device": DEVICE,
            "include_station_id": bool(args.include_station_id),
            "use_vs30": bool(args.use_vs30),
            "num_workers": args.num_workers,
            "num_epochs": NUM_EPOCHS,
            "val_fraction": float(args.val_fraction),
            "split_seed": int(args.split_seed),
            "val_every_n_epochs": VAL_EVERY_N_EPOCHS,
            "stft_stats_samples": int(args.stft_stats_samples),
            "stft_freq_bins": int(args.stft_freq_bins),
            "stft_time_bins": int(args.stft_time_bins),
            "log_images_every_n_batches": args.log_images_every_n_batches,
            "checkpoint_every_n_batches": args.checkpoint_every_n_batches,
            "keep_last_batch_checkpoints": args.keep_last_batch_checkpoints,
            "station_vs30": args.station_vs30,
            "wandb_project": args.wandb_project,
            "wandb_entity": args.wandb_entity,
        },
    )
    print(f"[train] wandb tracking: {wb.project}/{wb.id} ({wb.name})")

# --- Shared metadata conditioning / embedding provenance ---
# Read and validate the source artifact before deriving any split or statistics.
# This prevents an AE/embedding export mismatch from being silently recorded in a
# diffusion checkpoint that later cannot be reconstructed.
embedding_source = _validate_embedding_source(_read_json(SOURCE_PATH))
embedding_source_sha256 = _sha256_file(SOURCE_PATH)
recorded_ae_path = Path(embedding_source["ae_checkpoint"]).expanduser()
ae_candidates = [recorded_ae_path]
if not recorded_ae_path.is_absolute():
    ae_candidates = [Path.cwd() / recorded_ae_path, SOURCE_PATH.parent / recorded_ae_path]
ae_checkpoint_path = next((path.resolve() for path in ae_candidates if path.is_file()), None)
if ae_checkpoint_path is None:
    raise FileNotFoundError(
        f"AE checkpoint recorded by {SOURCE_PATH} does not exist: "
        f"{embedding_source['ae_checkpoint']!r}."
    )
ae_checkpoint_sha256 = _sha256_file(ae_checkpoint_path)
metadatas = _read_json(METADATA_PATH)
if not isinstance(metadatas, list):
    raise ValueError(f"{METADATA_PATH} must contain a JSON list.")
if not metadatas:
    raise ValueError(f"{METADATA_PATH} is empty.")
index_mappings = _build_index_mappings(metadatas)

if int(embedding_source["num_embeddings"]) != len(metadatas):
    raise ValueError(
        f"Embedding source count ({embedding_source['num_embeddings']}) != metadata count "
        f"({len(metadatas)}). Recreate embeddings so their artifacts match."
    )

station_locations_path = EMBEDDINGS_DIR / "station_locations.json"
if not station_locations_path.exists():
    raise FileNotFoundError(
        f"Missing {station_locations_path}. Run fetch_station_locations.py first."
    )
station_locations = _read_json(station_locations_path)

station_vs30 = None
if args.use_vs30:
    vs30_path = Path(args.station_vs30)
    if not vs30_path.exists():
        raise FileNotFoundError(
            f"Missing {vs30_path}. Run compute_station_vs30.py first to build the "
            "station -> Vs30 lookup."
        )
    station_vs30 = _read_json(vs30_path)
    print(f"[train] Vs30 conditioning enabled ({len(station_vs30)} stations).")

raw_cond_vectors = torch.stack(
    [
        create_conditioning_vector(
            m,
            station_locations,
            station_vs30,
            include_station_id=args.include_station_id,
        )
        for m in metadatas
    ]
)

# --- Train/val split ---
# A single reproducible permutation drives the split for both data modes.
num_total = len(raw_cond_vectors)
val_fraction = float(args.val_fraction)
if val_fraction < 0 or val_fraction >= 1:
    raise ValueError(f"--val_fraction must be in [0, 1); got {val_fraction}.")
split_generator = torch.Generator().manual_seed(int(args.split_seed))
perm = torch.randperm(num_total, generator=split_generator).tolist()
num_val = int(round(num_total * val_fraction))
val_indices = sorted(perm[:num_val])
train_indices = sorted(perm[num_val:])
if len(train_indices) == 0:
    raise ValueError("Train split is empty; lower --val_fraction.")
print(
    f"[train] split: {len(train_indices)} train / {len(val_indices)} val "
    f"(val_fraction={val_fraction}, seed={args.split_seed})"
)

# --- Held-out station evaluation ---
# When training without station-id conditioning, hold out a few random stations
# entirely from training (and the regular val split) so we can monitor how the
# model generalizes to unseen stations. The chosen stations are reproducible via
# --split_seed.
held_out_indices: List[int] = []
held_out_station_ids: List[int] = []
if NUM_HELD_OUT_STATIONS > 0:
    station_ids = sorted({int(m["station_idx"]) for m in metadatas})
    if len(station_ids) <= NUM_HELD_OUT_STATIONS:
        raise ValueError(
            f"--num_held_out_stations={NUM_HELD_OUT_STATIONS} must be less than the "
            f"total number of stations ({len(station_ids)})."
        )
    station_id_to_indices: Dict[int, List[int]] = {}
    for i, m in enumerate(metadatas):
        station_id_to_indices.setdefault(int(m["station_idx"]), []).append(i)

    _rng = np.random.default_rng(int(args.split_seed))
    held_out_station_ids = sorted(
        _rng.choice(station_ids, size=NUM_HELD_OUT_STATIONS, replace=False).tolist()
    )
    for sid in held_out_station_ids:
        held_out_indices.extend(station_id_to_indices[sid])
    held_out_indices = sorted(held_out_indices)
    held_out_set = set(held_out_indices)
    train_indices = [i for i in train_indices if i not in held_out_set]
    val_indices = [i for i in val_indices if i not in held_out_set]

    if len(train_indices) == 0:
        raise ValueError("Train split is empty after removing held-out stations.")
    print(
        f"[train] held-out stations ({NUM_HELD_OUT_STATIONS}): {held_out_station_ids} "
        f"-> {len(held_out_indices)} samples excluded from train/val"
    )

# Fit continuous conditioning statistics only after the final train split is
# known. In particular, validation and held-out stations must not influence
# z-score parameters used by the model.
train_raw_continuous = raw_cond_vectors[train_indices, :NUM_CONTINUOUS]
cond_mean = train_raw_continuous.mean(dim=0)
cond_std = train_raw_continuous.std(dim=0, unbiased=False).clamp(min=1e-8)
if not torch.isfinite(cond_mean).all() or not torch.isfinite(cond_std).all():
    raise ValueError("Non-finite train-only conditioning normalization statistics.")
cond_vectors = raw_cond_vectors.clone()
cond_vectors[:, :NUM_CONTINUOUS] = (
    cond_vectors[:, :NUM_CONTINUOUS] - cond_mean
) / cond_std
print(
    f"[train] conditioning normalization fit on {len(train_indices)} final train samples "
    f"(validation and held-out stations excluded)."
)

# Keep the fixed preview sample inside the train split for stable monitoring.
fixed_real_idx = train_indices[0]
fixed_real_cond = raw_cond_vectors[fixed_real_idx]
fixed_rand_idx = torch.randint(len(raw_cond_vectors), (1,)).item()
fixed_rand_cond = raw_cond_vectors[fixed_rand_idx]
fixed_real_stft = None

# --- Data mode specific setup ---
if args.data_mode == "latent":
    data_tensor = torch.load(EMBEDDINGS_PATH, map_location="cpu").float()
    if len(data_tensor) != len(cond_vectors):
        raise ValueError(
            f"Embeddings count ({len(data_tensor)}) != metadata count ({len(cond_vectors)})."
        )
    if data_tensor.ndim != 4:
        raise ValueError(
            f"Expected 4D embedding tensor (N, C, H, W), got shape {tuple(data_tensor.shape)}."
        )
    expected_embedding_shape = tuple(int(dim) for dim in embedding_source["embedding_shape"])
    if tuple(data_tensor.shape[1:]) != expected_embedding_shape:
        raise ValueError(
            f"Embedding tensor shape {tuple(data_tensor.shape[1:])} != source.json "
            f"embedding_shape {expected_embedding_shape}. Recreate embeddings so the artifacts match."
        )
    if not torch.isfinite(data_tensor).all():
        raise ValueError("Embedding tensor contains non-finite values.")

    # Standardize using *only* final train examples. A non-zero latent mean is
    # normal for an AE and must be removed before diffusion training.
    train_embeddings = data_tensor[train_indices]
    data_mean = float(train_embeddings.mean().item())
    data_std = float(train_embeddings.std(unbiased=False).clamp(min=1e-8).item())
    if not math.isfinite(data_mean) or not math.isfinite(data_std):
        raise ValueError("Non-finite train-only latent normalization statistics.")
    train_data = (data_tensor - data_mean) / data_std
    fixed_real_stft = decode_embedding(data_tensor[fixed_real_idx])
    full_dataset = TensorDataset(train_data, cond_vectors)
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices) if val_indices else None
    held_out_dataset = Subset(full_dataset, held_out_indices) if held_out_indices else None
    data_shape = tuple(train_data.shape[1:])
    num_workers = 0

    print(
        f"[train] data_mode=latent, shape={data_shape}, "
        f"mean={data_mean:.6f}, std={data_std:.6f} (fit on final train split only)"
    )

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
        resample_hz=stft_cfg["resample_hz"],
        target_seconds=stft_cfg["target_seconds"],
    )

    data_mean, data_std = stft_dataset.estimate_stats(
        args.stft_stats_samples, indices=train_indices
    )
    stft_dataset.set_normalization(data_mean, data_std)
    fixed_real_stft = stft_dataset._compute_raw_stft(fixed_real_idx)
    x0, _ = stft_dataset[0]
    data_shape = tuple(x0.shape)
    train_dataset = Subset(stft_dataset, train_indices)
    val_dataset = Subset(stft_dataset, val_indices) if val_indices else None
    held_out_dataset = Subset(stft_dataset, held_out_indices) if held_out_indices else None
    num_workers = max(0, int(args.num_workers))

    print(
        f"[train] data_mode=stft, shape={data_shape}, "
        f"mean={data_mean:.6f}, std={data_std:.6f}, "
        f"target_override=(freq={target_freq_bins}, time={target_time_bins})"
    )

dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(DEVICE == "cuda"),
    persistent_workers=(num_workers > 0),
)

val_dataloader = None
if val_dataset is not None and len(val_dataset) > 0:
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(num_workers > 0),
    )

held_out_dataloader = None
if held_out_dataset is not None and len(held_out_dataset) > 0:
    held_out_dataloader = DataLoader(
        held_out_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(num_workers > 0),
    )

# --- Model & Scheduler ---
num_stations = max(int(m["station_idx"]) for m in metadatas) + 1 if args.include_station_id else 0
num_channels = max(int(m.get("channel_idx", 0)) for m in metadatas) + 1
model = DiffusionUNet2D(
    in_channels=int(data_shape[0]),
    out_channels=int(data_shape[0]),
    num_stations=num_stations,
    station_emb_dim=STATION_EMB_DIM,
    include_station_id=args.include_station_id,
    num_continuous=NUM_CONTINUOUS,
    num_channels=num_channels,
    channel_emb_dim=CHANNEL_EMB_DIM,
)
model.to(DEVICE)
print(
    f"Conditioning: metadata + channel-type embedding (num_channels={num_channels}), "
    f"station_id={'on' if args.include_station_id else 'off'}, "
    f"num_continuous={NUM_CONTINUOUS}, vs30={'on' if args.use_vs30 else 'off'}"
)
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
    "schema_version": 2,
    "emb_mean": float(data_mean),
    "emb_std": float(data_std),
    "data_normalization": {
        "mean": float(data_mean),
        "std": float(data_std),
        "fit_split": "train",
    },
    "cond_mean": cond_mean.tolist(),
    "cond_std": cond_std.tolist(),
    "conditioning_normalization": {
        "mean": cond_mean.tolist(),
        "std": cond_std.tolist(),
        "num_continuous": NUM_CONTINUOUS,
        "fit_split": "train",
    },
    "num_continuous": NUM_CONTINUOUS,
    "use_vs30": bool(args.use_vs30),
    "include_station_id": bool(args.include_station_id),
    "station_emb_dim": STATION_EMB_DIM,
    "num_channels": num_channels,
    "channel_emb_dim": CHANNEL_EMB_DIM,
    "data_mode": args.data_mode,
    "experiment_name": EXPERIMENT_NAME,
    "data_shape": [int(data_shape[0]), int(data_shape[1]), int(data_shape[2])],
    "stft_freq_bins": int(data_shape[1]) if args.data_mode == "stft" else None,
    "stft_time_bins": int(data_shape[2]) if args.data_mode == "stft" else None,
    "split_seed": int(args.split_seed),
    "val_fraction": float(val_fraction),
    "train_indices": train_indices,
    "val_indices": val_indices,
    "held_out_indices": held_out_indices,
    "held_out_station_ids": held_out_station_ids,
    "mappings": index_mappings,
}
embedding_provenance = {
    "schema_version": 1,
    "source_path": str(SOURCE_PATH.resolve()),
    "source_sha256": embedding_source_sha256,
    "source": embedding_source,
    "ae_checkpoint": str(ae_checkpoint_path),
    "ae_checkpoint_sha256": ae_checkpoint_sha256,
    # Direct aliases make the critical inverse-normalization contract easy for
    # inference tools to consume without unpacking the nested source snapshot.
    "normalization_mode": embedding_source["normalization_mode"],
    "global_min": embedding_source.get("global_min"),
    "global_max": embedding_source.get("global_max"),
    "normalization": {
        "mode": embedding_source["normalization_mode"],
        "global_min": embedding_source.get("global_min"),
        "global_max": embedding_source.get("global_max"),
    },
    "stft": embedding_source["stft"],
    "channels": embedding_source.get("channels"),
    "num_embeddings": int(embedding_source["num_embeddings"]),
    "embedding_shape": [int(dim) for dim in embedding_source["embedding_shape"]],
}
with (EMBEDDINGS_DIR / "scale.json").open("w", encoding="utf-8") as handle:
    json.dump(scale_payload, handle, indent=2)
if wb is not None:
    wb.config.update(
        {
            "data_shape": scale_payload["data_shape"],
            "emb_mean": float(data_mean),
            "emb_std": float(data_std),
            "num_train": len(train_indices),
            "num_val": len(val_indices),
            "num_stations": num_stations,
            "num_channels": num_channels,
            "num_held_out_stations": len(held_out_station_ids),
            "held_out_station_ids": held_out_station_ids,
            "num_held_out_samples": len(held_out_indices),
            "stft_freq_bins_resolved": scale_payload.get("stft_freq_bins"),
            "stft_time_bins_resolved": scale_payload.get("stft_time_bins"),
            "total_train_steps": NUM_EPOCHS * max(1, len(dataloader)),
        }
    )

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

# Per-step LR schedule — warmup is capped at 10% of total steps so short runs aren't hurt
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
        training_type=TRAINING_TYPE,
    )
    if args.data_mode == "latent":
        vis_real = decode_embedding(gen_real)
    else:
        vis_real = gen_real
    _print_data_stats(f"real_cond step={log_step}", vis_real, epoch)
    writer.add_image("Generation/real_cond", _normalise_for_tb_image(vis_real), log_step)
    ims = {"Generation/real_cond": wandb.Image(_normalise_for_tb_image(vis_real))}
    if fixed_real_stft is not None:
        _print_data_stats(f"real_stft step={log_step}", fixed_real_stft, epoch)
        writer.add_image("Generation/real_stft", _normalise_for_tb_image(fixed_real_stft), log_step)
        ims["Generation/real_stft"] = wandb.Image(_normalise_for_tb_image(fixed_real_stft))

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
        training_type=TRAINING_TYPE,
    )
    if args.data_mode == "latent":
        vis_rand = decode_embedding(gen_rand)
    else:
        vis_rand = gen_rand
    _print_data_stats(f"rand_cond step={log_step}", vis_rand, epoch)
    writer.add_image("Generation/rand_cond", _normalise_for_tb_image(vis_rand), log_step)
    ims["Generation/rand_cond"] = wandb.Image(_normalise_for_tb_image(vis_rand))
    if wb is not None:
        wb.log(ims, step=log_step)


def _save_checkpoint(ckpt_name: str):
    ckpt_path = CHECKPOINT_ROOT / ckpt_name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_path))
    noise_scheduler.save_pretrained(str(ckpt_path))
    training_config = {
        "schema_version": 2,
        "data_mode": args.data_mode,
        "training_type": TRAINING_TYPE,
        "experiment_name": EXPERIMENT_NAME,
        "data_shape": [int(data_shape[0]), int(data_shape[1]), int(data_shape[2])],
        # Legacy aliases remain so existing consumers continue to work.
        "emb_mean": float(data_mean),
        "emb_std": float(data_std),
        "data_normalization": scale_payload["data_normalization"],
        "conditioning_normalization": scale_payload["conditioning_normalization"],
        "num_continuous": NUM_CONTINUOUS,
        "station_emb_dim": STATION_EMB_DIM,
        "include_station_id": bool(args.include_station_id),
        "num_channels": num_channels,
        "channel_emb_dim": CHANNEL_EMB_DIM,
        "stft_freq_bins": int(data_shape[1]) if args.data_mode == "stft" else None,
        "stft_time_bins": int(data_shape[2]) if args.data_mode == "stft" else None,
        # A complete, checkpoint-local record of the embedding export is the
        # source of truth at inference time. Never rely on a future mutable
        # embeddings/source.json when reconstructing this model.
        "embedding_provenance": embedding_provenance,
        "embedding_scale": scale_payload,
        "split": {
            "split_seed": int(args.split_seed),
            "val_fraction": float(val_fraction),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "held_out_indices": held_out_indices,
            "held_out_station_ids": held_out_station_ids,
        },
        "mappings": index_mappings,
    }
    with (ckpt_path / "training_config.json").open("w", encoding="utf-8") as handle:
        json.dump(training_config, handle, indent=2)
    # Sidecar copies make provenance discoverable without parsing a large
    # training_config and keep the checkpoint self-contained when embeddings
    # are regenerated for a subsequent experiment.
    with (ckpt_path / "embedding_source.json").open("w", encoding="utf-8") as handle:
        json.dump(embedding_source, handle, indent=2)
    with (ckpt_path / "embedding_scale.json").open("w", encoding="utf-8") as handle:
        json.dump(scale_payload, handle, indent=2)
    print(f"Checkpoint saved to {ckpt_path}")


def _cleanup_checkpoints(pattern: str, keep: int):
    keep = max(1, int(keep))
    all_ckpts = sorted(CHECKPOINT_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime)
    for old in all_ckpts[:-keep]:
        shutil.rmtree(old)


def _diffusion_loss(batch_data, batch_cond, generator=None):
    """Compute the training objective for one batch (shared by train and val)."""
    noise = torch.randn(batch_data.shape, device=batch_data.device, generator=generator)
    if TRAINING_TYPE == "flow_matching":
        t_cont = torch.rand(batch_data.shape[0], device=batch_data.device, generator=generator)
        timesteps = (t_cont * NUM_TRAIN_TIMESTEPS).long().clamp(0, NUM_TRAIN_TIMESTEPS - 1)
        noisy_data = (1 - t_cont[:, None, None, None]) * batch_data + t_cont[:, None, None, None] * noise
        target = noise - batch_data
    else:
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (batch_data.shape[0],),
            device=batch_data.device, generator=generator,
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

    cond = batch_cond.unsqueeze(1)
    model_pred = model.forward(noisy_data, timesteps, cond).sample
    return torch.nn.functional.mse_loss(model_pred, target)


@torch.no_grad()
def _evaluate(loader) -> float:
    """Average validation loss with a fixed noise seed for epoch-to-epoch comparability."""
    model.eval()
    # CUDA RNG generators must live on the data device; CPU otherwise.
    gen = torch.Generator(device=DEVICE).manual_seed(int(args.split_seed))
    total_loss = 0.0
    total_count = 0
    for batch_data, batch_cond in loader:
        batch_data = batch_data.to(DEVICE)
        batch_cond = batch_cond.to(DEVICE)
        loss = _diffusion_loss(batch_data, batch_cond, generator=gen)
        total_loss += loss.item() * batch_data.shape[0]
        total_count += batch_data.shape[0]
    model.train()
    return total_loss / max(1, total_count)


# --- Training Loop ---
embedding_shape = data_shape
global_step = 0
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_data, batch_cond in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
        batch_data = batch_data.to(DEVICE)
        batch_cond = batch_cond.to(DEVICE)

        step_lr = _lr_for_step(global_step)
        for pg in optimizer.param_groups:
            pg["lr"] = step_lr

        loss = _diffusion_loss(batch_data, batch_cond)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        global_step += 1

        epoch_loss += loss.item()
        if wb is not None:
            wb.log({"Loss/train_step": loss.item(), "lr": step_lr}, step=global_step)
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
    if wb is not None:
        wb.log({"Loss/train": avg_loss, "lr": current_lr}, step=((epoch + 1) * len(dataloader)))

    if (
        val_dataloader is not None
        and VAL_EVERY_N_EPOCHS > 0
        and (epoch + 1) % VAL_EVERY_N_EPOCHS == 0
    ):
        val_loss = _evaluate(val_dataloader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Val Loss: {val_loss:.6f}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        if wb is not None:
            wb.log({"Loss/val": val_loss}, step=((epoch + 1) * len(dataloader)))

    if held_out_dataloader is not None and VAL_EVERY_N_EPOCHS > 0 and (
        epoch + 1
    ) % VAL_EVERY_N_EPOCHS == 0:
        held_out_loss = _evaluate(held_out_dataloader)
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - Held-out Station Loss: {held_out_loss:.6f}"
        )
        writer.add_scalar("Loss/val_heldout", held_out_loss, epoch)
        if wb is not None:
            wb.log(
                {"Loss/val_heldout": held_out_loss},
                step=((epoch + 1) * len(dataloader)),
            )

    if (epoch + 1) % CHECKPOINT_EVERY_N_EPOCHS == 0:
        _save_checkpoint(f"epoch_{epoch + 1}")
        _cleanup_checkpoints("epoch_*", 3)

# --- Save model ---
writer.close()
if wb is not None:
    wb.finish()
final_model_path = CHECKPOINT_ROOT / "unet2d"
_save_checkpoint("unet2d")
print(f"Model saved to {final_model_path}")
