#!/usr/bin/env python
"""
Create diffusion-training embeddings from a trained autoencoder checkpoint.
"""
import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ML.autoencoder.inference import load_model
from ML.autoencoder.stft_dataset_with_metadata import SeismicSTFTDatasetWithMetadata


def _find_latest_timestamped_ae_checkpoint() -> Path:
    """
    Return latest best_model.pt from timestamped AE checkpoint dirs only.

    Ignores temporary/non-timestamped folders such as _tmp_*.
    """
    ckpt_root = Path(__file__).resolve().parent.parent / "autoencoder" / "checkpoints"
    timestamp_pat = re.compile(r"^\d{8}_\d{6}$")
    ckpts = sorted(
        p for p in ckpt_root.glob("*/best_model.pt")
        if timestamp_pat.match(p.parent.name)
    )
    if not ckpts:
        raise FileNotFoundError(
            f"No timestamped AE checkpoint found under: {ckpt_root}"
        )
    return ckpts[-1]


def parse_args():
    default_output = Path(__file__).resolve().parent / "embeddings"

    parser = argparse.ArgumentParser(description="Create embeddings for diffusion training")
    parser.add_argument(
        "--ae_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to autoencoder best_model.pt checkpoint. "
            "If omitted, the latest timestamped checkpoint is used."
        ),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../../data/filtered_waveforms",
        help="Path to filtered waveform directory",
    )
    parser.add_argument(
        "--event_file",
        type=str,
        default="../../data/events/20140101_20251101_0.0_9.0_9_339.txt",
        help="Path to event catalog file",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Channel groups to include (e.g. HH HN). If omitted, uses the "
            "channels saved in the AE checkpoint (HH for legacy checkpoints)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output),
        help="Directory where embeddings.pt / metadata.json / source.json are saved",
    )
    parser.add_argument(
        "--waveform_summary",
        type=str,
        default=None,
        help="Path to waveform_summary.csv; if provided, snr_max is added to each metadata entry",
    )
    parser.add_argument(
        "--target_freq_bins",
        type=int,
        default=None,
        help="Resize STFT frequency axis to this many bins. Must match the value used "
             "when training the autoencoder. None = keep native bins.",
    )
    parser.add_argument(
        "--target_time_bins",
        type=int,
        default=None,
        help="Resize STFT time axis to this many bins. Must match the autoencoder. "
             "None = keep native bins.",
    )
    parser.add_argument(
        "--resample_hz",
        type=float,
        default=None,
        help="Resample every trace to this rate before STFT. If omitted, uses the "
             "AE checkpoint value (100 Hz for legacy checkpoints). 0 disables.",
    )
    parser.add_argument(
        "--target_seconds",
        type=float,
        default=None,
        help="Trim/zero-pad each resampled trace to round(resample_hz*target_seconds) "
             "samples. If omitted, uses the AE checkpoint value (70 s for legacy checkpoints).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for autoencoder inference",
    )
    return parser.parse_args()


def _checkpoint_field(
    checkpoint_path: Path,
    config: Dict[str, Any],
    field_name: str,
    raw_checkpoint: Optional[Dict[str, Any]] = None,
) -> tuple[Any, Optional[Dict[str, Any]]]:
    """Read a field from the checkpoint config, with top-level legacy fallback."""
    value = config.get(field_name)
    if value is not None:
        return value, raw_checkpoint

    if raw_checkpoint is None:
        raw_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return raw_checkpoint.get(field_name), raw_checkpoint


def _normalization_from_checkpoint(
    checkpoint_path: Path, config: Dict[str, Any]
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Resolve AE input normalization without ever fitting on export samples.

    New checkpoints store these values in ``config``. The top-level fallback
    keeps checkpoints saved during the normalization rollout compatible.
    """
    raw_checkpoint = None
    mode, raw_checkpoint = _checkpoint_field(
        checkpoint_path, config, "normalization_mode", raw_checkpoint
    )
    global_flag, raw_checkpoint = _checkpoint_field(
        checkpoint_path, config, "global_normalization", raw_checkpoint
    )

    if mode is None:
        is_global = bool(global_flag)
        normalized_mode = "global" if is_global else "per_event"
    else:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"global", "per_event"}:
            raise ValueError(
                "Unsupported AE checkpoint normalization_mode "
                f"{mode!r}; expected 'global' or 'per_event'."
            )
        is_global = normalized_mode == "global"

    result = {
        "mode": normalized_mode,
        "global_min": None,
        "global_max": None,
    }
    if not is_global:
        return result, raw_checkpoint

    global_min, raw_checkpoint = _checkpoint_field(
        checkpoint_path, config, "global_min", raw_checkpoint
    )
    global_max, raw_checkpoint = _checkpoint_field(
        checkpoint_path, config, "global_max", raw_checkpoint
    )
    try:
        global_min = float(global_min)
        global_max = float(global_max)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "The globally normalized AE checkpoint is missing usable global_min/global_max "
            "statistics. Re-export requires the exact bounds saved during AE training."
        ) from error
    if not math.isfinite(global_min) or not math.isfinite(global_max):
        raise ValueError(
            "The globally normalized AE checkpoint has non-finite global_min/global_max statistics."
        )
    if global_max < global_min:
        raise ValueError(
            "The globally normalized AE checkpoint has invalid bounds: "
            f"global_max ({global_max}) is below global_min ({global_min})."
        )
    result["global_min"] = global_min
    result["global_max"] = global_max
    return result, raw_checkpoint


def _resolve_preprocessing(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Use explicit CLI settings or, by default, the AE's saved preprocessing."""
    channels = args.channels if args.channels is not None else config.get("channels", ["HH"])
    if not channels:
        channels = ["HH"]

    def option_or_checkpoint(option_name: str, legacy_default: Any) -> Any:
        value = getattr(args, option_name)
        return value if value is not None else config.get(option_name, legacy_default)

    return {
        "channels": list(channels),
        "nperseg": int(config.get("nperseg", 256)),
        "noverlap": int(config.get("noverlap", 192)),
        "nfft": int(config.get("nfft", 256)),
        "target_freq_bins": option_or_checkpoint("target_freq_bins", None),
        "target_time_bins": option_or_checkpoint("target_time_bins", None),
        "resample_hz": option_or_checkpoint("resample_hz", 100.0),
        "target_seconds": option_or_checkpoint("target_seconds", 70.0),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    ae_ckpt = Path(args.ae_checkpoint).expanduser().resolve() if args.ae_checkpoint else _find_latest_timestamped_ae_checkpoint()
    if not ae_ckpt.exists():
        raise FileNotFoundError(f"AE checkpoint does not exist: {ae_ckpt}")

    print(f"Loading AE checkpoint: {ae_ckpt}")
    model, config = load_model(str(ae_ckpt), device=str(device))
    model.eval()

    normalization, _ = _normalization_from_checkpoint(ae_ckpt, config)
    preprocessing = _resolve_preprocessing(args, config)
    nperseg = preprocessing["nperseg"]
    noverlap = preprocessing["noverlap"]
    nfft = preprocessing["nfft"]
    print(
        "Using STFT params from AE config: "
        f"nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}"
    )
    print(
        "Using AE input normalization: "
        f"{normalization['mode']}"
        + (
            f" (min={normalization['global_min']:.6g}, "
            f"max={normalization['global_max']:.6g})"
            if normalization["mode"] == "global"
            else ""
        )
    )

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=preprocessing["channels"],
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        normalize=True,
        log_scale=True,
        target_freq_bins=preprocessing["target_freq_bins"],
        target_time_bins=preprocessing["target_time_bins"],
        resample_hz=preprocessing["resample_hz"],
        target_seconds=preprocessing["target_seconds"],
        global_normalization=normalization["mode"] == "global",
    )
    if normalization["mode"] == "global":
        # Embedding export must use the AE's training bounds, never bounds fit
        # over the full embedding dataset (which would leak held-out samples).
        dataset.set_global_normalization_stats(
            normalization["global_min"], normalization["global_max"]
        )

    snr_lookup = {}
    if args.waveform_summary:
        summary = pd.read_csv(args.waveform_summary)
        summary["snr_max"] = summary[["snr1", "snr2", "snr3"]].max(axis=1)
        for _, row in summary.iterrows():
            key = Path(row["waveform_file"]).name
            snr_lookup[key] = float(row["snr_max"])

    embeddings = []
    metadatas = []
    for sample in tqdm(dataset, desc="Encoding"):
        spectrogram_tensor, _, _, _, metadata = sample
        if "error" in metadata:
            continue
        if metadata["channel_type"] not in preprocessing["channels"]:
            continue

        x = spectrogram_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.create_embedding(x)[0].cpu().squeeze(0)
        metadata["snr"] = snr_lookup.get(Path(metadata["file_path"]).name, 1.0)
        embeddings.append(embedding)
        metadatas.append(metadata)

    if not embeddings:
        raise RuntimeError("No embeddings were created. Check data paths/channels.")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings_tensor = torch.stack(embeddings)
    torch.save(embeddings_tensor, out_dir / "embeddings.pt")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadatas, f, indent=4)

    source_payload = {
        "ae_checkpoint": str(ae_ckpt),
        "stft": {
            "nperseg": nperseg,
            "noverlap": noverlap,
            "nfft": nfft,
            "target_freq_bins": preprocessing["target_freq_bins"],
            "target_time_bins": preprocessing["target_time_bins"],
            "resample_hz": preprocessing["resample_hz"],
            "target_seconds": preprocessing["target_seconds"],
        },
        "normalization_mode": normalization["mode"],
        "global_min": normalization["global_min"],
        "global_max": normalization["global_max"],
        "channels": preprocessing["channels"],
        "num_embeddings": len(embeddings),
        "embedding_shape": list(embeddings_tensor.shape[1:]),
    }
    with open(out_dir / "source.json", "w") as f:
        json.dump(source_payload, f, indent=4)

    print(f"Saved embeddings: {out_dir / 'embeddings.pt'}")
    print(f"Saved metadata:   {out_dir / 'metadata.json'}")
    print(f"Saved source:     {out_dir / 'source.json'}")
    print(f"Embeddings shape: {tuple(embeddings_tensor.shape)}")


if __name__ == "__main__":
    main()
