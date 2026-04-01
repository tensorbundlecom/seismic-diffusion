#!/usr/bin/env python
"""
Create diffusion-training embeddings from a trained autoencoder checkpoint.
"""
import argparse
import json
import re
from pathlib import Path

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
        default=["HH"],
        help="Channel groups to include (e.g. HH HN)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output),
        help="Directory where embeddings.pt / metadata.json / source.json are saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for autoencoder inference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ae_ckpt = Path(args.ae_checkpoint).expanduser().resolve() if args.ae_checkpoint else _find_latest_timestamped_ae_checkpoint()
    if not ae_ckpt.exists():
        raise FileNotFoundError(f"AE checkpoint does not exist: {ae_ckpt}")

    print(f"Loading AE checkpoint: {ae_ckpt}")
    model, config = load_model(str(ae_ckpt), device=str(device))
    model.eval()

    nperseg = int(config.get("nperseg", 256))
    noverlap = int(config.get("noverlap", 192))
    nfft = int(config.get("nfft", 256))
    print(
        "Using STFT params from AE config: "
        f"nperseg={nperseg}, noverlap={noverlap}, nfft={nfft}"
    )

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=args.channels,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        normalize=True,
        log_scale=True,
    )

    embeddings = []
    metadatas = []
    for sample in tqdm(dataset, desc="Encoding"):
        spectrogram_tensor, _, _, _, metadata = sample
        if "error" in metadata:
            continue
        if metadata["channel_type"] not in args.channels:
            continue

        x = spectrogram_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.create_embedding(x)[0].cpu().squeeze(0)
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
        },
        "channels": args.channels,
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
