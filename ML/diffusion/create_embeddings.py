#!/usr/bin/env python
"""
Inference script to test a trained autoencoder model.
Loads a checkpoint and visualizes reconstructions.
"""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ML.autoencoder.stft_dataset_with_metadata import SeismicSTFTDatasetWithMetadata
from ML.autoencoder.model import VariationalAutoencoder
from ML.autoencoder.inference import load_model

import json
from tqdm import tqdm

model, config = load_model("../autoencoder/checkpoints/20260227_010846/best_model.pt", device='cuda')

channels = ["HH"]
dataset = SeismicSTFTDatasetWithMetadata(
    data_dir="../../data/filtered_waveforms",
    event_file="../../data/events/20140101_20251101_0.0_9.0_9_339.txt",
    channels=channels,
    nperseg=config.get('nperseg', 256),
    noverlap=config.get('noverlap', 192),
    nfft=config.get('nfft', 256),
    normalize=True,
    log_scale=True,
)

sample_data = dataset[0][0]
print(dataset[0])
input_tensor = sample_data.unsqueeze(0).to('cuda')
with torch.no_grad():
    reconstructed = model(input_tensor)[0].cpu().squeeze(0)
    embedding = model.create_embedding(input_tensor)[0].cpu().squeeze(0)


embeddings = []
metadatas = []
for d in tqdm(dataset):
    spectrogram_tensor, magnitude_tensor, location_tensor, station_idx_tensor, metadata = d
    if metadata['channel_type'] in channels:
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0).to('cuda')
        with torch.no_grad():
            embedding = model.create_embedding(spectrogram_tensor)[0].cpu().squeeze(0)

        embeddings.append(embedding)
        metadatas.append(metadata)


# Save embeddings and metadata
embeddings = torch.stack(embeddings)
torch.save(embeddings, "embeddings/embeddings.pt")
with open("embeddings/metadata.json", "w") as f:
    json.dump(metadatas, f, indent=4)