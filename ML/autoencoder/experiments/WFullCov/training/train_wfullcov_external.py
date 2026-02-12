import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import (
    SeismicSTFTDatasetWithMetadata,
    collate_fn_with_metadata,
)
from ML.autoencoder.experiments.WFullCov.core.loss_utils import wfullcov_loss_function
from ML.autoencoder.experiments.WFullCov.core.model_wfullcov import WFullCovCVAE


def train_wfullcov_external():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 128
    w_dim = 64
    batch_size = 64
    epochs = 100
    lr = 5e-4
    beta = 0.1

    data_dir = 'data/external_dataset/extracted/data/filtered_waveforms'
    event_file = 'data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt'
    station_list_file = 'data/station_list_external_full.json'

    if not os.path.exists(data_dir):
        print(f'[ERROR] Data directory not found: {data_dir}')
        return

    with open(station_list_file, 'r') as f:
        station_list = json.load(f)
    print(f'[INFO] Loaded fixed station list: {len(station_list)} stations')

    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=data_dir,
        event_file=event_file,
        channels=['HH'],
        magnitude_col='ML',
        station_list=station_list,
    )

    num_stations = len(dataset.station_names)
    print(f'[INFO] Dataset size: {len(dataset)} | stations in dataset mapping: {num_stations}')

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f'[INFO] Train/Val split: {train_size}/{val_size}')

    num_workers = 16
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_metadata,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_metadata,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
    )

    model = WFullCovCVAE(
        in_channels=3,
        latent_dim=latent_dim,
        num_stations=num_stations,
        w_dim=w_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    exp_dir = Path('ML/autoencoder/experiments/WFullCov')
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print('[INFO] Starting WFullCov training')
    print(f'[INFO] Device={device} latent_dim={latent_dim} w_dim={w_dim}')
    print(f'[INFO] batch_size={batch_size} epochs={epochs} lr={lr} beta={beta}')

    best_val_loss = float('inf')

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_loader):
            specs, mags, locs, stations, _ = batch
            if specs is None:
                continue

            specs = specs.to(device)
            mags = mags.to(device)
            locs = locs.to(device)
            stations = stations.to(device)

            optimizer.zero_grad(set_to_none=True)
            recon, mu, L = model(specs, mags, locs, stations)
            loss, recon_l, kl_l = wfullcov_loss_function(recon, specs, mu, L, beta=beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_l.item()
            train_kl += kl_l.item()
            train_steps += 1

            if (batch_idx + 1) % 200 == 0:
                print(
                    f'[TRAIN][E{epoch+1:03d}][B{batch_idx+1:05d}] '
                    f'loss={loss.item():.2f} recon={recon_l.item():.2f} kl={kl_l.item():.2f}'
                )

        if train_steps == 0:
            print(f'[WARN] Epoch {epoch+1}: no valid training batches')
            continue

        avg_train_loss = train_loss / train_steps
        avg_train_recon = train_recon / train_steps
        avg_train_kl = train_kl / train_steps

        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                specs, mags, locs, stations, _ = batch
                if specs is None:
                    continue

                specs = specs.to(device)
                mags = mags.to(device)
                locs = locs.to(device)
                stations = stations.to(device)

                recon, mu, L = model(specs, mags, locs, stations)
                loss, recon_l, kl_l = wfullcov_loss_function(recon, specs, mu, L, beta=beta)

                val_loss += loss.item()
                val_recon += recon_l.item()
                val_kl += kl_l.item()
                val_steps += 1

        if val_steps == 0:
            print(f'[WARN] Epoch {epoch+1}: no valid validation batches')
            continue

        avg_val_loss = val_loss / val_steps
        avg_val_recon = val_recon / val_steps
        avg_val_kl = val_kl / val_steps
        elapsed = time.time() - epoch_start

        print(
            f'[EPOCH {epoch+1:03d}/{epochs}] '
            f'train_loss={avg_train_loss:.2f} train_recon={avg_train_recon:.2f} train_kl={avg_train_kl:.2f} | '
            f'val_loss={avg_val_loss:.2f} val_recon={avg_val_recon:.2f} val_kl={avg_val_kl:.2f} | '
            f'time={elapsed:.1f}s'
        )

        payload = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'config': {
                'latent_dim': latent_dim,
                'w_dim': w_dim,
                'num_stations': num_stations,
                'in_channels': 3,
                'beta': beta,
                'lr': lr,
            },
        }

        torch.save(payload, checkpoint_dir / 'wfullcov_external_latest.pt')

        if (epoch + 1) % 10 == 0:
            torch.save(payload, checkpoint_dir / f'wfullcov_external_epoch_{epoch+1:03d}.pt')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(payload, checkpoint_dir / 'wfullcov_external_best.pt')
            print(f'[INFO] New best model saved: val_loss={best_val_loss:.2f}')

    print('[INFO] WFullCov training completed.')


if __name__ == '__main__':
    train_wfullcov_external()
