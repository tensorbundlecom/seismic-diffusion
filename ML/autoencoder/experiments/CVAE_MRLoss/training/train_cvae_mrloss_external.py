import argparse
import csv
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

from ML.autoencoder.experiments.CVAE_MRLoss.core.mr_loss_utils import (
    multi_resolution_stft_loss,
    reconstruct_waveform_with_gt_phase,
)
from ML.autoencoder.experiments.CVAE_MRLoss.core.waveform_dataset import (
    SeismicSTFTDatasetWithWaveform,
    collate_fn_with_waveform,
)
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder


def build_parser():
    p = argparse.ArgumentParser(description='Train CVAE with Multi-Resolution STFT loss')
    p.add_argument('--run_name', type=str, default='default')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--lambda_img', type=float, default=0.5)
    p.add_argument('--lambda_mr', type=float, default=0.5)
    p.add_argument('--latent_dim', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--fft_sizes', type=str, default='64,256,1024')
    p.add_argument('--hop_sizes', type=str, default='16,64,256')
    p.add_argument('--win_lengths', type=str, default='64,256,1024')
    p.add_argument('--scale_weights', type=str, default='1.0,0.7,0.5')

    p.add_argument('--data_dir', type=str, default='data/external_dataset/extracted/data/filtered_waveforms')
    p.add_argument('--event_file', type=str, default='data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt')
    p.add_argument('--station_list_file', type=str, default='data/station_list_external_full.json')
    return p


def parse_csv_numbers(text, cast=float):
    return tuple(cast(x.strip()) for x in text.split(',') if x.strip())


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train():
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fft_sizes = parse_csv_numbers(args.fft_sizes, int)
    hop_sizes = parse_csv_numbers(args.hop_sizes, int)
    win_lengths = parse_csv_numbers(args.win_lengths, int)
    scale_weights = parse_csv_numbers(args.scale_weights, float)

    if not os.path.exists(args.data_dir):
        print(f'[ERROR] Data directory not found: {args.data_dir}')
        return

    with open(args.station_list_file, 'r') as f:
        station_list = json.load(f)

    dataset = SeismicSTFTDatasetWithWaveform(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=['HH'],
        magnitude_col='ML',
        station_list=station_list,
        target_fs=100.0,
        target_len=7300,
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_with_waveform,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_with_waveform,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    model = ConditionalVariationalAutoencoder(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_stations=len(station_list),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    exp_dir = Path('ML/autoencoder/experiments/CVAE_MRLoss')
    checkpoint_dir = exp_dir / 'checkpoints'
    result_dir = exp_dir / 'results'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_name
    history_csv = result_dir / f'train_history_{run_id}.csv'

    with open(result_dir / f'config_{run_id}.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print('[INFO] Starting CVAE_MRLoss training')
    print(f'[INFO] run_name={run_id} device={device}')
    print(f'[INFO] data_size={len(dataset)} train/val={train_size}/{val_size}')
    print(
        f'[INFO] epochs={args.epochs} bs={args.batch_size} lr={args.lr} '
        f'beta={args.beta} lambda_img={args.lambda_img} lambda_mr={args.lambda_mr}'
    )
    print(
        f'[INFO] MR params: fft={fft_sizes} hop={hop_sizes} win={win_lengths} '
        f'scale_w={scale_weights}'
    )

    fieldnames = [
        'epoch',
        'train_total', 'train_img', 'train_mr', 'train_kl',
        'val_total', 'val_img', 'val_mr', 'val_kl',
        'val_scale0_sc', 'val_scale0_mag',
        'val_scale1_sc', 'val_scale1_mag',
        'val_scale2_sc', 'val_scale2_mag',
        'epoch_sec',
    ]
    with open(history_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    best_val = float('inf')

    for epoch in range(args.epochs):
        t0 = time.time()

        model.train()
        tr_total = tr_img = tr_mr = tr_kl = 0.0
        tr_steps = 0

        for bidx, batch in enumerate(train_loader):
            specs, mags, locs, stations, waveforms, metas = batch
            if specs is None:
                continue

            specs = specs.to(device)
            mags = mags.to(device)
            locs = locs.to(device)
            stations = stations.to(device)
            waveforms = waveforms.to(device)

            mag_min = torch.tensor([float(m.get('mag_min', 0.0)) for m in metas], device=device)
            mag_max = torch.tensor([float(m.get('mag_max', 1.0)) for m in metas], device=device)

            optimizer.zero_grad(set_to_none=True)

            recon, mu, logvar = model(specs, mags, locs, stations)

            img_loss = torch.nn.functional.mse_loss(recon, specs, reduction='sum') / specs.size(0)
            kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / specs.size(0)

            pred_spec_z = recon[:, 2, :, :]
            pred_wave = reconstruct_waveform_with_gt_phase(
                pred_spec_z=pred_spec_z,
                gt_waveform=waveforms,
                mag_min=mag_min,
                mag_max=mag_max,
                n_fft=256,
                hop_length=64,
                win_length=256,
            )

            mr_loss, mr_detail = multi_resolution_stft_loss(
                pred_wave=pred_wave,
                gt_wave=waveforms,
                fft_sizes=fft_sizes,
                hop_sizes=hop_sizes,
                win_lengths=win_lengths,
                scale_weights=scale_weights,
                alpha_sc=1.0,
                alpha_mag=1.0,
            )

            total = args.lambda_img * img_loss + args.lambda_mr * mr_loss + args.beta * kl_loss
            total.backward()
            optimizer.step()

            tr_total += total.item()
            tr_img += img_loss.item()
            tr_mr += mr_loss.item()
            tr_kl += kl_loss.item()
            tr_steps += 1

            if (bidx + 1) % 100 == 0:
                print(
                    f'[TRAIN][E{epoch+1:03d}][B{bidx+1:05d}] '
                    f'total={total.item():.4f} img={img_loss.item():.4f} '
                    f'mr={mr_loss.item():.4f} kl={kl_loss.item():.4f} | '
                    f's0_sc={mr_detail.get("scale0_sc", 0):.4f} s0_mag={mr_detail.get("scale0_mag", 0):.4f} '
                    f's1_sc={mr_detail.get("scale1_sc", 0):.4f} s1_mag={mr_detail.get("scale1_mag", 0):.4f} '
                    f's2_sc={mr_detail.get("scale2_sc", 0):.4f} s2_mag={mr_detail.get("scale2_mag", 0):.4f}'
                )

        if tr_steps == 0:
            print(f'[WARN] Epoch {epoch+1} has no valid training batch')
            continue

        tr_total /= tr_steps
        tr_img /= tr_steps
        tr_mr /= tr_steps
        tr_kl /= tr_steps

        model.eval()
        va_total = va_img = va_mr = va_kl = 0.0
        va_steps = 0

        # Keep latest MR diagnostics from val for logging.
        val_mr_detail = {
            'scale0_sc': 0.0, 'scale0_mag': 0.0,
            'scale1_sc': 0.0, 'scale1_mag': 0.0,
            'scale2_sc': 0.0, 'scale2_mag': 0.0,
        }

        with torch.no_grad():
            for batch in val_loader:
                specs, mags, locs, stations, waveforms, metas = batch
                if specs is None:
                    continue

                specs = specs.to(device)
                mags = mags.to(device)
                locs = locs.to(device)
                stations = stations.to(device)
                waveforms = waveforms.to(device)

                mag_min = torch.tensor([float(m.get('mag_min', 0.0)) for m in metas], device=device)
                mag_max = torch.tensor([float(m.get('mag_max', 1.0)) for m in metas], device=device)

                recon, mu, logvar = model(specs, mags, locs, stations)

                img_loss = torch.nn.functional.mse_loss(recon, specs, reduction='sum') / specs.size(0)
                kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / specs.size(0)

                pred_spec_z = recon[:, 2, :, :]
                pred_wave = reconstruct_waveform_with_gt_phase(
                    pred_spec_z=pred_spec_z,
                    gt_waveform=waveforms,
                    mag_min=mag_min,
                    mag_max=mag_max,
                    n_fft=256,
                    hop_length=64,
                    win_length=256,
                )

                mr_loss, mr_detail = multi_resolution_stft_loss(
                    pred_wave=pred_wave,
                    gt_wave=waveforms,
                    fft_sizes=fft_sizes,
                    hop_sizes=hop_sizes,
                    win_lengths=win_lengths,
                    scale_weights=scale_weights,
                    alpha_sc=1.0,
                    alpha_mag=1.0,
                )

                total = args.lambda_img * img_loss + args.lambda_mr * mr_loss + args.beta * kl_loss

                va_total += total.item()
                va_img += img_loss.item()
                va_mr += mr_loss.item()
                va_kl += kl_loss.item()
                va_steps += 1

                # running mean for detailed val diagnostics
                for k in val_mr_detail:
                    val_mr_detail[k] += mr_detail.get(k, 0.0)

        if va_steps == 0:
            print(f'[WARN] Epoch {epoch+1} has no valid validation batch')
            continue

        va_total /= va_steps
        va_img /= va_steps
        va_mr /= va_steps
        va_kl /= va_steps
        for k in val_mr_detail:
            val_mr_detail[k] /= va_steps

        ep_sec = time.time() - t0

        print(
            f'[EPOCH {epoch+1:03d}/{args.epochs}] '
            f'train(total/img/mr/kl)=({tr_total:.4f}/{tr_img:.4f}/{tr_mr:.4f}/{tr_kl:.4f}) '
            f'val(total/img/mr/kl)=({va_total:.4f}/{va_img:.4f}/{va_mr:.4f}/{va_kl:.4f}) '
            f'val_s0(sc/mag)=({val_mr_detail["scale0_sc"]:.4f}/{val_mr_detail["scale0_mag"]:.4f}) '
            f'val_s1(sc/mag)=({val_mr_detail["scale1_sc"]:.4f}/{val_mr_detail["scale1_mag"]:.4f}) '
            f'val_s2(sc/mag)=({val_mr_detail["scale2_sc"]:.4f}/{val_mr_detail["scale2_mag"]:.4f}) '
            f'time={ep_sec:.1f}s'
        )

        row = {
            'epoch': epoch + 1,
            'train_total': tr_total,
            'train_img': tr_img,
            'train_mr': tr_mr,
            'train_kl': tr_kl,
            'val_total': va_total,
            'val_img': va_img,
            'val_mr': va_mr,
            'val_kl': va_kl,
            'val_scale0_sc': val_mr_detail['scale0_sc'],
            'val_scale0_mag': val_mr_detail['scale0_mag'],
            'val_scale1_sc': val_mr_detail['scale1_sc'],
            'val_scale1_mag': val_mr_detail['scale1_mag'],
            'val_scale2_sc': val_mr_detail['scale2_sc'],
            'val_scale2_mag': val_mr_detail['scale2_mag'],
            'epoch_sec': ep_sec,
        }

        with open(history_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        payload = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': va_total,
            'config': vars(args),
        }

        torch.save(payload, checkpoint_dir / f'cvae_mrloss_{run_id}_latest.pt')

        if (epoch + 1) % 10 == 0:
            torch.save(payload, checkpoint_dir / f'cvae_mrloss_{run_id}_epoch_{epoch+1:03d}.pt')

        if va_total < best_val:
            best_val = va_total
            torch.save(payload, checkpoint_dir / f'cvae_mrloss_{run_id}_best.pt')
            print(f'[INFO] New best checkpoint for {run_id}: val_total={best_val:.4f}')

    print('[INFO] Training completed.')


if __name__ == '__main__':
    train()
