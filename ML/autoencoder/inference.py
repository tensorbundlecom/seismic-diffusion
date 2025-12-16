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

from model import ConvAutoencoder
from stft_dataset import SeismicSTFTDataset


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    config = checkpoint.get('config', {})
    latent_dim = config.get('latent_dim', 128)
    
    # Create model
    model = ConvAutoencoder(in_channels=3, latent_dim=latent_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return model, config


def visualize_reconstruction(input_spec, reconstructed_spec, metadata, save_path=None):
    """Visualize input and reconstructed spectrograms."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Convert to numpy
    input_np = input_spec.cpu().numpy()
    reconstructed_np = reconstructed_spec.cpu().numpy()
    
    # Channel names
    channel_names = ['E (East)', 'N (North)', 'Z (Vertical)']
    
    # Plot input spectrograms
    for i in range(3):
        axes[0, i].imshow(
            input_np[i],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        axes[0, i].set_title(f'Input - {channel_names[i]}')
        axes[0, i].set_ylabel('Frequency Bin')
        axes[0, i].set_xlabel('Time Bin')
        plt.colorbar(axes[0, i].images[0], ax=axes[0, i])
    
    # Plot reconstructed spectrograms
    for i in range(3):
        axes[1, i].imshow(
            reconstructed_np[i],
            aspect='auto',
            origin='lower',
            cmap='viridis',
            interpolation='nearest'
        )
        axes[1, i].set_title(f'Reconstructed - {channel_names[i]}')
        axes[1, i].set_ylabel('Frequency Bin')
        axes[1, i].set_xlabel('Time Bin')
        plt.colorbar(axes[1, i].images[0], ax=axes[1, i])
    
    # Add metadata as suptitle
    file_name = metadata.get('file_name', 'Unknown')
    plt.suptitle(f'File: {file_name}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compute_reconstruction_metrics(input_spec, reconstructed_spec):
    """Compute reconstruction metrics."""
    mse = torch.nn.functional.mse_loss(reconstructed_spec, input_spec).item()
    mae = torch.nn.functional.l1_loss(reconstructed_spec, input_spec).item()
    
    # Compute per-channel metrics
    channel_mse = []
    for i in range(input_spec.shape[0]):
        ch_mse = torch.nn.functional.mse_loss(
            reconstructed_spec[i], input_spec[i]
        ).item()
        channel_mse.append(ch_mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'channel_mse': channel_mse,
    }


def main():
    parser = argparse.ArgumentParser(description="Test trained autoencoder")
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--data_dir', type=str, default='../../data/filtered_waveforms',
                        help='Path to data directory')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    
    # Create dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    channels = config.get('channels', ['HH'])
    nperseg = config.get('nperseg', 256)
    noverlap = config.get('noverlap', 192)
    nfft = config.get('nfft', 256)
    
    dataset = SeismicSTFTDataset(
        data_dir=args.data_dir,
        channels=channels,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        normalize=True,
        log_scale=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test on random samples
    print(f"\nTesting on {args.num_samples} random samples...")
    
    all_metrics = []
    
    for i in range(args.num_samples):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        spectrogram, metadata = dataset[idx]
        
        # Move to device and add batch dimension
        input_spec = spectrogram.unsqueeze(0).to(device)
        
        # Get reconstruction
        with torch.no_grad():
            reconstructed_spec = model(input_spec)
        
        # Remove batch dimension
        input_spec = input_spec.squeeze(0)
        reconstructed_spec = reconstructed_spec.squeeze(0)
        
        # Compute metrics
        metrics = compute_reconstruction_metrics(input_spec, reconstructed_spec)
        all_metrics.append(metrics)
        
        # Print metrics
        print(f"\nSample {i+1}: {metadata['file_name']}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Channel MSE: E={metrics['channel_mse'][0]:.6f}, "
              f"N={metrics['channel_mse'][1]:.6f}, Z={metrics['channel_mse'][2]:.6f}")
        
        # Visualize
        save_path = output_dir / f"reconstruction_{i+1}_{metadata['file_name']}.png"
        visualize_reconstruction(input_spec, reconstructed_spec, metadata, save_path)
    
    # Compute average metrics
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_mae = np.mean([m['mae'] for m in all_metrics])
    avg_channel_mse = np.mean([m['channel_mse'] for m in all_metrics], axis=0)
    
    print("\n" + "="*60)
    print("Average Metrics:")
    print(f"  MSE: {avg_mse:.6f}")
    print(f"  MAE: {avg_mae:.6f}")
    print(f"  Channel MSE: E={avg_channel_mse[0]:.6f}, "
          f"N={avg_channel_mse[1]:.6f}, Z={avg_channel_mse[2]:.6f}")
    print("="*60)
    
    print(f"\nVisualizations saved to {output_dir}/")


if __name__ == "__main__":
    main()
