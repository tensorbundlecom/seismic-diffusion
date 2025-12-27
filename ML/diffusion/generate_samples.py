"""
Script to generate seismic waveform samples using a trained diffusion model and VAE decoder.

This script:
1. Loads a trained diffusion model
2. Generates latent embeddings using the diffusion model with conditional inputs
3. Decodes the latent embeddings using a VAE decoder to produce waveform spectrograms
4. Saves the generated samples as images
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent / "autoencoder"))
from model import VariationalAutoencoder

# Import diffusion model components
from diffusion_model import ConditionalMLP, GaussianDiffusion


def load_diffusion_checkpoint(checkpoint_path, config, device='cuda'):
    """
    Load diffusion model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration dictionary
        device: Device to load the model on
        
    Returns:
        model: Loaded diffusion model
        diffusion: Diffusion process object
    """
    print(f"Loading diffusion checkpoint from {checkpoint_path}")
    
    # Create model
    model = ConditionalMLP(
        embedding_dim=config['embedding_dim'],
        time_dim=config['time_dim'],
        hidden_dims=config['hidden_dims'],
        num_stations=config['num_stations'],
        station_embed_dim=config['station_embed_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create diffusion process
    diffusion = GaussianDiffusion(
        timesteps=config['timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'],
        device=device
    )
    
    print(f"Loaded diffusion model from epoch {checkpoint['epoch']}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'N/A')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model, diffusion


def load_vae_checkpoint(checkpoint_path, device='cuda'):
    """
    Load VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        model: Loaded VAE model
        config: Configuration dictionary from checkpoint
    """
    print(f"Loading VAE checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    
    # Detect model type from state_dict keys
    has_vae_keys = any('fc_mu' in k or 'fc_logvar' in k for k in state_dict.keys())
    model_type = config.get('model_type', 'vae' if has_vae_keys else 'autoencoder')
    
    print(f"Detected model type: {model_type}")
    
    if model_type == 'vae':
        model = VariationalAutoencoder(
            in_channels=config.get('in_channels', 3),
            latent_dim=config.get('latent_dim', 256)
        ).to(device)
        
        # Handle dynamic FC layer initialization in VAE encoder
        # If state_dict has fc_mu/fc_logvar, we need to initialize them first
        if 'encoder.fc_mu.weight' in state_dict:
            flattened_dim = state_dict['encoder.fc_mu.weight'].shape[1]
            model.encoder._initialize_fc_layers(flattened_dim)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded VAE model:")
    print(f"  In channels: {config.get('in_channels', 3)}")
    print(f"  Latent dim: {config.get('latent_dim', 256)}")
    
    return model, config


def load_station_mapping(metadata_path):
    """
    Load station name mapping from metadata.
    
    Args:
        metadata_path: Path to metadata JSON file
        
    Returns:
        Dictionary mapping station indices to station names
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Create mapping from station_idx to station_name
    station_map = {}
    for sample in metadata:
        station_map[sample['station_idx']] = sample['station_name']
    
    return station_map


def generate_samples(
    diffusion_model,
    diffusion_process,
    vae_decoder,
    num_samples,
    magnitudes,
    locations,
    station_indices,
    embedding_dim,
    device='cuda'
):
    """
    Generate samples using diffusion model and VAE decoder.
    
    Args:
        diffusion_model: Trained diffusion model
        diffusion_process: GaussianDiffusion object
        vae_decoder: VAE decoder
        num_samples: Number of samples to generate
        magnitudes: Earthquake magnitudes (B,) or (B, 1)
        locations: Earthquake locations (B, 3) - [lat, lon, depth]
        station_indices: Station indices (B,) or (B, 1)
        embedding_dim: Dimension of embeddings
        device: Device to generate on
        
    Returns:
        embeddings: Generated embeddings from diffusion model
        waveforms: Generated waveform spectrograms from VAE decoder
    """
    print(f"\nGenerating {num_samples} samples...")
    
    # Ensure correct shapes and types
    if len(magnitudes.shape) == 1:
        magnitudes = magnitudes.unsqueeze(-1)  # (B, 1)
    if len(station_indices.shape) == 1:
        station_indices = station_indices.unsqueeze(-1)  # (B, 1)
    
    magnitudes = magnitudes.to(device)
    locations = locations.to(device)
    station_indices = station_indices.to(device)
    
    # Generate embeddings using diffusion model
    with torch.no_grad():
        embeddings = diffusion_process.sample(
            model=diffusion_model,
            shape=(num_samples, embedding_dim),
            magnitude=magnitudes,
            location=locations,
            station_idx=station_indices
        )
    
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Decode embeddings to waveforms using VAE decoder
    with torch.no_grad():
        waveforms = vae_decoder.decode(embeddings)
    
    print(f"Generated waveforms shape: {waveforms.shape}")
    
    return embeddings, waveforms


def save_samples(waveforms, output_dir, metadata_list, station_map):
    """
    Save generated waveform samples as images.
    
    Args:
        waveforms: Generated waveforms (B, C, H, W)
        output_dir: Directory to save images
        metadata_list: List of metadata dictionaries for each sample
        station_map: Dictionary mapping station indices to names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving samples to {output_dir}...")
    
    for i in tqdm(range(waveforms.shape[0]), desc="Saving"):
        waveform = waveforms[i].cpu().numpy()  # (C, H, W)
        metadata = metadata_list[i]
        
        # Create figure with 3 channels
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        channel_names = ['Z', 'N', 'E']  # Vertical, North, East components
        
        for j, (ax, channel_name) in enumerate(zip(axes, channel_names)):
            im = ax.imshow(waveform[j], aspect='auto', cmap='seismic', 
                          interpolation='nearest', origin='lower')
            ax.set_title(f'Channel {channel_name}')
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Time')
            plt.colorbar(im, ax=ax)
        
        # Add metadata as title
        station_name = station_map.get(metadata['station_idx'], 'Unknown')
        title = (f"Generated Waveform - Magnitude: {metadata['magnitude']:.1f}, "
                f"Station: {station_name}, "
                f"Location: ({metadata['lat']:.2f}, {metadata['lon']:.2f}, {metadata['depth']:.1f} km)")
        fig.suptitle(title, fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"sample_{i:04d}_mag{metadata['magnitude']:.1f}_{station_name}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Saved {waveforms.shape[0]} samples to {output_dir}")


def create_conditioning_from_params(magnitudes, latitudes, longitudes, depths, 
                                    station_indices, location_stats):
    """
    Create conditioning tensors from parameters.
    
    Args:
        magnitudes: List or array of magnitudes
        latitudes: List or array of latitudes
        longitudes: List or array of longitudes
        depths: List or array of depths (km)
        station_indices: List or array of station indices
        location_stats: Dictionary with location normalization stats
        
    Returns:
        magnitude_tensor: (B, 1)
        location_tensor: (B, 3)
        station_tensor: (B, 1)
    """
    num_samples = len(magnitudes)
    
    # Create tensors
    magnitude_tensor = torch.FloatTensor(magnitudes).view(num_samples, 1)
    
    # Normalize locations using training stats
    lat_normalized = (np.array(latitudes) - location_stats['lat_mean']) / location_stats['lat_std']
    lon_normalized = (np.array(longitudes) - location_stats['lon_mean']) / location_stats['lon_std']
    depth_normalized = (np.array(depths) - location_stats['depth_mean']) / location_stats['depth_std']
    
    location_tensor = torch.FloatTensor(np.stack([lat_normalized, lon_normalized, depth_normalized], axis=1))
    station_tensor = torch.LongTensor(station_indices).view(num_samples, 1)
    
    return magnitude_tensor, location_tensor, station_tensor


def main():
    parser = argparse.ArgumentParser(description='Generate seismic waveforms using diffusion model and VAE')
    
    # Model checkpoints
    parser.add_argument('--diffusion_checkpoint', type=str, required=True,
                        help='Path to diffusion model checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, required=True,
                        help='Path to VAE model checkpoint')
    parser.add_argument('--diffusion_config', type=str, default=None,
                        help='Path to diffusion config.json (if not in checkpoint dir)')
    
    # Generation parameters
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='generated_samples',
                        help='Directory to save generated samples')
    
    # Conditioning parameters (can specify manually or use random from metadata)
    parser.add_argument('--magnitude', type=float, nargs='+', default=None,
                        help='Earthquake magnitude(s) for generation')
    parser.add_argument('--latitude', type=float, nargs='+', default=None,
                        help='Earthquake latitude(s) for generation')
    parser.add_argument('--longitude', type=float, nargs='+', default=None,
                        help='Earthquake longitude(s) for generation')
    parser.add_argument('--depth', type=float, nargs='+', default=None,
                        help='Earthquake depth(s) in km for generation')
    parser.add_argument('--station_idx', type=int, nargs='+', default=None,
                        help='Station index/indices for generation')
    
    # Use random samples from training data
    parser.add_argument('--use_random_conditioning', action='store_true',
                        help='Use random conditioning from training metadata')
    parser.add_argument('--metadata_path', type=str, 
                        default='embeddings/train_metadata.json',
                        help='Path to metadata JSON for conditioning')
    parser.add_argument('--conditioning_path', type=str,
                        default='embeddings/train_conditioning.npz',
                        help='Path to conditioning .npz file for normalization stats')
    
    # Other
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load diffusion config
    if args.diffusion_config is None:
        # Try to load from checkpoint directory
        checkpoint_dir = os.path.dirname(args.diffusion_checkpoint)
        config_path = os.path.join(checkpoint_dir, 'config.json')
    else:
        config_path = args.diffusion_config
    
    print(f"Loading diffusion config from {config_path}")
    with open(config_path, 'r') as f:
        diffusion_config = json.load(f)
    
    # Load models
    diffusion_model, diffusion_process = load_diffusion_checkpoint(
        args.diffusion_checkpoint, diffusion_config, device
    )
    
    vae_model, vae_config = load_vae_checkpoint(args.vae_checkpoint, device)
    
    # Verify embedding dimensions match
    if diffusion_config['embedding_dim'] != vae_config['latent_dim']:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"diffusion={diffusion_config['embedding_dim']}, "
            f"vae={vae_config['latent_dim']}"
        )
    
    # Load station mapping
    station_map = load_station_mapping(args.metadata_path)
    print(f"Loaded {len(station_map)} station mappings")
    
    # Load normalization statistics from config file
    config_path = os.path.join(os.path.dirname(args.conditioning_path), 'train_config.json')
    if not os.path.exists(config_path):
        # Fallback: compute from the conditioning data itself
        print(f"\nWarning: {config_path} not found. Computing normalization stats from data.")
        conditioning_data = np.load(args.conditioning_path)
        locations = conditioning_data['locations']
        location_stats = {
            'lat_mean': float(locations[:, 0].mean()),
            'lat_std': float(locations[:, 0].std()),
            'lon_mean': float(locations[:, 1].mean()),
            'lon_std': float(locations[:, 1].std()),
            'depth_mean': float(locations[:, 2].mean()),
            'depth_std': float(locations[:, 2].std())
        }
    else:
        with open(config_path, 'r') as f:
            embed_config = json.load(f)
            location_mean = embed_config['location_stats']['mean']
            location_std = embed_config['location_stats']['std']
            location_stats = {
                'lat_mean': location_mean[0],
                'lat_std': location_std[0],
                'lon_mean': location_mean[1],
                'lon_std': location_std[1],
                'depth_mean': location_mean[2],
                'depth_std': location_std[2]
            }
    
    print("\nLocation normalization statistics:")
    print(f"  Latitude: {location_stats['lat_mean']:.2f} ± {location_stats['lat_std']:.2f}")
    print(f"  Longitude: {location_stats['lon_mean']:.2f} ± {location_stats['lon_std']:.2f}")
    print(f"  Depth: {location_stats['depth_mean']:.2f} ± {location_stats['depth_std']:.2f}")
    
    # Create conditioning
    metadata_list = []
    
    if args.use_random_conditioning:
        print("\nUsing random conditioning from training metadata")
        
        # Load metadata
        with open(args.metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Sample random indices
        indices = np.random.choice(len(metadata), args.num_samples, replace=True)
        
        magnitudes = []
        latitudes = []
        longitudes = []
        depths = []
        station_indices = []
        
        for idx in indices:
            sample = metadata[idx]
            magnitudes.append(sample['magnitude'])
            latitudes.append(sample['latitude'])
            longitudes.append(sample['longitude'])
            depths.append(sample['depth'])
            station_indices.append(sample['station_idx'])
            
            metadata_list.append({
                'magnitude': sample['magnitude'],
                'lat': sample['latitude'],
                'lon': sample['longitude'],
                'depth': sample['depth'],
                'station_idx': sample['station_idx']
            })
        
        magnitude_tensor, location_tensor, station_tensor = create_conditioning_from_params(
            magnitudes, latitudes, longitudes, depths, station_indices, location_stats
        )
        
    else:
        # Use manually specified conditioning
        print("\nUsing manually specified conditioning")
        
        # Check that all conditioning parameters are provided
        required_params = [args.magnitude, args.latitude, args.longitude, 
                          args.depth, args.station_idx]
        if any(p is None for p in required_params):
            raise ValueError(
                "All conditioning parameters (magnitude, latitude, longitude, depth, station_idx) "
                "must be provided when not using random conditioning"
            )
        
        # Expand parameters if single values provided
        if len(args.magnitude) == 1 and args.num_samples > 1:
            magnitudes = args.magnitude * args.num_samples
        else:
            magnitudes = args.magnitude
            
        if len(args.latitude) == 1 and args.num_samples > 1:
            latitudes = args.latitude * args.num_samples
        else:
            latitudes = args.latitude
            
        if len(args.longitude) == 1 and args.num_samples > 1:
            longitudes = args.longitude * args.num_samples
        else:
            longitudes = args.longitude
            
        if len(args.depth) == 1 and args.num_samples > 1:
            depths = args.depth * args.num_samples
        else:
            depths = args.depth
            
        if len(args.station_idx) == 1 and args.num_samples > 1:
            station_indices = args.station_idx * args.num_samples
        else:
            station_indices = args.station_idx
        
        # Verify lengths
        if not all(len(p) == args.num_samples for p in [magnitudes, latitudes, longitudes, depths, station_indices]):
            raise ValueError(
                f"All conditioning parameters must have length {args.num_samples} "
                f"or length 1 (to be repeated)"
            )
        
        magnitude_tensor, location_tensor, station_tensor = create_conditioning_from_params(
            magnitudes, latitudes, longitudes, depths, station_indices, location_stats
        )
        
        for i in range(args.num_samples):
            metadata_list.append({
                'magnitude': magnitudes[i],
                'lat': latitudes[i],
                'lon': longitudes[i],
                'depth': depths[i],
                'station_idx': station_indices[i]
            })
    
    # Print conditioning info
    print("\nConditioning information:")
    for i, meta in enumerate(metadata_list[:5]):  # Print first 5
        station_name = station_map.get(meta['station_idx'], 'Unknown')
        print(f"  Sample {i}: Mag={meta['magnitude']:.1f}, "
              f"Loc=({meta['lat']:.2f}, {meta['lon']:.2f}, {meta['depth']:.1f}km), "
              f"Station={station_name}")
    if len(metadata_list) > 5:
        print(f"  ... and {len(metadata_list) - 5} more samples")
    
    # Generate samples
    embeddings, waveforms = generate_samples(
        diffusion_model=diffusion_model,
        diffusion_process=diffusion_process,
        vae_decoder=vae_model,
        num_samples=args.num_samples,
        magnitudes=magnitude_tensor,
        locations=location_tensor,
        station_indices=station_tensor,
        embedding_dim=diffusion_config['embedding_dim'],
        device=device
    )
    
    # Save samples
    save_samples(waveforms, args.output_dir, metadata_list, station_map)
    
    # Save embeddings and conditioning info
    np.save(os.path.join(args.output_dir, 'generated_embeddings.npy'), 
            embeddings.cpu().numpy())
    
    with open(os.path.join(args.output_dir, 'generation_metadata.json'), 'w') as f:
        json.dump({
            'num_samples': args.num_samples,
            'diffusion_checkpoint': args.diffusion_checkpoint,
            'vae_checkpoint': args.vae_checkpoint,
            'seed': args.seed,
            'samples': metadata_list
        }, f, indent=2)
    
    print("\n✓ Generation complete!")
    print(f"  Saved {args.num_samples} samples to {args.output_dir}")


if __name__ == '__main__':
    main()
