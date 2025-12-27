"""
Script to create embeddings from trained VAE model and save them along with conditional variables.
These embeddings will be used as the latent space for diffusion models.
"""

import os
import sys
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent / "autoencoder"))
from model import VariationalAutoencoder
from stft_dataset_with_metadata import SeismicSTFTDatasetWithMetadata, collate_fn_with_metadata


def load_vae_checkpoint(checkpoint_path, device='cuda'):
    """
    Load VAE model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        
    Returns:
        model: Loaded model (VAE or ConvAutoencoder)
        config: Configuration dictionary from checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    config = checkpoint.get('config', {})
    state_dict = checkpoint['model_state_dict']
    
    # Detect model type from state_dict keys
    has_vae_keys = any('fc_mu' in k or 'fc_logvar' in k for k in state_dict.keys())
    model_type = config.get('model_type', 'vae' if has_vae_keys else 'autoencoder')
    
    print(f"Detected model type: {model_type}")
    
    if model_type == 'vae':
        # Check if this is an old checkpoint (before fc_projection was added)
        has_fc_projection = any('fc_projection' in k for k in state_dict.keys())
        
        if not has_fc_projection:
            print("WARNING: Loading old VAE checkpoint (before model updates)")
            print("  This checkpoint doesn't have the fc_projection layer")
            print("  The encoder will work, but decoder architecture differs")
            print("  Embeddings will still be created correctly using the encoder!")
        
        # Create VAE model
        from model import VariationalAutoencoder
        model = VariationalAutoencoder(
            in_channels=config.get('in_channels', 3),
            latent_dim=config.get('latent_dim', 128)
        )
        
        # Load state dict with strict=False to handle architecture differences
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"  Missing keys (will use initialized values): {len(missing_keys)}")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"    - {key}")
        
        if unexpected_keys:
            print(f"  Unexpected keys (ignored): {len(unexpected_keys)}")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"    - {key}")
    else:
        # Regular ConvAutoencoder - use encoder only for embeddings
        print("WARNING: This is a regular ConvAutoencoder, not a VAE")
        print("  Embeddings will be spatial (256, H/16, W/16) instead of 1D vectors")
        print("  This may not be ideal for latent diffusion - consider training a VAE instead")
        
        from model import ConvAutoencoder
        model = ConvAutoencoder(
            in_channels=config.get('in_channels', 3),
            latent_dim=config.get('latent_dim', 128)
        )
        
        # Load state dict
        model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"  Model type: {model_type}")
    print(f"  In channels: {config.get('in_channels', 3)}")
    print(f"  Latent dim: {config.get('latent_dim', 128)}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model, config


def create_embeddings(
    model,
    dataloader,
    device='cuda',
    use_mean=True,
    max_samples=None,
    is_vae=True
):
    """
    Create embeddings for all data in the dataloader.
    
    Args:
        model: VAE or ConvAutoencoder model
        dataloader: DataLoader with waveform data
        device: Device to run on
        use_mean: If True, use mu (deterministic). If False, sample z (VAE only).
        max_samples: Maximum number of samples to process (None = all)
        is_vae: Whether the model is a VAE (True) or ConvAutoencoder (False)
        
    Returns:
        embeddings: Numpy array of embeddings (N, latent_dim) or (N, C, H, W)
        magnitudes: Numpy array of magnitudes (N,)
        locations: Numpy array of locations (N, 3) - [lat, lon, depth]
        station_indices: Numpy array of station indices (N,)
        station_names: List of station names
        metadata_list: List of metadata dictionaries
    """
    model.eval()
    
    embeddings_list = []
    magnitudes_list = []
    locations_list = []
    station_indices_list = []
    metadata_list = []
    
    total_samples = 0
    
    print(f"Creating embeddings (use_mean={use_mean}, is_vae={is_vae})...")
    
    with torch.no_grad():
        for batch_idx, (spectrograms, magnitudes, locations, station_indices, metadata) in enumerate(tqdm(dataloader)):
            if spectrograms is None:
                continue
            
            # Move to device
            spectrograms = spectrograms.to(device)
            
            # Create embeddings based on model type
            if is_vae:
                # VAE returns 1D latent vectors
                batch_embeddings = model.create_embedding(spectrograms, use_mean=use_mean)
            else:
                # ConvAutoencoder returns spatial features
                batch_embeddings = model.create_embedding(spectrograms)
                # Flatten spatial dimensions for consistency
                batch_size = batch_embeddings.size(0)
                batch_embeddings = batch_embeddings.view(batch_size, -1)
            
            # Convert to numpy and collect
            embeddings_list.append(batch_embeddings.cpu().numpy())
            magnitudes_list.append(magnitudes.numpy())
            locations_list.append(locations.numpy())
            station_indices_list.append(station_indices.numpy())
            metadata_list.extend(metadata)
            
            total_samples += spectrograms.size(0)
            
            # Check if we've reached max_samples
            if max_samples is not None and total_samples >= max_samples:
                break
    
    # Concatenate all batches
    embeddings = np.concatenate(embeddings_list, axis=0)
    magnitudes = np.concatenate(magnitudes_list, axis=0)
    locations = np.concatenate(locations_list, axis=0)
    station_indices = np.concatenate(station_indices_list, axis=0)
    
    # Truncate if we exceeded max_samples
    if max_samples is not None and len(embeddings) > max_samples:
        embeddings = embeddings[:max_samples]
        magnitudes = magnitudes[:max_samples]
        locations = locations[:max_samples]
        station_indices = station_indices[:max_samples]
        metadata_list = metadata_list[:max_samples]
    
    print(f"\nCreated embeddings for {len(embeddings)} samples")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Magnitude range: [{magnitudes.min():.2f}, {magnitudes.max():.2f}]")
    print(f"  Location shape: {locations.shape}")
    print(f"  Unique stations: {len(np.unique(station_indices))}")
    
    return embeddings, magnitudes, locations, station_indices, metadata_list


def save_embeddings(
    output_dir,
    embeddings,
    magnitudes,
    locations,
    station_indices,
    station_names,
    metadata_list,
    config,
    split_name='train'
):
    """
    Save embeddings and conditional variables to disk.
    
    Args:
        output_dir: Directory to save the embeddings
        embeddings: Numpy array of embeddings
        magnitudes: Numpy array of magnitudes
        locations: Numpy array of locations
        station_indices: Numpy array of station indices
        station_names: List of station names
        metadata_list: List of metadata dictionaries
        config: Configuration dictionary
        split_name: Name of the split (train, val, test)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving embeddings to {output_dir}")
    
    # Save embeddings
    embeddings_path = output_dir / f"{split_name}_embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"  Saved embeddings: {embeddings_path}")
    
    # Save conditioning variables
    conditioning_path = output_dir / f"{split_name}_conditioning.npz"
    np.savez(
        conditioning_path,
        magnitudes=magnitudes,
        locations=locations,
        station_indices=station_indices
    )
    print(f"  Saved conditioning variables: {conditioning_path}")
    
    # Save station names mapping
    station_mapping_path = output_dir / "station_mapping.json"
    station_mapping = {
        'station_names': station_names,
        'num_stations': len(station_names)
    }
    with open(station_mapping_path, 'w') as f:
        json.dump(station_mapping, f, indent=2)
    print(f"  Saved station mapping: {station_mapping_path}")
    
    # Save metadata
    metadata_path = output_dir / f"{split_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    # Save configuration
    config_path = output_dir / f"{split_name}_config.json"
    embedding_config = {
        'vae_config': config,
        'num_samples': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'magnitude_stats': {
            'min': float(magnitudes.min()),
            'max': float(magnitudes.max()),
            'mean': float(magnitudes.mean()),
            'std': float(magnitudes.std())
        },
        'location_stats': {
            'min': locations.min(axis=0).tolist(),
            'max': locations.max(axis=0).tolist(),
            'mean': locations.mean(axis=0).tolist(),
            'std': locations.std(axis=0).tolist()
        },
        'num_stations': len(station_names)
    }
    with open(config_path, 'w') as f:
        json.dump(embedding_config, f, indent=2)
    print(f"  Saved config: {config_path}")
    
    # Print summary
    print(f"\nSummary for {split_name} split:")
    print(f"  Total samples: {len(embeddings)}")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    print(f"  Magnitude: {magnitudes.mean():.2f} ± {magnitudes.std():.2f}")
    print(f"  Locations: {locations.shape}")
    print(f"  Stations: {len(station_names)} unique")


def main():
    parser = argparse.ArgumentParser(description='Create embeddings from VAE model')
    
    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to VAE checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                        default='../../data/filtered_waveforms',
                        help='Path to filtered waveforms directory')
    parser.add_argument('--event_file', type=str,
                        default='../../data/events/20140101_20251101_0.0_9.0_9_339.txt',
                        help='Path to event catalog file')
    parser.add_argument('--channels', nargs='+', default=['HH'],
                        help='Channel types to include')
    parser.add_argument('--magnitude_col', type=str, default='xM',
                        help='Magnitude column to use')
    
    # STFT parameters
    parser.add_argument('--nperseg', type=int, default=256,
                        help='STFT segment length')
    parser.add_argument('--noverlap', type=int, default=192,
                        help='STFT overlap')
    parser.add_argument('--nfft', type=int, default=256,
                        help='STFT FFT length')
    
    # Dataset split arguments
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Fraction of data for training')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Embedding arguments
    parser.add_argument('--use_mean', action='store_true', default=True,
                        help='Use mean of latent distribution (deterministic)')
    parser.add_argument('--use_sample', dest='use_mean', action='store_false',
                        help='Sample from latent distribution (stochastic)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (None = all)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='embeddings',
                        help='Directory to save embeddings')
    parser.add_argument('--create_splits', action='store_true', default=True,
                        help='Create separate train/val/test splits')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load model
    model, vae_config = load_vae_checkpoint(args.checkpoint, device=args.device)
    
    # Detect if model is VAE or ConvAutoencoder
    is_vae = vae_config.get('model_type', 'vae') == 'vae'
    
    # Create dataset
    print(f"\nLoading dataset...")
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=args.channels,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        normalize=True,
        log_scale=True,
        return_magnitude=True,
        magnitude_col=args.magnitude_col
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if args.create_splits:
        # Split dataset
        from torch.utils.data import random_split
        
        train_size = int(args.train_split * len(dataset))
        val_size = int(args.val_split * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        print(f"\nDataset splits:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val: {len(val_dataset)}")
        print(f"  Test: {len(test_dataset)}")
        
        # Process each split
        for split_name, split_dataset in [
            ('train', train_dataset),
            ('val', val_dataset),
            ('test', test_dataset)
        ]:
            print(f"\n{'='*60}")
            print(f"Processing {split_name} split")
            print('='*60)
            
            dataloader = DataLoader(
                split_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=collate_fn_with_metadata,
                pin_memory=True if args.device == 'cuda' else False
            )
            
            # Create embeddings
            embeddings, magnitudes, locations, station_indices, metadata_list = create_embeddings(
                model=model,
                dataloader=dataloader,
                device=args.device,
                use_mean=args.use_mean,
                max_samples=args.max_samples,
                is_vae=is_vae
            )
            
            # Save embeddings
            save_embeddings(
                output_dir=args.output_dir,
                embeddings=embeddings,
                magnitudes=magnitudes,
                locations=locations,
                station_indices=station_indices,
                station_names=dataset.station_names,
                metadata_list=metadata_list,
                config=vae_config,
                split_name=split_name
            )
    
    else:
        # Process entire dataset
        print(f"\n{'='*60}")
        print(f"Processing entire dataset")
        print('='*60)
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_with_metadata,
            pin_memory=True if args.device == 'cuda' else False
        )
        
        # Create embeddings
        embeddings, magnitudes, locations, station_indices, metadata_list = create_embeddings(
            model=model,
            dataloader=dataloader,
            device=args.device,
            use_mean=args.use_mean,
            max_samples=args.max_samples,
            is_vae=is_vae
        )
        
        # Save embeddings
        save_embeddings(
            output_dir=args.output_dir,
            embeddings=embeddings,
            magnitudes=magnitudes,
            locations=locations,
            station_indices=station_indices,
            station_names=dataset.station_names,
            metadata_list=metadata_list,
            config=vae_config,
            split_name='all'
        )
    
    print(f"\n{'='*60}")
    print("Embedding creation complete!")
    print(f"Output directory: {args.output_dir}")
    print('='*60)


if __name__ == "__main__":
    main()
