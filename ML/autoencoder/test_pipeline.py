#!/usr/bin/env python
"""
Quick test script to verify the training pipeline works.
Trains on a small subset of data for 2 epochs.
"""
import sys
import torch
from torch.utils.data import DataLoader, Subset
from model import ConvAutoencoder
from stft_dataset import SeismicSTFTDataset, collate_fn


def test_pipeline():
    """Test the complete training pipeline."""
    print("=" * 60)
    print("Testing Seismic Autoencoder Training Pipeline")
    print("=" * 60)
    
    # Check CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device: {device}")
    
    # Create dataset with small subset
    print("\n2. Creating dataset...")
    try:
        dataset = SeismicSTFTDataset(
            data_dir="../../data/filtered_waveforms",
            channels=["HH"],
            nperseg=256,
            noverlap=192,
            nfft=256,
            normalize=True,
            log_scale=True,
        )
        print(f"   Total dataset size: {len(dataset)}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Use only first 100 samples for testing
    subset = Subset(dataset, range(min(100, len(dataset))))
    print(f"   Using subset size: {len(subset)} samples")
    
    # Create data loader
    print("\n3. Creating data loader...")
    loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    print(f"   Number of batches: {len(loader)}")
    
    # Load one batch
    print("\n4. Loading a sample batch...")
    try:
        batch_specs, batch_metadata = next(iter(loader))
        print(f"   Batch shape: {batch_specs.shape}")
        print(f"   Batch range: [{batch_specs.min():.4f}, {batch_specs.max():.4f}]")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Create model
    print("\n5. Creating model...")
    try:
        model = ConvAutoencoder(in_channels=3, latent_dim=128)
        model = model.to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {num_params:,}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test forward pass
    print("\n6. Testing forward pass...")
    try:
        batch_specs = batch_specs.to(device)
        with torch.no_grad():
            output = model(batch_specs)
        print(f"   Input shape:  {batch_specs.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test training step
    print("\n7. Testing training step...")
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        optimizer.zero_grad()
        output = model(batch_specs)
        loss = criterion(output, batch_specs)
        loss.backward()
        optimizer.step()
        
        print(f"   Loss: {loss.item():.6f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test multiple batches
    print("\n8. Testing multiple batches...")
    try:
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i, (specs, meta) in enumerate(loader):
            if specs is None:
                continue
            
            specs = specs.to(device)
            optimizer.zero_grad()
            output = model(specs)
            loss = criterion(output, specs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if i >= 4:  # Test only first 5 batches
                break
        
        avg_loss = total_loss / num_batches
        print(f"   Processed {num_batches} batches")
        print(f"   Average loss: {avg_loss:.6f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Training pipeline is ready.")
    print("=" * 60)
    print("\nYou can now run full training with:")
    print("  python train.py")
    print("or")
    print("  python train.py --channels HH --batch_size 16 --num_epochs 50")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
