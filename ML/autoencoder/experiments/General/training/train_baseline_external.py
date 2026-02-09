import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import os
import json

# Add project root (go up 5 levels: training -> General -> experiments -> autoencoder -> ML -> root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata, collate_fn_with_metadata
from ML.autoencoder.experiments.General.core.model_baseline import ConditionalVariationalAutoencoder

def train_baseline_external():
    # Setup parameters (Optimized for 29GB Dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 128
    batch_size = 64
    epochs = 100
    lr = 5e-4
    beta = 0.1  # KL weight
    
    # Data paths (External Dataset)
    data_dir = "data/external_dataset/extracted/data/filtered_waveforms"
    event_file = "data/external_dataset/extracted/data/events/20140101_20251101_0.0_9.0_9_339.txt"
    station_list_file = "data/station_list_external_full.json"
    
    if not os.path.exists(data_dir):
         print(f"Data directory {data_dir} not found.")
         return

    # Load fixed station list
    with open(station_list_file, 'r') as f:
        station_list = json.load(f)
    print(f"Loaded {len(station_list)} stations from {station_list_file}")

    # Load dataset
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=data_dir,
        event_file=event_file,
        channels=['HH'],
        magnitude_col='ML',
        station_list=station_list
    )
    
    num_stations = len(dataset.station_names)
    print(f"Dataset initialized with {num_stations} stations.")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Optimized DataLoader
    num_workers = 16 
    pin_memory = True if torch.cuda.is_available() else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn_with_metadata,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn_with_metadata,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True
    )
    
    # Initialize Baseline CVAE model
    model = ConditionalVariationalAutoencoder(
        in_channels=3, 
        latent_dim=latent_dim, 
        num_stations=num_stations
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting Baseline CVAE External Training - Latent Dim: {latent_dim}, Stations: {num_stations}")
    
    best_val_loss = float('inf')
    
    checkpoint_dir = Path("checkpoints_baseline_external")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0
        
        for batch in train_loader:
            specs, mags, locs, stations, _ = batch
            if specs is None: continue
            
            specs, mags, locs, stations = specs.to(device), mags.to(device), locs.to(device), stations.to(device)
            
            optimizer.zero_grad()
            
            recon, mu, logvar = model(specs, mags, locs, stations)
            
            # Loss
            recon_loss = torch.nn.functional.mse_loss(recon, specs, reduction='sum') / specs.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / specs.size(0)
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} (Recon: {train_recon/len(train_loader):.1f}, KL: {train_kl/len(train_loader):.1f})")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                specs, mags, locs, stations, _ = batch
                if specs is None: continue
                specs, mags, locs, stations = specs.to(device), mags.to(device), locs.to(device), stations.to(device)
                
                recon, mu, logvar = model(specs, mags, locs, stations)
                recon_loss = torch.nn.functional.mse_loss(recon, specs, reduction='sum') / specs.size(0)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / specs.size(0)
                loss = recon_loss + beta * kl_loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = checkpoint_dir / "baseline_cvae_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': {
                    'latent_dim': latent_dim,
                    'num_stations': num_stations,
                    'in_channels': 3
                }
            }, save_path)
            print(f"New best model saved to {save_path}")

    print("Baseline CVAE External Training Complete.")

if __name__ == "__main__":
    train_baseline_external()
