import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import os

# Add necessary paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from ..core.stft_dataset_fixed import SeismicSTFTDatasetWithMetadata, collate_fn_with_metadata
from ..core.model_full_cov import FullCovCVAE
from ..core.loss_utils import full_cov_loss_function

def train_experiment():
    # Setup parameters (minimal for experiment)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 128 # Increased as requested
    batch_size = 16
    epochs = 100
    lr = 1e-4
    
    # Data paths (assuming relative to project root)
    data_dir = "data/filtered_waveforms"
    event_file = "data/events/koeri_catalog.txt"
    
    if not os.path.exists(data_dir):
         print(f"Data directory {data_dir} not found. Running in dummy mode for script validation.")
         # You would normally exit or handle this.
         return

    # Load dataset
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=data_dir,
        event_file=event_file,
        channels=['HH'],
        magnitude_col='ML'
    )
    
    num_stations = len(dataset.station_names)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_metadata)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_metadata)
    
    # Initialize experimental model
    model = FullCovCVAE(in_channels=3, latent_dim=latent_dim, num_stations=num_stations).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting Full Covariance Experiment - Latent Dim: {latent_dim}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            specs, mags, locs, stations, _ = batch
            if specs is None: continue
            
            specs, mags, locs, stations = specs.to(device), mags.to(device), locs.to(device), stations.to(device)
            
            optimizer.zero_grad()
            recon, mu, L = model(specs, mags, locs, stations)
            
            loss, recon_l, kl_l = full_cov_loss_function(recon, specs, mu, L, beta=0.1)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}")
        
    # Save final model
    timestamp = Path("checkpoints_full_cov") / Path(event_file).stem
    timestamp.mkdir(parents=True, exist_ok=True)
    save_path = timestamp / "full_cov_cvae_best.pt"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss/len(train_loader),
        'config': {
            'latent_dim': latent_dim,
            'num_stations': num_stations,
            'in_channels': 3
        }
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_experiment()
