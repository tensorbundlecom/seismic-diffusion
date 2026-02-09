import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import os

# Add necessary paths for imports
# Add necessary paths for imports (Project Root)
# Add necessary paths for imports (Project Root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))
from ML.autoencoder.experiments.General.core.stft_dataset import SeismicSTFTDatasetWithMetadata, collate_fn_with_metadata
from ML.autoencoder.experiments.NormalizingFlow.core.model_flow import FlowCVAE
from ML.autoencoder.experiments.NormalizingFlow.core.loss_utils import flow_cvae_loss_function

def train_flow_experiment():
    # Setup parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 128
    batch_size = 16
    epochs = 100
    lr = 5e-4
    beta = 0.1 # Weight for KL/Flow term
    
    # Data paths
    data_dir = "data/filtered_waveforms_broadband"
    event_file = "data/events/koeri_catalog.txt"
    station_list_file = "data/station_list_125.json"
    
    if not os.path.exists(data_dir):
         print(f"Data directory {data_dir} not found. Running in dummy mode.")
         return

    # Load fixed station list
    import json
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_metadata)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_with_metadata)
    
    # Initialize Flow-CVAE model
    model = FlowCVAE(
        in_channels=3, 
        latent_dim=latent_dim, 
        num_stations=num_stations,
        flow_layers=8 # Increased flexibility
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting Normalizing Flow Experiment - Latent Dim: {latent_dim}, Flow Layers: 8")
    
    best_val_loss = float('inf')
    
    experiment_dir = Path("ML/autoencoder/experiments/NormalizingFlow")
    checkpoint_dir = experiment_dir / "checkpoints"
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
            
            # Forward pass: returns reconstructed, mu, logvar, zk, log_det
            recon, mu, logvar, zk, log_det = model(specs, mags, locs, stations)
            
            loss, recon_l, kl_l = flow_cvae_loss_function(recon, specs, mu, logvar, zk, log_det, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_l.item()
            train_kl += kl_l.item()
            
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
                
                recon, mu, logvar, zk, log_det = model(specs, mags, locs, stations)
                loss, _, _ = flow_cvae_loss_function(recon, specs, mu, logvar, zk, log_det, beta=beta)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = checkpoint_dir / "flow_cvae_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': {
                    'latent_dim': latent_dim,
                    'num_stations': num_stations,
                    'in_channels': 3,
                    'flow_layers': 8
                }
            }, save_path)
            print(f"New best model saved to {save_path}")

    print("Normalizing Flow Training Complete.")

if __name__ == "__main__":
    train_flow_experiment()
