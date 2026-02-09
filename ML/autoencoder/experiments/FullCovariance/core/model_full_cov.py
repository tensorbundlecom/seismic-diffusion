import torch
import torch.nn as nn
import sys
import os

# Add path to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# Import from fixed baseline model using absolute paths
from ML.autoencoder.experiments.General.core.model_baseline import Decoder, CVAEEncoder

class FullCovCVAEEncoder(nn.Module):
    """
    Encoder that outputs mu and the lower triangular Cholesky factor L for full covariance.
    """
    def __init__(self, in_channels=3, latent_dim=128, num_stations=100, condition_dim=64):
        super(FullCovCVAEEncoder, self).__init__()
        
        # Use existing image encoder logic
        self.encoder_base = CVAEEncoder(in_channels, latent_dim, num_stations, condition_dim)
        
        # We need to redefine the FC layers to produce full cov parameters
        # For N=latent_dim, we need N for mu and N(N+1)/2 for L
        self.latent_dim = latent_dim
        self.num_chol = (latent_dim * (latent_dim + 1)) // 2
        
        # These will be initialized dynamically in CVAEEncoder._initialize_fc_layers
        # But we need to override the output heads
        self.fc_mu = None # Will be Linear(combined_dim, latent_dim)
        self.fc_chol = None # Will be Linear(combined_dim, num_chol)

    def forward(self, x, magnitude, location, station_idx):
        batch_size = x.size(0)
        
        # Standard image encoding + condition processing logic
        # Since CVAEEncoder is built to produce mu/logvar, we reuse its logic but intercept
        
        # This is a bit tricky since CVAEEncoder.forward calls its own fc_mu/fc_logvar
        # Let's implement the logic explicitly to be safe
        
        # 1. Image features
        h = self.encoder_base.encoder(x)
        h_flat = h.view(batch_size, -1)
        
        # 2. Condition features
        station_emb = self.encoder_base.station_embedding(station_idx)
        magnitude_expanded = magnitude.unsqueeze(1)
        condition_input = torch.cat([magnitude_expanded, location, station_emb], dim=1)
        condition_features = self.encoder_base.condition_network(condition_input)
        
        combined_features = torch.cat([h_flat, condition_features], dim=1)
        
        # Initialize FC layers if needed
        if self.fc_mu is None:
            combined_dim = combined_features.size(1)
            self.fc_mu = nn.Linear(combined_dim, self.latent_dim).to(x.device)
            self.fc_chol = nn.Linear(combined_dim, self.num_chol).to(x.device)
            
        mu = self.fc_mu(combined_features)
        chol_flat = self.fc_chol(combined_features)
        
        # Construct lower triangular matrix L
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=x.device)
        
        # Fill L with chol_flat values
        # This is computationally expensive in a loop, but clear for an experiment
        # A more optimized way would use torch.tril
        tril_indices = torch.tril_indices(row=self.latent_dim, col=self.latent_dim, offset=0)
        L[:, tril_indices[0], tril_indices[1]] = chol_flat
        
        # Ensure positive diagonal for Cholesky factor
        diag_indices = torch.arange(self.latent_dim)
        L[:, diag_indices, diag_indices] = torch.exp(L[:, diag_indices, diag_indices])
        
        return mu, L

class FullCovCVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, num_stations=100, condition_dim=64):
        super(FullCovCVAE, self).__init__()
        # We increase latent_dim to 128 as requested.
        
        self.encoder = FullCovCVAEEncoder(in_channels, latent_dim, num_stations, condition_dim)
        # Use existing Decoder but projected from latent_dim + condition_dim
        from ML.autoencoder.experiments.General.core.model_baseline import CVAEDecoder
        self.decoder = CVAEDecoder(in_channels, latent_dim, num_stations, condition_dim)
        
        self.latent_dim = latent_dim
        
        # Mandatory dummy pass to initialize dynamic layers before optimizer creation
        self._dummy_init()

    def _dummy_init(self):
        """Force initialization of lazy layers."""
        with torch.no_grad():
            # Use a typical input size (3 channels, 129x111 spectrogram)
            dummy_in = torch.randn(1, 3, 129, 111)
            dummy_mag = torch.zeros(1)
            dummy_loc = torch.zeros(1, 3)
            dummy_stat = torch.zeros(1, dtype=torch.long)
            self.forward(dummy_in, dummy_mag, dummy_loc, dummy_stat)

    def reparameterize(self, mu, L):
        if not self.training:
            return mu
            
        batch_size = mu.size(0)
        eps = torch.randn(batch_size, self.latent_dim, 1, device=mu.device)
        
        # z = mu + L @ eps
        # L is (B, N, N), eps is (B, N, 1)
        z = mu.unsqueeze(2) + torch.bmm(L, eps)
        return z.squeeze(2)

    def forward(self, x, magnitude, location, station_idx):
        original_size = x.shape[2:]
        mu, L = self.encoder(x, magnitude, location, station_idx)
        z = self.reparameterize(mu, L)
        reconstructed = self.decoder(z, magnitude, location, station_idx)
        
        if reconstructed.shape[2:] != original_size:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, size=original_size, mode='bilinear', align_corners=False
            )
            
        return reconstructed, mu, L
