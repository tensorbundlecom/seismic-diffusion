import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add path to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

# Import from fixed baseline model using absolute paths
from ML.autoencoder.experiments.General.core.model_baseline import Decoder, CVAEEncoder

class ConditionalAffineCouplingLayer(nn.Module):
    """
    Conditional Affine Coupling Layer for RealNVP.
    Splits the input into two halves and transforms one half conditioned on the other and external context.
    """
    def __init__(self, dim, condition_dim, hidden_dim=256, mask_type='even'):
        super(ConditionalAffineCouplingLayer, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.mask_type = mask_type
        
        # d is the split point
        self.d = dim // 2
        
        # Networks to compute scale (s) and translation (t)
        # Input to these networks: the first half of z (d dims) + condition (condition_dim)
        input_dim = self.d + condition_dim
        
        self.st_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, (dim - self.d) * 2) # Outputs both s and t
        )

    def forward(self, z, c):
        # z: (batch_size, dim), c: (batch_size, condition_dim)
        
        if self.mask_type == 'even':
            z1, z2 = z[:, :self.d], z[:, self.d:]
        else:
            z2, z1 = z[:, :self.d], z[:, self.d:]
            
        # Compute scale and translation
        st_input = torch.cat([z1, c], dim=1)
        st = self.st_net(st_input)
        
        s = st[:, :(self.dim - self.d)]
        t = st[:, (self.dim - self.d):]
        
        # Use tanh for scale stabilization, or just exp if standard
        s = torch.tanh(s)
        
        # Transformation
        y1 = z1
        y2 = z2 * torch.exp(s) + t
        
        if self.mask_type == 'even':
            y = torch.cat([y1, y2], dim=1)
        else:
            y = torch.cat([y2, y1], dim=1)
            
        # Log-determinant Jacobian for this layer is sum(s)
        log_det_jacobian = s.sum(dim=1)
        
        return y, log_det_jacobian

class NormalizingFlow(nn.Module):
    """
    A sequence of Conditional Affine Coupling Layers.
    """
    def __init__(self, dim, condition_dim, num_layers=6, hidden_dim=256):
        super(NormalizingFlow, self).__init__()
        self.layers = nn.ModuleList([
            ConditionalAffineCouplingLayer(
                dim, condition_dim, hidden_dim, 
                mask_type='even' if i % 2 == 0 else 'odd'
            ) for i in range(num_layers)
        ])

    def forward(self, z, c):
        log_det_total = torch.zeros(z.size(0), device=z.device)
        for layer in self.layers:
            z, log_det = layer(z, c)
            log_det_total += log_det
        return z, log_det_total

class FlowCVAE(nn.Module):
    """
    Conditional VAE enhanced with Normalizing Flows in the latent space.
    """
    def __init__(self, in_channels=3, latent_dim=128, num_stations=100, condition_dim=64, flow_layers=6):
        super(FlowCVAE, self).__init__()
        
        self.encoder = CVAEEncoder(in_channels, latent_dim, num_stations, condition_dim)
        # Use existing Decoder but projected from latent_dim + condition_dim
        from ML.autoencoder.experiments.General.core.model_baseline import CVAEDecoder
        self.decoder = CVAEDecoder(in_channels, latent_dim, num_stations, condition_dim)
        
        # The condition network in CVAEEncoder produces context features
        # We'll use these same features to condition the Flow
        self.flow = NormalizingFlow(latent_dim, condition_dim, num_layers=flow_layers)
        
        self.latent_dim = latent_dim
        self._dummy_init()

    def _dummy_init(self):
        """Force initialization of lazy layers in baseline components."""
        with torch.no_grad():
            dummy_in = torch.randn(1, 3, 129, 111)
            dummy_mag = torch.zeros(1)
            dummy_loc = torch.zeros(1, 3)
            dummy_stat = torch.zeros(1, dtype=torch.long)
            self.forward(dummy_in, dummy_mag, dummy_loc, dummy_stat)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, magnitude, location, station_idx):
        batch_size = x.size(0)
        original_size = x.shape[2:]
        
        # 1. Encode to initial latent distribution (Gaussian approximation)
        # We reuse the condition network outputs if we want, but CVAEEncoder.forward
        # doesn't expose them easily. Let's re-calculate or intercept.
        
        # Actually, let's look at CVAEEncoder.forward to see if we can get condition features
        # It's better to just reuse the encoder's logic for conditioning
        
        # Intercept condition features for the flow
        station_emb = self.encoder.station_embedding(station_idx)
        magnitude_expanded = magnitude.unsqueeze(1)
        condition_input = torch.cat([magnitude_expanded, location, station_emb], dim=1)
        condition_features = self.encoder.condition_network(condition_input)
        
        # Standard encoding with image
        h = self.encoder.encoder(x)
        h_flat = h.view(batch_size, -1)
        
        # Initialize FC if needed (baseline lazy registration)
        if self.encoder.fc_mu is None:
            self.encoder._initialize_fc_layers(h_flat.size(1))
            self.encoder.fc_mu = self.encoder.fc_mu.to(x.device)
            self.encoder.fc_logvar = self.encoder.fc_logvar.to(x.device)
            
        combined_features = torch.cat([h_flat, condition_features], dim=1)
        mu = self.encoder.fc_mu(combined_features)
        logvar = self.encoder.fc_logvar(combined_features)
        
        # 2. Sample from initial posterior q(z0|x,c)
        z0 = self.reparameterize(mu, logvar)
        
        # 3. Transform through Normalizing Flow: zK = fK(...f1(z0)...)
        zk, log_det = self.flow(z0, condition_features)
        
        # 4. Decode
        reconstructed = self.decoder(zk, magnitude, location, station_idx)
        
        # Resize to match input if necessary
        if reconstructed.shape[2:] != original_size:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed, size=original_size, mode='bilinear', align_corners=False
            )
            
        return reconstructed, mu, logvar, zk, log_det

    def sample(self, num_samples, magnitude, location, station_idx, device='cuda'):
        """
        Sample from the prior. Here we assume the flow is on the posterior.
        If the prior is still N(0, I), we just sample zK and decode.
        """
        zk = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(zk, magnitude, location, station_idx)
        return samples
