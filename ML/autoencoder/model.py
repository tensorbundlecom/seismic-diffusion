import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder network for the convolutional autoencoder.
    Progressively downsamples the input image while increasing the number of features.
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        
        # Encoding layers
        self.encoder = nn.Sequential(
            # Input: (in_channels, H, W)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: (32, H/2, W/2)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: (64, H/4, W/4)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: (128, H/8, W/8)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Output: (256, H/16, W/16)
        )
        
        self.latent_dim = latent_dim
        
    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    """
    Decoder network for the convolutional autoencoder.
    Progressively upsamples the latent representation back to the original image size.
    """
    def __init__(self, out_channels=3, latent_dim=128, use_latent_projection=False):
        super(Decoder, self).__init__()
        
        self.use_latent_projection = use_latent_projection
        
        if use_latent_projection:
            # For VAE: project 1D latent vector back to spatial dimensions
            # Target spatial size for 256x256 input: (256, 16, 16)
            self.spatial_dim = 16
            self.projection_dim = 256 * self.spatial_dim * self.spatial_dim
            self.fc_projection = nn.Linear(latent_dim, self.projection_dim)
            
        # Decoding layers
        self.decoder = nn.Sequential(
            # Input: (256, H/16, W/16)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: (128, H/8, W/8)
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: (64, H/4, W/4)
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: (32, H/2, W/2)
            
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.Sigmoid()  # Output in range [0, 1]
            # Output: (out_channels, H, W)
        )
        
    def forward(self, x):
        if self.use_latent_projection:
            # x is (batch_size, latent_dim) - 1D vector
            batch_size = x.size(0)
            h = self.fc_projection(x)
            h = h.view(batch_size, 256, self.spatial_dim, self.spatial_dim)
            return self.decoder(h)
        else:
            # x is already spatial (batch_size, 256, H/16, W/16)
            return self.decoder(x)


class ConvAutoencoder(nn.Module):
    """
    Standard 2D Convolutional Autoencoder.
    
    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        latent_dim (int): Dimension of the latent space representation
        
    Example:
        >>> model = ConvAutoencoder(in_channels=3, latent_dim=128)
        >>> x = torch.randn(8, 3, 256, 256)  # Batch of 8 RGB images
        >>> reconstructed = model(x)
        >>> print(reconstructed.shape)  # torch.Size([8, 3, 256, 256])
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(out_channels=in_channels, latent_dim=latent_dim)
        
    def forward(self, x):
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Reconstructed tensor of the same shape as input
        """
        original_size = x.shape[2:]  # Store original H, W
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        
        # Resize to match input if necessary
        if reconstructed.shape[2:] != original_size:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
        
        return reconstructed
    
    def encode(self, x):
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Latent representation tensor
        """
        return self.encoder(x)
    
    def decode(self, latent):
        """
        Decode latent representation to reconstructed image.
        
        Args:
            latent: Latent tensor from encoder
            
        Returns:
            Reconstructed image tensor
        """
        return self.decoder(latent)
    
    def create_embedding(self, x):
        """
        Create latent embedding from input without reconstruction.
        
        This is useful for dimensionality reduction, visualization,
        or as features for downstream tasks.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Latent embedding tensor of shape (batch_size, 256, H/16, W/16)
        """
        return self.encode(x)


def get_model(in_channels=3, latent_dim=128, device='cuda'):
    """
    Factory function to create and initialize the autoencoder model.
    
    Args:
        in_channels (int): Number of input channels
        latent_dim (int): Dimension of the latent space
        device (str): Device to place the model on ('cuda' or 'cpu')
        
    Returns:
        ConvAutoencoder model on the specified device
    """
    model = ConvAutoencoder(in_channels=in_channels, latent_dim=latent_dim)
    model = model.to(device)
    return model


class VAEEncoder(nn.Module):
    """
    Encoder network for the Variational Autoencoder.
    Outputs parameters for the latent distribution (mean and log-variance).
    Uses standard non-spatial latent representation.
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(VAEEncoder, self).__init__()
        
        # Shared encoding layers
        self.encoder = nn.Sequential(
            # Input: (in_channels, H, W)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: (32, H/2, W/2)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: (64, H/4, W/4)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: (128, H/8, W/8)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Output: (256, H/16, W/16)
        )
        
        self.latent_dim = latent_dim
        
        # For 256x256 input: 256 * 16 * 16 = 65536
        # This will be calculated dynamically in the first forward pass
        self.flattened_dim = None
        self.fc_mu = None
        self.fc_logvar = None
        
    def _initialize_fc_layers(self, flattened_dim):
        """Initialize fully connected layers once we know the flattened dimension."""
        self.flattened_dim = flattened_dim
        self.fc_mu = nn.Linear(flattened_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, self.latent_dim)
        
    def forward(self, x):
        """
        Forward pass through the VAE encoder.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            mu: Mean of the latent distribution (batch_size, latent_dim)
            logvar: Log-variance of the latent distribution (batch_size, latent_dim)
        """
        batch_size = x.size(0)
        h = self.encoder(x)
        
        # Flatten spatial dimensions
        h_flat = h.view(batch_size, -1)
        
        # Initialize FC layers on first forward pass
        if self.fc_mu is None:
            self._initialize_fc_layers(h_flat.size(1))
            # Move to same device as input
            self.fc_mu = self.fc_mu.to(x.device)
            self.fc_logvar = self.fc_logvar.to(x.device)
        
        # Project to latent space (1D vectors)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        
        return mu, logvar


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) with convolutional layers.
    
    The VAE learns a probabilistic mapping to a latent space, allowing for
    generation of new samples by sampling from the learned distribution.
    
    Uses standard 1D latent vectors (non-spatial) for the reparameterization trick,
    which reduces graininess compared to spatial latent representations.
    
    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        latent_dim (int): Dimension of the latent space representation (1D vector)
        
    Example:
        >>> model = VariationalAutoencoder(in_channels=3, latent_dim=128)
        >>> x = torch.randn(8, 3, 256, 256)  # Batch of 8 RGB images
        >>> reconstructed, mu, logvar = model(x)
        >>> print(reconstructed.shape)  # torch.Size([8, 3, 256, 256])
        >>> 
        >>> # Calculate VAE loss
        >>> recon_loss = nn.functional.mse_loss(reconstructed, x, reduction='sum')
        >>> kl_loss = model.kl_divergence(mu, logvar)
        >>> total_loss = recon_loss + kl_loss
    """
    def __init__(self, in_channels=3, latent_dim=128):
        super(VariationalAutoencoder, self).__init__()
        
        self.encoder = VAEEncoder(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder(out_channels=in_channels, latent_dim=latent_dim, use_latent_projection=True)
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon is sampled from N(0, 1).
        
        Now operates on 1D latent vectors instead of spatial tensors,
        which significantly reduces graininess in reconstructions.
        
        Args:
            mu: Mean of the latent distribution (batch_size, latent_dim)
            logvar: Log-variance of the latent distribution (batch_size, latent_dim)
            
        Returns:
            Sampled latent vector z (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            reconstructed: Reconstructed tensor of the same shape as input
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
        """
        original_size = x.shape[2:]  # Store original H, W
        
        # Encode to latent distribution parameters
        mu, logvar = self.encoder(x)
        
        # Sample from the latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(z)
        
        # Resize to match input if necessary
        if reconstructed.shape[2:] != original_size:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
        
        return reconstructed, mu, logvar
    
    def encode(self, x):
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representation to reconstructed image.
        
        Args:
            z: Latent tensor (can be sampled or deterministic)
            
        Returns:
            Reconstructed image tensor
        """
        return self.decoder(z)
    
    def sample(self, num_samples, device='cuda'):
        """
        Generate new samples by sampling from the prior distribution N(0, I).
        
        Args:
            num_samples (int): Number of samples to generate
            device (str): Device to generate samples on
            
        Returns:
            Generated image tensor
        """
        # Sample from standard normal distribution in 1D latent space
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def create_embedding(self, x, use_mean=True):
        """
        Create latent embedding from input without reconstruction.
        
        For VAE, this returns the mean of the latent distribution by default
        (deterministic embedding), or a sampled embedding if use_mean=False.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            use_mean (bool): If True, return mu (deterministic). If False, sample z.
            
        Returns:
            Latent embedding tensor of shape (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x)
        if use_mean:
            return mu
        else:
            return self.reparameterize(mu, logvar)
    
    @staticmethod
    def kl_divergence(mu, logvar):
        """
        Compute the KL divergence loss: KL(Q(z|x) || P(z))
        where Q(z|x) is the approximate posterior and P(z) is the prior N(0, I).
        
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
            
        Returns:
            KL divergence loss (scalar)
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    @staticmethod
    def loss_function(reconstructed, original, mu, logvar, beta=1.0):
        """
        Compute the VAE loss: reconstruction loss + beta * KL divergence.
        
        Args:
            reconstructed: Reconstructed images
            original: Original input images
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
            beta (float): Weight for the KL divergence term (beta-VAE)
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
        
        # KL divergence
        kl_loss = VariationalAutoencoder.kl_divergence(mu, logvar)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def get_vae_model(in_channels=3, latent_dim=128, device='cuda'):
    """
    Factory function to create and initialize the VAE model.
    
    Args:
        in_channels (int): Number of input channels
        latent_dim (int): Dimension of the latent space
        device (str): Device to place the model on ('cuda' or 'cpu')
        
    Returns:
        VariationalAutoencoder model on the specified device
    """
    model = VariationalAutoencoder(in_channels=in_channels, latent_dim=latent_dim)
    model = model.to(device)
    return model


class CVAEEncoder(nn.Module):
    """
    Conditional Encoder network for the Conditional Variational Autoencoder.
    Takes both the input image and conditioning information (magnitude, location, station).
    Outputs parameters for the latent distribution (mean and log-variance).
    """
    def __init__(self, in_channels=3, latent_dim=128, num_stations=100, condition_dim=64):
        super(CVAEEncoder, self).__init__()
        
        # Shared encoding layers for the image
        self.encoder = nn.Sequential(
            # Input: (in_channels, H, W)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: (32, H/2, W/2)
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: (64, H/4, W/4)
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: (128, H/8, W/8)
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Output: (256, H/16, W/16)
        )
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        
        # Station embedding layer
        self.station_embedding = nn.Embedding(num_stations, condition_dim // 4)
        
        # Condition processing network
        # Input: magnitude (1) + location (3) + station_embedding (condition_dim // 4)
        condition_input_dim = 1 + 3 + condition_dim // 4
        self.condition_network = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(inplace=True),
        )
        
        # For 256x256 input: 256 * 16 * 16 = 65536
        # This will be calculated dynamically in the first forward pass
        self.flattened_dim = None
        self.fc_mu = None
        self.fc_logvar = None
        
    def _initialize_fc_layers(self, flattened_dim):
        """Initialize fully connected layers once we know the flattened dimension."""
        self.flattened_dim = flattened_dim
        # Combine image features and condition features
        combined_dim = flattened_dim + self.condition_dim
        self.fc_mu = nn.Linear(combined_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(combined_dim, self.latent_dim)
        
    def forward(self, x, magnitude, location, station_idx):
        """
        Forward pass through the CVAE encoder.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            magnitude: Event magnitude tensor of shape (batch_size,)
            location: Normalized location tensor of shape (batch_size, 3) [lat, lon, depth]
            station_idx: Station index tensor of shape (batch_size,)
            
        Returns:
            mu: Mean of the latent distribution (batch_size, latent_dim)
            logvar: Log-variance of the latent distribution (batch_size, latent_dim)
        """
        batch_size = x.size(0)
        
        # Encode the image
        h = self.encoder(x)
        
        # Flatten spatial dimensions
        h_flat = h.view(batch_size, -1)
        
        # Process the conditioning information
        station_emb = self.station_embedding(station_idx)  # (batch_size, condition_dim // 4)
        magnitude_expanded = magnitude.unsqueeze(1)  # (batch_size, 1)
        
        # Concatenate all conditioning information
        condition_input = torch.cat([magnitude_expanded, location, station_emb], dim=1)
        condition_features = self.condition_network(condition_input)  # (batch_size, condition_dim)
        
        # Initialize FC layers on first forward pass
        if self.fc_mu is None:
            self._initialize_fc_layers(h_flat.size(1))
            # Move to same device as input
            self.fc_mu = self.fc_mu.to(x.device)
            self.fc_logvar = self.fc_logvar.to(x.device)
        
        # Combine image and condition features
        combined_features = torch.cat([h_flat, condition_features], dim=1)
        
        # Project to latent space (1D vectors)
        mu = self.fc_mu(combined_features)
        logvar = self.fc_logvar(combined_features)
        
        return mu, logvar


class CVAEDecoder(nn.Module):
    """
    Conditional Decoder network for the Conditional Variational Autoencoder.
    Takes both the latent representation and conditioning information.
    """
    def __init__(self, out_channels=3, latent_dim=128, num_stations=100, condition_dim=64):
        super(CVAEDecoder, self).__init__()
        
        self.condition_dim = condition_dim
        
        # Station embedding layer
        self.station_embedding = nn.Embedding(num_stations, condition_dim // 4)
        
        # Condition processing network
        condition_input_dim = 1 + 3 + condition_dim // 4
        self.condition_network = nn.Sequential(
            nn.Linear(condition_input_dim, condition_dim),
            nn.ReLU(inplace=True),
            nn.Linear(condition_dim, condition_dim),
            nn.ReLU(inplace=True),
        )
        
        # Project combined latent + condition vector back to spatial dimensions
        # Target spatial size for 256x256 input: (256, 16, 16)
        self.spatial_dim = 16
        self.projection_dim = 256 * self.spatial_dim * self.spatial_dim
        self.fc_projection = nn.Linear(latent_dim + condition_dim, self.projection_dim)
        
        # Decoding layers
        self.decoder = nn.Sequential(
            # Input: (256, H/16, W/16)
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Output: (128, H/8, W/8)
            
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Output: (64, H/4, W/4)
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Output: (32, H/2, W/2)
            
            nn.ConvTranspose2d(32, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in range [0, 1]
            # Output: (out_channels, H, W)
        )
        
    def forward(self, z, magnitude, location, station_idx):
        """
        Forward pass through the CVAE decoder.
        
        Args:
            z: Latent tensor (batch_size, latent_dim)
            magnitude: Event magnitude tensor of shape (batch_size,)
            location: Normalized location tensor of shape (batch_size, 3)
            station_idx: Station index tensor of shape (batch_size,)
            
        Returns:
            Reconstructed image tensor
        """
        batch_size = z.size(0)
        
        # Process the conditioning information
        station_emb = self.station_embedding(station_idx)
        magnitude_expanded = magnitude.unsqueeze(1)
        
        # Concatenate all conditioning information
        condition_input = torch.cat([magnitude_expanded, location, station_emb], dim=1)
        condition_features = self.condition_network(condition_input)
        
        # Combine latent and condition features
        combined = torch.cat([z, condition_features], dim=1)
        
        # Project to spatial dimensions
        h = self.fc_projection(combined)
        h = h.view(batch_size, 256, self.spatial_dim, self.spatial_dim)
        
        # Decode
        return self.decoder(h)


class ConditionalVariationalAutoencoder(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) with convolutional layers.
    
    The CVAE conditions the generation process on metadata including:
    - Event magnitude
    - Event location (latitude, longitude, depth - normalized)
    - Station information (as embedding)
    
    This allows the model to learn location-aware and station-specific patterns
    in seismic waveforms, and enables controlled generation of waveforms with
    specific characteristics.
    
    Args:
        in_channels (int): Number of input channels (e.g., 3 for E/N/Z components)
        latent_dim (int): Dimension of the latent space representation
        num_stations (int): Number of unique stations (for embedding)
        condition_dim (int): Dimension of the conditioning feature vector
        
    Example:
        >>> model = ConditionalVariationalAutoencoder(
        ...     in_channels=3, latent_dim=128, num_stations=100, condition_dim=64
        ... )
        >>> x = torch.randn(8, 3, 256, 256)
        >>> magnitude = torch.randn(8)
        >>> location = torch.randn(8, 3)
        >>> station_idx = torch.randint(0, 100, (8,))
        >>> reconstructed, mu, logvar = model(x, magnitude, location, station_idx)
    """
    def __init__(self, in_channels=3, latent_dim=128, num_stations=100, condition_dim=64):
        super(ConditionalVariationalAutoencoder, self).__init__()
        
        self.encoder = CVAEEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim
        )
        self.decoder = CVAEDecoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            num_stations=num_stations,
            condition_dim=condition_dim
        )
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_stations = num_stations
        
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        where epsilon is sampled from N(0, 1).
        
        Args:
            mu: Mean of the latent distribution (batch_size, latent_dim)
            logvar: Log-variance of the latent distribution (batch_size, latent_dim)
            
        Returns:
            Sampled latent vector z (batch_size, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, magnitude, location, station_idx):
        """
        Forward pass through the CVAE.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            magnitude: Event magnitude tensor of shape (batch_size,)
            location: Normalized location tensor of shape (batch_size, 3)
            station_idx: Station index tensor of shape (batch_size,)
            
        Returns:
            reconstructed: Reconstructed tensor of the same shape as input
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
        """
        original_size = x.shape[2:]  # Store original H, W
        
        # Encode to latent distribution parameters (conditioned on metadata)
        mu, logvar = self.encoder(x, magnitude, location, station_idx)
        
        # Sample from the latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode (conditioned on metadata)
        reconstructed = self.decoder(z, magnitude, location, station_idx)
        
        # Resize to match input if necessary
        if reconstructed.shape[2:] != original_size:
            reconstructed = torch.nn.functional.interpolate(
                reconstructed,
                size=original_size,
                mode='bilinear',
                align_corners=False
            )
        
        return reconstructed, mu, logvar
    
    def encode(self, x, magnitude, location, station_idx):
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            magnitude: Event magnitude tensor of shape (batch_size,)
            location: Normalized location tensor of shape (batch_size, 3)
            station_idx: Station index tensor of shape (batch_size,)
            
        Returns:
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
        """
        return self.encoder(x, magnitude, location, station_idx)
    
    def decode(self, z, magnitude, location, station_idx):
        """
        Decode latent representation to reconstructed image.
        
        Args:
            z: Latent tensor (batch_size, latent_dim)
            magnitude: Event magnitude tensor of shape (batch_size,)
            location: Normalized location tensor of shape (batch_size, 3)
            station_idx: Station index tensor of shape (batch_size,)
            
        Returns:
            Reconstructed image tensor
        """
        return self.decoder(z, magnitude, location, station_idx)
    
    def sample(self, num_samples, magnitude, location, station_idx, device='cuda'):
        """
        Generate new samples by sampling from the prior distribution N(0, I),
        conditioned on specified metadata.
        
        Args:
            num_samples (int): Number of samples to generate
            magnitude: Event magnitude tensor of shape (num_samples,)
            location: Normalized location tensor of shape (num_samples, 3)
            station_idx: Station index tensor of shape (num_samples,)
            device (str): Device to generate samples on
            
        Returns:
            Generated image tensor
        """
        # Sample from standard normal distribution in 1D latent space
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z, magnitude, location, station_idx)
        return samples
    
    def create_embedding(self, x, magnitude, location, station_idx, use_mean=True):
        """
        Create latent embedding from input and conditioning information without reconstruction.
        
        For CVAE, this returns the mean of the conditioned latent distribution by default
        (deterministic embedding), or a sampled embedding if use_mean=False.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            magnitude: Event magnitude tensor of shape (batch_size,)
            location: Normalized location tensor of shape (batch_size, 3)
            station_idx: Station index tensor of shape (batch_size,)
            use_mean (bool): If True, return mu (deterministic). If False, sample z.
            
        Returns:
            Latent embedding tensor of shape (batch_size, latent_dim)
        """
        mu, logvar = self.encode(x, magnitude, location, station_idx)
        if use_mean:
            return mu
        else:
            return self.reparameterize(mu, logvar)
    
    @staticmethod
    def kl_divergence(mu, logvar):
        """
        Compute the KL divergence loss: KL(Q(z|x,c) || P(z))
        where Q(z|x,c) is the approximate posterior conditioned on x and c,
        and P(z) is the prior N(0, I).
        
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
            
        Returns:
            KL divergence loss (scalar)
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    @staticmethod
    def loss_function(reconstructed, original, mu, logvar, beta=1.0):
        """
        Compute the CVAE loss: reconstruction loss + beta * KL divergence.
        
        Args:
            reconstructed: Reconstructed images
            original: Original input images
            mu: Mean of the latent distribution
            logvar: Log-variance of the latent distribution
            beta (float): Weight for the KL divergence term (beta-CVAE)
            
        Returns:
            total_loss: Combined loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence loss
        """
        # Reconstruction loss (MSE)
        recon_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')
        
        # KL divergence
        kl_loss = ConditionalVariationalAutoencoder.kl_divergence(mu, logvar)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def get_cvae_model(in_channels=3, latent_dim=128, num_stations=100, condition_dim=64, device='cuda'):
    """
    Factory function to create and initialize the CVAE model.
    
    Args:
        in_channels (int): Number of input channels
        latent_dim (int): Dimension of the latent space
        num_stations (int): Number of unique stations
        condition_dim (int): Dimension of conditioning features
        device (str): Device to place the model on ('cuda' or 'cpu')
        
    Returns:
        ConditionalVariationalAutoencoder model on the specified device
    """
    model = ConditionalVariationalAutoencoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        num_stations=num_stations,
        condition_dim=condition_dim
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test the models
    print("="*60)
    print("Testing Convolutional Autoencoder...")
    print("="*60)
    
    # Create model
    model = ConvAutoencoder(in_channels=3, latent_dim=128)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test with a sample input
    batch_size = 4
    height, width = 256, 256
    x = torch.randn(batch_size, 3, height, width)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    reconstructed = model(x)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test encoder and decoder separately
    latent = model.encode(x)
    print(f"Latent shape: {latent.shape}")
    
    decoded = model.decode(latent)
    print(f"Decoded shape: {decoded.shape}")
    
    print("\nAutoencoder test successful!")
    
    # Test VAE
    print("\n" + "="*60)
    print("Testing Variational Autoencoder...")
    print("="*60)
    
    vae = VariationalAutoencoder(in_channels=3, latent_dim=128)
    print(f"VAE created with {sum(p.numel() for p in vae.parameters()):,} parameters")
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    reconstructed, mu, logvar = vae(x)
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test loss calculation
    total_loss, recon_loss, kl_loss = vae.loss_function(reconstructed, x, mu, logvar, beta=1.0)
    print(f"\nLoss calculation:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL divergence: {kl_loss.item():.4f}")
    
    # Test sampling
    samples = vae.sample(num_samples=2, device='cpu')
    print(f"\nGenerated samples shape: {samples.shape}")
    
    print("\nVAE test successful!")
    
    # Test CVAE
    print("\n" + "="*60)
    print("Testing Conditional Variational Autoencoder...")
    print("="*60)
    
    cvae = ConditionalVariationalAutoencoder(
        in_channels=3,
        latent_dim=128,
        num_stations=50,
        condition_dim=64
    )
    print(f"CVAE created with {sum(p.numel() for p in cvae.parameters()):,} parameters")
    
    # Create conditioning data
    magnitude = torch.randn(batch_size)  # Random magnitudes
    location = torch.randn(batch_size, 3)  # Random normalized locations [lat, lon, depth]
    station_idx = torch.randint(0, 50, (batch_size,))  # Random station indices
    
    print(f"\nInput shape: {x.shape}")
    print(f"Magnitude shape: {magnitude.shape}")
    print(f"Location shape: {location.shape}")
    print(f"Station indices shape: {station_idx.shape}")
    
    # Forward pass
    reconstructed, mu, logvar = cvae(x, magnitude, location, station_idx)
    print(f"\nReconstructed shape: {reconstructed.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    
    # Test loss calculation
    total_loss, recon_loss, kl_loss = cvae.loss_function(reconstructed, x, mu, logvar, beta=1.0)
    print(f"\nLoss calculation:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Reconstruction loss: {recon_loss.item():.4f}")
    print(f"  KL divergence: {kl_loss.item():.4f}")
    
    # Test conditional sampling
    samples = cvae.sample(
        num_samples=2,
        magnitude=magnitude[:2],
        location=location[:2],
        station_idx=station_idx[:2],
        device='cpu'
    )
    print(f"\nConditional generated samples shape: {samples.shape}")
    
    print("\nCVAE test successful!")
    print("="*60)
