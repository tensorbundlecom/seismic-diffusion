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
    def __init__(self, out_channels=3, latent_dim=128):
        super(Decoder, self).__init__()
        
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
        
    def forward(self, x):
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


if __name__ == "__main__":
    # Test the model
    print("Testing Convolutional Autoencoder...")
    
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
    
    print("\nModel test successful!")
