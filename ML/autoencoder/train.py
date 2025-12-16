import os
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from model import ConvAutoencoder, VariationalAutoencoder
from stft_dataset import SeismicSTFTDataset, collate_fn


class Trainer:
    """
    Trainer class for the seismic autoencoder.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        checkpoint_dir,
        log_dir,
        config,
        test_loader=None,
        log_interval=10,
        is_vae=False,
        beta=1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.is_vae = is_vae
        self.beta = beta
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Save config
        self.config = config
        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, eval_interval=None):
        """Train for one epoch with optional periodic evaluation."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch_idx, (spectrograms, metadata) in enumerate(pbar):
            if spectrograms is None:
                continue
                
            # Move to device
            spectrograms = spectrograms.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.is_vae:
                reconstructed, mu, logvar = self.model(spectrograms)
                # Compute VAE loss
                loss, recon_loss, kl_loss = VariationalAutoencoder.loss_function(
                    reconstructed, spectrograms, mu, logvar, beta=self.beta
                )
                # Normalize by batch size for consistency
                loss = loss / spectrograms.size(0)
                recon_loss_item = recon_loss.item() / spectrograms.size(0)
                kl_loss_item = kl_loss.item() / spectrograms.size(0)
            else:
                reconstructed = self.model(spectrograms)
                # Compute standard autoencoder loss
                loss = self.criterion(reconstructed, spectrograms)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard every N batches
            if (batch_idx + 1) % self.log_interval == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/batch_loss', loss.item(), global_step)
                
                # Log VAE-specific metrics
                if self.is_vae:
                    self.writer.add_scalar('Train/recon_loss', recon_loss_item, global_step)
                    self.writer.add_scalar('Train/kl_loss', kl_loss_item, global_step)
                
                # Periodic evaluation during training
                if eval_interval and (batch_idx + 1) % eval_interval == 0:
                    self._quick_eval(global_step)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _quick_eval(self, global_step):
        """Quick evaluation on a few batches of val/test data."""
        # Validate on first batch
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (spectrograms, metadata) in enumerate(self.val_loader):
                if spectrograms is None:
                    continue
                spectrograms = spectrograms.to(self.device)
                
                if self.is_vae:
                    reconstructed, mu, logvar = self.model(spectrograms)
                    val_loss, recon_loss, kl_loss = VariationalAutoencoder.loss_function(
                        reconstructed, spectrograms, mu, logvar, beta=self.beta
                    )
                    val_loss = val_loss.item() / spectrograms.size(0)
                else:
                    reconstructed = self.model(spectrograms)
                    val_loss = self.criterion(reconstructed, spectrograms).item()
                
                self.writer.add_scalar('Val/batch_loss', val_loss, global_step)
                break  # Only evaluate on first batch
            
            # Test on first batch if available
            if self.test_loader is not None:
                for batch_idx, (spectrograms, metadata) in enumerate(self.test_loader):
                    if spectrograms is None:
                        continue
                    spectrograms = spectrograms.to(self.device)
                    
                    if self.is_vae:
                        reconstructed, mu, logvar = self.model(spectrograms)
                        test_loss, _, _ = VariationalAutoencoder.loss_function(
                            reconstructed, spectrograms, mu, logvar, beta=self.beta
                        )
                        test_loss = test_loss.item() / spectrograms.size(0)
                    else:
                        reconstructed = self.model(spectrograms)
                        test_loss = self.criterion(reconstructed, spectrograms).item()
                    
                    self.writer.add_scalar('Test/batch_loss', test_loss, global_step)
                    break  # Only evaluate on first batch
        
        self.writer.flush()
        self.model.train()  # Back to training mode
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
            for batch_idx, (spectrograms, metadata) in enumerate(pbar):
                if spectrograms is None:
                    continue
                    
                # Move to device
                spectrograms = spectrograms.to(self.device)
                
                # Forward pass
                if self.is_vae:
                    reconstructed, mu, logvar = self.model(spectrograms)
                    # Compute VAE loss
                    loss, recon_loss, kl_loss = VariationalAutoencoder.loss_function(
                        reconstructed, spectrograms, mu, logvar, beta=self.beta
                    )
                    # Normalize by batch size
                    loss = loss / spectrograms.size(0)
                    recon_loss_item = recon_loss.item() / spectrograms.size(0)
                    kl_loss_item = kl_loss.item() / spectrograms.size(0)
                    
                    epoch_recon_loss += recon_loss_item
                    epoch_kl_loss += kl_loss_item
                else:
                    reconstructed = self.model(spectrograms)
                    # Compute standard loss
                    loss = self.criterion(reconstructed, spectrograms)
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                if self.is_vae:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'recon': recon_loss_item,
                        'kl': kl_loss_item
                    })
                else:
                    pbar.set_postfix({'loss': loss.item()})
                
                # Log to tensorboard every N batches
                if (batch_idx + 1) % self.log_interval == 0:
                    global_step = self.current_epoch * len(self.val_loader) + batch_idx
                    self.writer.add_scalar('Val/batch_loss', loss.item(), global_step)
                    if self.is_vae:
                        self.writer.add_scalar('Val/recon_loss', recon_loss_item, global_step)
                        self.writer.add_scalar('Val/kl_loss', kl_loss_item, global_step)
                
                # Log first batch images to tensorboard
                if batch_idx == 0:
                    self._log_images(spectrograms, reconstructed, prefix='Val')
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        
        # Log epoch averages for VAE
        if self.is_vae and num_batches > 0:
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            self.writer.add_scalar('Val/epoch_recon_loss', avg_recon_loss, self.current_epoch)
            self.writer.add_scalar('Val/epoch_kl_loss', avg_kl_loss, self.current_epoch)
        
        return avg_loss
    
    def test(self, num_images=8):
        """Test the model and log sample input-output pairs."""
        if self.test_loader is None:
            print("No test loader provided, skipping test evaluation.")
            return None
            
        self.model.eval()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"Testing")
            for batch_idx, (spectrograms, metadata) in enumerate(pbar):
                if spectrograms is None:
                    continue
                    
                # Move to device
                spectrograms = spectrograms.to(self.device)
                
                # Forward pass
                if self.is_vae:
                    reconstructed, mu, logvar = self.model(spectrograms)
                    # Compute VAE loss
                    loss, recon_loss, kl_loss = VariationalAutoencoder.loss_function(
                        reconstructed, spectrograms, mu, logvar, beta=self.beta
                    )
                    # Normalize by batch size
                    loss = loss / spectrograms.size(0)
                    recon_loss_item = recon_loss.item() / spectrograms.size(0)
                    kl_loss_item = kl_loss.item() / spectrograms.size(0)
                    
                    epoch_recon_loss += recon_loss_item
                    epoch_kl_loss += kl_loss_item
                else:
                    reconstructed = self.model(spectrograms)
                    # Compute standard loss
                    loss = self.criterion(reconstructed, spectrograms)
                
                # Update metrics
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                if self.is_vae:
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'recon': recon_loss_item,
                        'kl': kl_loss_item
                    })
                else:
                    pbar.set_postfix({'loss': loss.item()})
                
                # Log to tensorboard every N batches
                if (batch_idx + 1) % self.log_interval == 0:
                    global_step = self.current_epoch * len(self.test_loader) + batch_idx
                    self.writer.add_scalar('Test/batch_loss', loss.item(), global_step)
                    if self.is_vae:
                        self.writer.add_scalar('Test/recon_loss', recon_loss_item, global_step)
                        self.writer.add_scalar('Test/kl_loss', kl_loss_item, global_step)
                
                # Log first batch images to tensorboard
                if batch_idx == 0:
                    self._log_images(spectrograms, reconstructed, num_images=num_images, prefix='Test')
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"\nTest Loss: {avg_loss:.6f}")
        
        # Print VAE-specific losses
        if self.is_vae and num_batches > 0:
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kl_loss = epoch_kl_loss / num_batches
            print(f"Test Reconstruction Loss: {avg_recon_loss:.6f}")
            print(f"Test KL Divergence: {avg_kl_loss:.6f}")
        
        self.writer.flush()
        return avg_loss
    
    def _log_images(self, inputs, reconstructed, num_images=4, prefix='Val'):
        """Log sample images to tensorboard."""
        num_images = min(num_images, inputs.size(0))
        
        # Log original and reconstructed spectrograms
        for i in range(num_images):
            # Original
            self.writer.add_image(
                f'{prefix}/sample_{i}/input',
                inputs[i],
                self.current_epoch
            )
            # Reconstructed
            self.writer.add_image(
                f'{prefix}/sample_{i}/reconstructed',
                reconstructed[i],
                self.current_epoch
            )
            
            # Also log the difference (error)
            diff = torch.abs(inputs[i] - reconstructed[i])
            self.writer.add_image(
                f'{prefix}/sample_{i}/error',
                diff,
                self.current_epoch
            )
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Keep only last N checkpoints
        self._cleanup_checkpoints(keep_last=3)
    
    def _cleanup_checkpoints(self, keep_last=3):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > keep_last:
            for ckpt in checkpoints[:-keep_last]:
                ckpt.unlink()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs, save_interval=5, test_interval=10, eval_interval=None):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        if self.test_loader is not None:
            print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Device: {self.device}")
        if eval_interval:
            print(f"Evaluating val/test every {eval_interval} training batches")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train (with periodic evaluation if enabled)
            train_loss = self.train_epoch(eval_interval=eval_interval)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            # Log to tensorboard (using epoch number to avoid gaps in x-axis)
            # Note: batch logs use global_step, but epoch logs use epoch number
            self.writer.add_scalar('Train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('Val/epoch_loss', val_loss, epoch)
            
            # Also log with epoch number for easy viewing
            self.writer.add_scalar('Loss/train_per_epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/val_per_epoch', val_loss, epoch)
            
            # Flush to make sure data is written
            self.writer.flush()
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            
            # Periodically evaluate on test set
            if self.test_loader is not None and (epoch + 1) % test_interval == 0:
                test_loss = self.test()
                if test_loss is not None:
                    print(f"  Test Loss:  {test_loss:.6f}")
                    # Log test loss per epoch (using epoch number to avoid gaps)
                    self.writer.add_scalar('Test/epoch_loss', test_loss, epoch)
                    self.writer.add_scalar('Loss/test_per_epoch', test_loss, epoch)
                    self.writer.flush()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"  New best validation loss!")
            
            if (epoch + 1) % save_interval == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Run test evaluation if test loader is provided
        if self.test_loader is not None:
            print("\nRunning test evaluation...")
            self.test()
        
        self.writer.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train seismic autoencoder")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../../data/filtered_waveforms',
                        help='Path to filtered waveforms directory')
    parser.add_argument('--channels', type=str, nargs='+', default=['HH'],
                        help='Channel types to include (e.g., HH HN EH BH)')
    
    # STFT arguments
    parser.add_argument('--nperseg', type=int, default=256,
                        help='Length of each segment for STFT')
    parser.add_argument('--noverlap', type=int, default=192,
                        help='Number of points to overlap between segments')
    parser.add_argument('--nfft', type=int, default=256,
                        help='Length of the FFT used')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='autoencoder',
                        choices=['autoencoder', 'vae'],
                        help='Type of model to train (autoencoder or vae)')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Dimension of latent space')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta parameter for VAE loss (beta-VAE). Higher values encourage stronger disentanglement.')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='Evaluate on test set every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log batch losses to tensorboard every N batches')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='Evaluate on val/test every N training batches (None = only at epoch end)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = SeismicSTFTDataset(
        data_dir=args.data_dir,
        channels=args.channels,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        normalize=True,
        log_scale=True,
    )
    
    # Split into train, validation, and test
    test_size = int(len(dataset) * 0.1)  # 10% for test
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    if args.model_type == 'vae':
        model = VariationalAutoencoder(in_channels=3, latent_dim=args.latent_dim)
        print(f"Using beta-VAE with beta={args.beta}")
    else:
        model = ConvAutoencoder(in_channels=3, latent_dim=args.latent_dim)
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    
    # Create trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    log_dir = Path(args.log_dir) / timestamp
    
    config = vars(args)
    config['timestamp'] = timestamp
    config['num_params'] = num_params
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        config=config,
        test_loader=test_loader,
        log_interval=args.log_interval,
        is_vae=(args.model_type == 'vae'),
        beta=args.beta,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train(
        num_epochs=args.num_epochs, 
        save_interval=args.save_interval,
        test_interval=args.test_interval,
        eval_interval=args.eval_interval
    )


if __name__ == "__main__":
    main()
