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

from model import ConditionalVariationalAutoencoder
from stft_dataset_with_metadata import SeismicSTFTDatasetWithMetadata, collate_fn_with_metadata


class CVAETrainer:
    """
    Trainer class for the Conditional Variational Autoencoder with seismic metadata.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        checkpoint_dir,
        log_dir,
        config,
        test_loader=None,
        log_interval=10,
        beta=1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
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
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch_idx, (spectrograms, magnitudes, locations, station_indices, metadata) in enumerate(pbar):
            if spectrograms is None:
                continue
                
            # Move to device
            spectrograms = spectrograms.to(self.device)
            magnitudes = magnitudes.to(self.device)
            locations = locations.to(self.device)
            station_indices = station_indices.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            reconstructed, mu, logvar = self.model(spectrograms, magnitudes, locations, station_indices)
            
            # Compute CVAE loss
            loss, recon_loss, kl_loss = ConditionalVariationalAutoencoder.loss_function(
                reconstructed, spectrograms, mu, logvar, beta=self.beta
            )
            
            # Normalize by batch size for consistency
            loss = loss / spectrograms.size(0)
            recon_loss_item = recon_loss.item() / spectrograms.size(0)
            kl_loss_item = kl_loss.item() / spectrograms.size(0)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss_item
            epoch_kl_loss += kl_loss_item
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss_item,
                'kl': kl_loss_item
            })
            
            # Log to tensorboard every N batches
            if (batch_idx + 1) % self.log_interval == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/batch_loss', loss.item(), global_step)
                self.writer.add_scalar('Train/recon_loss', recon_loss_item, global_step)
                self.writer.add_scalar('Train/kl_loss', kl_loss_item, global_step)
                
                # Log metadata statistics
                self.writer.add_scalar('Train/avg_magnitude', magnitudes.mean().item(), global_step)
                
                # Periodic evaluation during training
                if eval_interval and (batch_idx + 1) % eval_interval == 0:
                    self._quick_eval(global_step)
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_kl_loss = epoch_kl_loss / num_batches if num_batches > 0 else 0.0
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def _quick_eval(self, global_step):
        """Quick evaluation on a few batches of val/test data."""
        # Validate on first batch
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (spectrograms, magnitudes, locations, station_indices, metadata) in enumerate(self.val_loader):
                if spectrograms is None:
                    continue
                spectrograms = spectrograms.to(self.device)
                magnitudes = magnitudes.to(self.device)
                locations = locations.to(self.device)
                station_indices = station_indices.to(self.device)
                
                reconstructed, mu, logvar = self.model(spectrograms, magnitudes, locations, station_indices)
                val_loss, recon_loss, kl_loss = ConditionalVariationalAutoencoder.loss_function(
                    reconstructed, spectrograms, mu, logvar, beta=self.beta
                )
                val_loss = val_loss.item() / spectrograms.size(0)
                
                self.writer.add_scalar('Val/batch_loss', val_loss, global_step)
                break  # Only evaluate on first batch
            
            # Test on first batch if available
            if self.test_loader is not None:
                for batch_idx, (spectrograms, magnitudes, locations, station_indices, metadata) in enumerate(self.test_loader):
                    if spectrograms is None:
                        continue
                    spectrograms = spectrograms.to(self.device)
                    magnitudes = magnitudes.to(self.device)
                    locations = locations.to(self.device)
                    station_indices = station_indices.to(self.device)
                    
                    reconstructed, mu, logvar = self.model(spectrograms, magnitudes, locations, station_indices)
                    test_loss, _, _ = ConditionalVariationalAutoencoder.loss_function(
                        reconstructed, spectrograms, mu, logvar, beta=self.beta
                    )
                    test_loss = test_loss.item() / spectrograms.size(0)
                    
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
            for batch_idx, (spectrograms, magnitudes, locations, station_indices, metadata) in enumerate(pbar):
                if spectrograms is None:
                    continue
                    
                # Move to device
                spectrograms = spectrograms.to(self.device)
                magnitudes = magnitudes.to(self.device)
                locations = locations.to(self.device)
                station_indices = station_indices.to(self.device)
                
                # Forward pass
                reconstructed, mu, logvar = self.model(spectrograms, magnitudes, locations, station_indices)
                
                # Compute CVAE loss
                loss, recon_loss, kl_loss = ConditionalVariationalAutoencoder.loss_function(
                    reconstructed, spectrograms, mu, logvar, beta=self.beta
                )
                
                # Normalize by batch size
                loss = loss / spectrograms.size(0)
                recon_loss_item = recon_loss.item() / spectrograms.size(0)
                kl_loss_item = kl_loss.item() / spectrograms.size(0)
                
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss_item
                epoch_kl_loss += kl_loss_item
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'recon': recon_loss_item,
                    'kl': kl_loss_item
                })
                
                # Log to tensorboard every N batches
                if (batch_idx + 1) % self.log_interval == 0:
                    global_step = self.current_epoch * len(self.val_loader) + batch_idx
                    self.writer.add_scalar('Val/batch_loss', loss.item(), global_step)
                    self.writer.add_scalar('Val/recon_loss', recon_loss_item, global_step)
                    self.writer.add_scalar('Val/kl_loss', kl_loss_item, global_step)
                
                # Log first batch images to tensorboard
                if batch_idx == 0:
                    self._log_images(spectrograms, reconstructed, metadata, prefix='Val')
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_kl_loss = epoch_kl_loss / num_batches if num_batches > 0 else 0.0
        
        # Log epoch averages
        self.writer.add_scalar('Val/epoch_recon_loss', avg_recon_loss, self.current_epoch)
        self.writer.add_scalar('Val/epoch_kl_loss', avg_kl_loss, self.current_epoch)
        
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def test(self, num_images=8):
        """Test the model and log sample input-output pairs."""
        if self.test_loader is None:
            print("No test loader provided, skipping test evaluation.")
            return None, None, None
            
        self.model.eval()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f"Testing")
            for batch_idx, (spectrograms, magnitudes, locations, station_indices, metadata) in enumerate(pbar):
                if spectrograms is None:
                    continue
                    
                # Move to device
                spectrograms = spectrograms.to(self.device)
                magnitudes = magnitudes.to(self.device)
                locations = locations.to(self.device)
                station_indices = station_indices.to(self.device)
                
                # Forward pass
                reconstructed, mu, logvar = self.model(spectrograms, magnitudes, locations, station_indices)
                
                # Compute CVAE loss
                loss, recon_loss, kl_loss = ConditionalVariationalAutoencoder.loss_function(
                    reconstructed, spectrograms, mu, logvar, beta=self.beta
                )
                
                # Normalize by batch size
                loss = loss / spectrograms.size(0)
                recon_loss_item = recon_loss.item() / spectrograms.size(0)
                kl_loss_item = kl_loss.item() / spectrograms.size(0)
                
                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss_item
                epoch_kl_loss += kl_loss_item
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'recon': recon_loss_item,
                    'kl': kl_loss_item
                })
                
                # Log to tensorboard every N batches
                if (batch_idx + 1) % self.log_interval == 0:
                    global_step = self.current_epoch * len(self.test_loader) + batch_idx
                    self.writer.add_scalar('Test/batch_loss', loss.item(), global_step)
                    self.writer.add_scalar('Test/recon_loss', recon_loss_item, global_step)
                    self.writer.add_scalar('Test/kl_loss', kl_loss_item, global_step)
                
                # Log first batch images to tensorboard
                if batch_idx == 0:
                    self._log_images(spectrograms, reconstructed, metadata, num_images=num_images, prefix='Test')
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_kl_loss = epoch_kl_loss / num_batches if num_batches > 0 else 0.0
        
        print(f"\nTest Loss: {avg_loss:.6f}")
        print(f"Test Reconstruction Loss: {avg_recon_loss:.6f}")
        print(f"Test KL Divergence: {avg_kl_loss:.6f}")
        
        self.writer.flush()
        return avg_loss, avg_recon_loss, avg_kl_loss
    
    def _log_images(self, inputs, reconstructed, metadata, num_images=4, prefix='Val'):
        """Log sample images to tensorboard with metadata."""
        num_images = min(num_images, inputs.size(0))
        
        # Log original and reconstructed spectrograms
        for i in range(num_images):
            # Get metadata for this sample
            meta = metadata[i]
            mag = meta['magnitude']
            station = meta['station_name']
            
            # Create descriptive tag
            tag_suffix = f"mag_{mag:.1f}_{station}"
            
            # Original
            self.writer.add_image(
                f'{prefix}/sample_{i}_{tag_suffix}/input',
                inputs[i],
                self.current_epoch
            )
            # Reconstructed
            self.writer.add_image(
                f'{prefix}/sample_{i}_{tag_suffix}/reconstructed',
                reconstructed[i],
                self.current_epoch
            )
            
            # Also log the difference (error)
            diff = torch.abs(inputs[i] - reconstructed[i])
            self.writer.add_image(
                f'{prefix}/sample_{i}_{tag_suffix}/error',
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
            train_loss, train_recon, train_kl = self.train_epoch(eval_interval=eval_interval)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_recon, val_kl = self.validate()
            self.val_losses.append(val_loss)
            
            # Log to tensorboard (using epoch number to avoid gaps in x-axis)
            self.writer.add_scalar('Train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('Train/epoch_recon_loss', train_recon, epoch)
            self.writer.add_scalar('Train/epoch_kl_loss', train_kl, epoch)
            self.writer.add_scalar('Val/epoch_loss', val_loss, epoch)
            self.writer.add_scalar('Val/epoch_recon_loss', val_recon, epoch)
            self.writer.add_scalar('Val/epoch_kl_loss', val_kl, epoch)
            
            # Also log with epoch number for easy viewing (consolidated view)
            self.writer.add_scalar('Loss/train_per_epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/val_per_epoch', val_loss, epoch)
            self.writer.add_scalar('Loss/train_recon_per_epoch', train_recon, epoch)
            self.writer.add_scalar('Loss/val_recon_per_epoch', val_recon, epoch)
            self.writer.add_scalar('Loss/train_kl_per_epoch', train_kl, epoch)
            self.writer.add_scalar('Loss/val_kl_per_epoch', val_kl, epoch)
            
            # Flush to make sure data is written
            self.writer.flush()
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f} (Recon: {train_recon:.6f}, KL: {train_kl:.6f})")
            print(f"  Val Loss:   {val_loss:.6f} (Recon: {val_recon:.6f}, KL: {val_kl:.6f})")
            
            # Periodically evaluate on test set
            if self.test_loader is not None and (epoch + 1) % test_interval == 0:
                test_loss, test_recon, test_kl = self.test()
                if test_loss is not None:
                    print(f"  Test Loss:  {test_loss:.6f} (Recon: {test_recon:.6f}, KL: {test_kl:.6f})")
                    # Log test loss per epoch (using epoch number to avoid gaps)
                    self.writer.add_scalar('Test/epoch_loss', test_loss, epoch)
                    self.writer.add_scalar('Test/epoch_recon_loss', test_recon, epoch)
                    self.writer.add_scalar('Test/epoch_kl_loss', test_kl, epoch)
                    self.writer.add_scalar('Loss/test_per_epoch', test_loss, epoch)
                    self.writer.add_scalar('Loss/test_recon_per_epoch', test_recon, epoch)
                    self.writer.add_scalar('Loss/test_kl_per_epoch', test_kl, epoch)
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
        
        # Run final test evaluation if test loader is provided
        if self.test_loader is not None:
            print("\nRunning final test evaluation...")
            final_test_loss, final_test_recon, final_test_kl = self.test()
            if final_test_loss is not None:
                # Log final test results with a special tag
                self.writer.add_scalar('Final/test_loss', final_test_loss, num_epochs - 1)
                self.writer.add_scalar('Final/test_recon_loss', final_test_recon, num_epochs - 1)
                self.writer.add_scalar('Final/test_kl_loss', final_test_kl, num_epochs - 1)
                self.writer.flush()
        
        self.writer.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Conditional VAE for seismic waveforms with metadata")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../../data/filtered_waveforms',
                        help='Path to filtered waveforms directory')
    parser.add_argument('--event_file', type=str, default='../../data/events/20140101_20251101_0.0_9.0_9_339.txt',
                        help='Path to event catalog file')
    parser.add_argument('--channels', type=str, nargs='+', default=['HH'],
                        help='Channel types to include (e.g., HH HN EH BH)')
    parser.add_argument('--magnitude_col', type=str, default='xM',
                        help='Which magnitude column to use (xM, MD, ML, Mw, Ms, Mb)')
    
    # STFT arguments
    parser.add_argument('--nperseg', type=int, default=256,
                        help='Length of each segment for STFT')
    parser.add_argument('--noverlap', type=int, default=192,
                        help='Number of points to overlap between segments')
    parser.add_argument('--nfft', type=int, default=256,
                        help='Length of the FFT used')
    
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Dimension of latent space')
    parser.add_argument('--condition_dim', type=int, default=64,
                        help='Dimension of conditioning features')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta parameter for CVAE loss (beta-CVAE). Higher values encourage stronger disentanglement.')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.000007,
                        help='Weight decay for optimizer')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--test_interval', type=int, default=5,
                        help='Evaluate on test set every N epochs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log batch losses to tensorboard every N batches')
    parser.add_argument('--eval_interval', type=int, default=None,
                        help='Evaluate on val/test every N training batches (None = only at epoch end)')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_cvae',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs_cvae',
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
    dataset = SeismicSTFTDatasetWithMetadata(
        data_dir=args.data_dir,
        event_file=args.event_file,
        channels=args.channels,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        normalize=True,
        log_scale=True,
        magnitude_col=args.magnitude_col,
    )
    
    # Get number of unique stations for the model
    num_stations = len(dataset.station_names)
    print(f"Number of unique stations: {num_stations}")
    
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
        collate_fn=collate_fn_with_metadata,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_metadata,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_with_metadata,
        pin_memory=True if args.device == 'cuda' else False,
    )
    
    # Create model
    print(f"Creating Conditional VAE model...")
    model = ConditionalVariationalAutoencoder(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_stations=num_stations,
        condition_dim=args.condition_dim
    )
    print(f"Using beta-CVAE with beta={args.beta}")
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    log_dir = Path(args.log_dir) / timestamp
    
    config = vars(args)
    config['timestamp'] = timestamp
    config['num_params'] = num_params
    config['num_stations'] = num_stations
    
    trainer = CVAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        config=config,
        test_loader=test_loader,
        log_interval=args.log_interval,
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
