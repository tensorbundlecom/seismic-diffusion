import os
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from model import VariationalAutoencoder
from stft_dataset import SeismicSTFTDataset, collate_fn
from perceptual import PhaseNetPerceptualLoss, VGGPerceptualLoss


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
        is_vae=True,
        beta=1.0,
        recon_weight=1.0,
        perceptual_loss_fn=None,
        perceptual_weight=0.0,
        normalize_loss_terms=False,
        loss_norm_decay=0.99,
        loss_norm_eps=1e-8,
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
        self.recon_weight = recon_weight
        self.perceptual_loss_fn = perceptual_loss_fn
        self.perceptual_weight = perceptual_weight
        self.normalize_loss_terms = normalize_loss_terms
        self.loss_norm_decay = loss_norm_decay
        self.loss_norm_eps = loss_norm_eps
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Save config
        self.config = config
        with open(self.checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_report_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.tb_steps = {
            'train': 0,
            'val': 0,
            'test': 0,
        }
        self.loss_ema = {
            'recon': None,
            'kl': None,
            'perc': None,
        }
        self.last_epoch_reports = {
            'train': {},
            'val': {},
            'test': {},
        }

    def _log_batch_metrics(self, split, step, loss_metrics, scope='Batch'):
        """Log batch-level metrics with a consistent TensorBoard schema."""
        base = f"{scope}/{split}"
        self.writer.add_scalar(f'{base}/total_loss', loss_metrics['reported_total_loss'], step)
        self.writer.add_scalar(f'{base}/objective_total_loss', loss_metrics['objective_total_loss'], step)
        self.writer.add_scalar(f'{base}/recon_loss', loss_metrics['raw_recon_loss'], step)
        if self.is_vae:
            self.writer.add_scalar(f'{base}/kl_loss', loss_metrics['raw_kl_loss'], step)
        if self.perceptual_loss_fn is not None and self.perceptual_weight > 0:
            self.writer.add_scalar(f'{base}/perceptual_loss', loss_metrics['raw_perceptual_loss'], step)
        if self.normalize_loss_terms:
            self.writer.add_scalar(f'{base}/pre_priority_recon', loss_metrics['recon_loss'], step)
            if self.is_vae:
                self.writer.add_scalar(f'{base}/pre_priority_kl', loss_metrics['kl_loss'], step)
            if self.perceptual_loss_fn is not None and self.perceptual_weight > 0:
                self.writer.add_scalar(f'{base}/pre_priority_perceptual', loss_metrics['perceptual_loss'], step)

    def _log_epoch_metrics(
        self,
        split,
        epoch,
        total_loss,
        recon_loss=None,
        kl_loss=None,
        perceptual_loss=None,
    ):
        """Log epoch-level metrics with a consistent TensorBoard schema."""
        base = f'Epoch/{split}'
        self.writer.add_scalar(f'{base}/total_loss', total_loss, epoch)
        if recon_loss is not None:
            self.writer.add_scalar(f'{base}/recon_loss', recon_loss, epoch)
        if kl_loss is not None:
            self.writer.add_scalar(f'{base}/kl_loss', kl_loss, epoch)
        if perceptual_loss is not None:
            self.writer.add_scalar(f'{base}/perceptual_loss', perceptual_loss, epoch)

    def _update_loss_ema(self, name, value):
        """Update EMA scale tracker for a loss term."""
        current = self.loss_ema.get(name)
        if current is None:
            self.loss_ema[name] = float(value)
            return
        d = self.loss_norm_decay
        self.loss_ema[name] = d * current + (1.0 - d) * float(value)

    def _pre_priority_term(self, name, raw_term, update_norm_stats):
        """
        Return the term used right before weighting/prioritization.
        If normalization is enabled, this is raw_term / EMA(raw_term).
        """
        raw_value = raw_term.detach().item()
        if not self.normalize_loss_terms:
            return raw_term, raw_value

        if update_norm_stats:
            self._update_loss_ema(name, raw_value)
        elif self.loss_ema.get(name) is None:
            # Use current value if no EMA history is available yet.
            self.loss_ema[name] = raw_value

        scale = self.loss_ema[name]
        normalized = raw_term / (scale + self.loss_norm_eps)
        return normalized, normalized.detach().item()

    def _compute_total_loss(self, reconstructed, spectrograms, mu=None, logvar=None, update_norm_stats=False):
        """Compute weighted loss from pre-priority terms and expose raw/pre-priority metrics."""
        metrics = {
            'recon_loss': None,
            'kl_loss': None,
            'perceptual_loss': 0.0,
            'raw_recon_loss': 0.0,
            'raw_kl_loss': 0.0,
            'raw_perceptual_loss': 0.0,
        }

        if self.is_vae:
            batch_size = spectrograms.size(0)
            recon_raw = F.mse_loss(reconstructed, spectrograms, reduction='sum') / batch_size
            kl_raw = VariationalAutoencoder.kl_divergence(mu, logvar) / batch_size
            recon_term, recon_term_item = self._pre_priority_term('recon', recon_raw, update_norm_stats)
            kl_term, kl_term_item = self._pre_priority_term('kl', kl_raw, update_norm_stats)
        else:
            recon_raw = self.criterion(reconstructed, spectrograms)
            kl_raw = None
            recon_term, recon_term_item = self._pre_priority_term('recon', recon_raw, update_norm_stats)
            kl_term = None
            kl_term_item = None

        total_loss = self.recon_weight * recon_term
        reported_total = self.recon_weight * recon_raw
        perceptual_term_item = 0.0
        perceptual_raw = None
        if self.perceptual_loss_fn is not None and self.perceptual_weight > 0:
            perceptual_raw = self.perceptual_loss_fn(reconstructed, spectrograms)
            perceptual_term, perceptual_term_item = self._pre_priority_term(
                'perc', perceptual_raw, update_norm_stats
            )
            total_loss = total_loss + self.perceptual_weight * perceptual_term
            reported_total = reported_total + self.perceptual_weight * perceptual_raw

        if kl_term is not None:
            total_loss = total_loss + self.beta * kl_term
            reported_total = reported_total + self.beta * kl_raw

        metrics['recon_loss'] = recon_term_item
        metrics['raw_recon_loss'] = recon_raw.detach().item()
        if kl_term_item is not None:
            metrics['kl_loss'] = kl_term_item
            metrics['raw_kl_loss'] = kl_raw.detach().item()
        if perceptual_raw is not None:
            metrics['perceptual_loss'] = perceptual_term_item
            metrics['raw_perceptual_loss'] = perceptual_raw.detach().item()
        metrics['objective_total_loss'] = total_loss.item()
        metrics['reported_total_loss'] = reported_total.detach().item()
        metrics['total_loss'] = metrics['reported_total_loss']
        return total_loss, metrics
        
    def train_epoch(self, eval_interval=None):
        """Train for one epoch with optional periodic evaluation."""
        self.model.train()
        epoch_objective_loss = 0.0
        epoch_report_total = 0.0
        epoch_recon_raw = 0.0
        epoch_kl_raw = 0.0
        epoch_perceptual_raw = 0.0
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
                loss, loss_metrics = self._compute_total_loss(
                    reconstructed, spectrograms, mu=mu, logvar=logvar, update_norm_stats=True
                )
            else:
                reconstructed = self.model(spectrograms)
                loss, loss_metrics = self._compute_total_loss(reconstructed, spectrograms, update_norm_stats=True)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_objective_loss += loss.item()
            epoch_report_total += loss_metrics['reported_total_loss']
            epoch_recon_raw += loss_metrics['raw_recon_loss']
            epoch_kl_raw += loss_metrics['raw_kl_loss']
            epoch_perceptual_raw += loss_metrics['raw_perceptual_loss']
            num_batches += 1
            self.tb_steps['train'] += 1
            train_step = self.tb_steps['train']
            
            # Update progress bar
            pbar_metrics = {'total': loss_metrics['reported_total_loss']}
            if self.is_vae:
                pbar_metrics.update({
                    'recon': loss_metrics['raw_recon_loss'],
                    'kl': loss_metrics['raw_kl_loss'],
                })
            if self.perceptual_loss_fn is not None and self.perceptual_weight > 0:
                pbar_metrics['perc'] = loss_metrics['raw_perceptual_loss']
            pbar.set_postfix(pbar_metrics)
            
            # Log to tensorboard every N batches
            if train_step % self.log_interval == 0:
                self._log_batch_metrics('train', train_step, loss_metrics, scope='Batch')

                # Periodic evaluation during training
                if eval_interval and train_step % eval_interval == 0:
                    self._quick_eval(train_step)

        if num_batches > 0:
            self.last_epoch_reports['train'] = {
                'total': epoch_report_total / num_batches,
                'recon': epoch_recon_raw / num_batches,
                'kl': epoch_kl_raw / num_batches if self.is_vae else None,
                'perceptual': epoch_perceptual_raw / num_batches
                if (self.perceptual_loss_fn is not None and self.perceptual_weight > 0)
                else None,
            }
        self.writer.flush()
        
        avg_objective_loss = epoch_objective_loss / num_batches if num_batches > 0 else 0.0
        return avg_objective_loss
    
    def _quick_eval(self, train_step):
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
                    val_loss, loss_metrics = self._compute_total_loss(
                        reconstructed, spectrograms, mu=mu, logvar=logvar
                    )
                    val_loss = val_loss.item()
                else:
                    reconstructed = self.model(spectrograms)
                    val_loss, loss_metrics = self._compute_total_loss(reconstructed, spectrograms)
                    val_loss = val_loss.item()
                
                self._log_batch_metrics('val', train_step, loss_metrics, scope='QuickEval')
                break  # Only evaluate on first batch
            
            # Test on first batch if available
            if self.test_loader is not None:
                for batch_idx, (spectrograms, metadata) in enumerate(self.test_loader):
                    if spectrograms is None:
                        continue
                    spectrograms = spectrograms.to(self.device)
                    
                    if self.is_vae:
                        reconstructed, mu, logvar = self.model(spectrograms)
                        test_loss, test_metrics = self._compute_total_loss(
                            reconstructed, spectrograms, mu=mu, logvar=logvar
                        )
                        test_loss = test_loss.item()
                    else:
                        reconstructed = self.model(spectrograms)
                        test_loss, test_metrics = self._compute_total_loss(reconstructed, spectrograms)
                        test_loss = test_loss.item()
                    
                    self._log_batch_metrics('test', train_step, test_metrics, scope='QuickEval')
                    break  # Only evaluate on first batch
        
        self.writer.flush()
        self.model.train()  # Back to training mode
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        epoch_objective_loss = 0.0
        epoch_report_total = 0.0
        epoch_recon_raw = 0.0
        epoch_kl_raw = 0.0
        epoch_perceptual_raw = 0.0
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
                    loss, loss_metrics = self._compute_total_loss(
                        reconstructed, spectrograms, mu=mu, logvar=logvar
                    )
                else:
                    reconstructed = self.model(spectrograms)
                    loss, loss_metrics = self._compute_total_loss(reconstructed, spectrograms)
                
                # Update metrics
                epoch_objective_loss += loss.item()
                epoch_report_total += loss_metrics['reported_total_loss']
                epoch_recon_raw += loss_metrics['raw_recon_loss']
                epoch_kl_raw += loss_metrics['raw_kl_loss']
                epoch_perceptual_raw += loss_metrics['raw_perceptual_loss']
                num_batches += 1
                self.tb_steps['val'] += 1
                val_step = self.tb_steps['val']
                
                # Update progress bar
                if self.is_vae:
                    pbar_metrics = {
                        'total': loss_metrics['reported_total_loss'],
                        'recon': loss_metrics['raw_recon_loss'],
                        'kl': loss_metrics['raw_kl_loss']
                    }
                else:
                    pbar_metrics = {'total': loss_metrics['reported_total_loss']}
                if self.perceptual_loss_fn is not None and self.perceptual_weight > 0:
                    pbar_metrics['perc'] = loss_metrics['raw_perceptual_loss']
                pbar.set_postfix(pbar_metrics)
                
                # Log to tensorboard every N batches
                if val_step % self.log_interval == 0:
                    self._log_batch_metrics('val', val_step, loss_metrics, scope='Batch')
                
                # Log first batch images to tensorboard
                if batch_idx == 0:
                    self._log_images(spectrograms, reconstructed, prefix='Val')
        
        avg_objective_loss = epoch_objective_loss / num_batches if num_batches > 0 else 0.0
        avg_report_total = epoch_report_total / num_batches if num_batches > 0 else 0.0
        
        avg_recon_loss = epoch_recon_raw / num_batches if num_batches > 0 else None
        avg_kl_loss = (epoch_kl_raw / num_batches) if (self.is_vae and num_batches > 0) else None
        avg_perceptual_loss = (
            epoch_perceptual_raw / num_batches
            if (num_batches > 0 and self.perceptual_loss_fn is not None and self.perceptual_weight > 0)
            else None
        )
        if num_batches > 0:
            self._log_epoch_metrics(
                split='val',
                epoch=self.current_epoch,
                total_loss=avg_report_total,
                recon_loss=avg_recon_loss,
                kl_loss=avg_kl_loss,
                perceptual_loss=avg_perceptual_loss,
            )
            self.last_epoch_reports['val'] = {
                'total': avg_report_total,
                'recon': avg_recon_loss,
                'kl': avg_kl_loss,
                'perceptual': avg_perceptual_loss,
            }
        
        # Flush writer to ensure logs are written
        self.writer.flush()
        
        return avg_objective_loss
    
    def test(self, num_images=8):
        """Test the model and log sample input-output pairs."""
        if self.test_loader is None:
            print("No test loader provided, skipping test evaluation.")
            return None
            
        self.model.eval()
        epoch_objective_loss = 0.0
        epoch_report_total = 0.0
        epoch_recon_raw = 0.0
        epoch_kl_raw = 0.0
        epoch_perceptual_raw = 0.0
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
                    loss, loss_metrics = self._compute_total_loss(
                        reconstructed, spectrograms, mu=mu, logvar=logvar
                    )
                else:
                    reconstructed = self.model(spectrograms)
                    loss, loss_metrics = self._compute_total_loss(reconstructed, spectrograms)
                
                # Update metrics
                epoch_objective_loss += loss.item()
                epoch_report_total += loss_metrics['reported_total_loss']
                epoch_recon_raw += loss_metrics['raw_recon_loss']
                epoch_kl_raw += loss_metrics['raw_kl_loss']
                epoch_perceptual_raw += loss_metrics['raw_perceptual_loss']
                num_batches += 1
                self.tb_steps['test'] += 1
                test_step = self.tb_steps['test']
                
                # Update progress bar
                if self.is_vae:
                    pbar_metrics = {
                        'total': loss_metrics['reported_total_loss'],
                        'recon': loss_metrics['raw_recon_loss'],
                        'kl': loss_metrics['raw_kl_loss']
                    }
                else:
                    pbar_metrics = {'total': loss_metrics['reported_total_loss']}
                if self.perceptual_loss_fn is not None and self.perceptual_weight > 0:
                    pbar_metrics['perc'] = loss_metrics['raw_perceptual_loss']
                pbar.set_postfix(pbar_metrics)
                
                # Log to tensorboard every N batches
                if test_step % self.log_interval == 0:
                    self._log_batch_metrics('test', test_step, loss_metrics, scope='Batch')
                
                # Log first batch images to tensorboard
                if batch_idx == 0:
                    self._log_images(spectrograms, reconstructed, num_images=num_images, prefix='Test')
        
        avg_objective_loss = epoch_objective_loss / num_batches if num_batches > 0 else 0.0
        avg_report_total = epoch_report_total / num_batches if num_batches > 0 else 0.0
        print(f"\nTest Loss: {avg_report_total:.6f}")
        
        # Print VAE-specific losses
        avg_recon_loss = None
        avg_kl_loss = None
        if self.is_vae and num_batches > 0:
            avg_recon_loss = epoch_recon_raw / num_batches
            avg_kl_loss = epoch_kl_raw / num_batches
            print(f"Test Reconstruction Loss: {avg_recon_loss:.6f}")
            print(f"Test KL Divergence: {avg_kl_loss:.6f}")
        avg_perceptual_loss = None
        if num_batches > 0 and self.perceptual_loss_fn is not None and self.perceptual_weight > 0:
            avg_perceptual_loss = epoch_perceptual_raw / num_batches
            print(f"Test Perceptual Loss: {avg_perceptual_loss:.6f}")
        if num_batches > 0:
            self._log_epoch_metrics(
                split='test',
                epoch=self.current_epoch,
                total_loss=avg_report_total,
                recon_loss=avg_recon_loss,
                kl_loss=avg_kl_loss,
                perceptual_loss=avg_perceptual_loss,
            )
            self.last_epoch_reports['test'] = {
                'total': avg_report_total,
                'recon': avg_recon_loss,
                'kl': avg_kl_loss,
                'perceptual': avg_perceptual_loss,
            }
        
        self.writer.flush()
        return avg_objective_loss
    
    def _log_images(self, inputs, reconstructed, num_images=4, prefix='Val'):
        """Log sample images to tensorboard."""
        num_images = min(num_images, inputs.size(0))
        split = str(prefix).lower()
        
        # Log original and reconstructed spectrograms
        for i in range(num_images):
            # Original
            self.writer.add_image(
                f'Images/{split}/sample_{i}/input',
                inputs[i],
                self.current_epoch
            )
            # Reconstructed
            self.writer.add_image(
                f'Images/{split}/sample_{i}/reconstructed',
                reconstructed[i],
                self.current_epoch
            )
            
            # Also log the difference (error)
            diff = torch.abs(inputs[i] - reconstructed[i])
            self.writer.add_image(
                f'Images/{split}/sample_{i}/error',
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
            'best_val_report_loss': self.best_val_report_loss,
            'tb_steps': self.tb_steps,
            'loss_ema': self.loss_ema,
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
        self.best_val_report_loss = checkpoint.get('best_val_report_loss', self.best_val_report_loss)
        self.tb_steps = checkpoint.get('tb_steps', self.tb_steps)
        self.loss_ema = checkpoint.get('loss_ema', self.loss_ema)
        
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
        last_test_epoch = None
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train (with periodic evaluation if enabled)
            train_loss = self.train_epoch(eval_interval=eval_interval)
            self.train_losses.append(train_loss)
            
            # Validate
            val_objective_loss = self.validate()
            self.val_losses.append(val_objective_loss)
            
            # Log train epoch metrics using raw reported values
            train_report = self.last_epoch_reports.get('train', {})
            self._log_epoch_metrics(
                split='train',
                epoch=epoch,
                total_loss=train_report.get('total', train_loss),
                recon_loss=train_report.get('recon'),
                kl_loss=train_report.get('kl'),
                perceptual_loss=train_report.get('perceptual'),
            )
            
            # Flush to make sure data is written
            self.writer.flush()
            
            # Print epoch summary
            val_report = self.last_epoch_reports.get('val', {})
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_report.get('total', train_loss):.6f}")
            print(f"  Val Loss:   {val_report.get('total', val_objective_loss):.6f}")
            
            # Periodically evaluate on test set
            if self.test_loader is not None and (epoch + 1) % test_interval == 0:
                test_loss = self.test()
                if test_loss is not None:
                    test_report = self.last_epoch_reports.get('test', {})
                    print(f"  Test Loss:  {test_report.get('total', test_loss):.6f}")
                    self.writer.flush()
                    last_test_epoch = epoch
            
            # Save checkpoint
            is_best = val_objective_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_objective_loss
                self.best_val_report_loss = val_report.get('total', val_objective_loss)
                print(f"  New best validation loss!")
            
            if (epoch + 1) % save_interval == 0 or is_best:
                self.save_checkpoint(is_best=is_best)
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_report_loss:.6f}")
        
        # Run test evaluation if test loader is provided
        if self.test_loader is not None and last_test_epoch != self.current_epoch:
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
    parser.add_argument('--latent_channels', type=int, default=4,
                        help='Number of channels in the latent space (4 gives 45x compression for 129x111 inputs)')
    parser.add_argument('--recon_weight', type=float, default=1.0,
                        help='Weight for reconstruction term after optional loss normalization.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Beta parameter for VAE loss. 1.0 = standard VAE; keeps the latent close to N(0,I) for diffusion.')
    parser.add_argument('--use_phasenet_perceptual', action='store_true',
                        help='Enable PhaseNet-based perceptual loss in addition to reconstruction + KL losses.')
    parser.add_argument('--phasenet_weight', type=float, default=0.05,
                        help='Weight for PhaseNet perceptual loss term (used only when --use_phasenet_perceptual is set).')
    parser.add_argument('--phasenet_pretrained', type=str, default='stead',
                        help='Name of pretrained PhaseNet weights to load via seisbench.')
    parser.add_argument('--use_vgg_perceptual', action='store_true',
                        help='Enable VGG-based perceptual loss in addition to reconstruction + KL losses.')
    parser.add_argument('--vgg_weight', type=float, default=0.05,
                        help='Weight for VGG perceptual loss term (used only when --use_vgg_perceptual is set).')
    parser.add_argument('--vgg_no_imagenet_weights', action='store_true',
                        help='Initialize VGG perceptual backbone without ImageNet pretrained weights.')
    parser.add_argument('--vgg_no_resize', action='store_true',
                        help='Disable resizing spectrogram inputs to 224x224 before VGG feature extraction.')
    parser.add_argument('--normalize_loss_terms', action='store_true',
                        help='Normalize recon/KL/perceptual terms with EMA before applying weights.')
    parser.add_argument('--loss_norm_decay', type=float, default=0.99,
                        help='EMA decay for loss-term normalization (used with --normalize_loss_terms).')
    parser.add_argument('--loss_norm_eps', type=float, default=1e-8,
                        help='Numerical epsilon for loss-term normalization.')

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
    model = VariationalAutoencoder(in_channels=3, latent_channels=args.latent_channels)
    print(
        f"Using beta-VAE with recon_weight={args.recon_weight}, beta={args.beta}"
    )
    if args.normalize_loss_terms:
        print(
            "Loss-term normalization enabled "
            f"(ema_decay={args.loss_norm_decay}, eps={args.loss_norm_eps}). "
            "Training uses normalized terms internally; terminal/TensorBoard report raw terms."
        )
    
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

    # Optional perceptual loss (choose one)
    perceptual_loss_fn = None
    perceptual_weight = 0.0
    perceptual_type = 'none'

    if args.use_phasenet_perceptual and args.use_vgg_perceptual:
        raise ValueError("Please enable only one perceptual option: PhaseNet OR VGG.")

    if args.use_vgg_perceptual:
        try:
            perceptual_loss_fn = VGGPerceptualLoss(
                use_imagenet_weights=not args.vgg_no_imagenet_weights,
                resize_to_224=not args.vgg_no_resize,
                device=str(device),
            )
            perceptual_weight = args.vgg_weight
            perceptual_type = 'vgg'
            print(
                "VGG perceptual loss enabled "
                f"(imagenet_weights={not args.vgg_no_imagenet_weights}, "
                f"resize_to_224={not args.vgg_no_resize}, weight={args.vgg_weight})."
            )
        except Exception as exc:
            print(f"Warning: Could not enable VGG perceptual loss. Falling back to base loss only. Reason: {exc}")
    elif args.use_phasenet_perceptual:
        try:
            perceptual_loss_fn = PhaseNetPerceptualLoss(
                pretrained=args.phasenet_pretrained,
                nperseg=args.nperseg,
                noverlap=args.noverlap,
                nfft=args.nfft,
                device=str(device),
            )
            perceptual_weight = args.phasenet_weight
            perceptual_type = 'phasenet'
            print(
                "PhaseNet perceptual loss enabled "
                f"(pretrained='{args.phasenet_pretrained}', weight={args.phasenet_weight})."
            )
        except Exception as exc:
            print(f"Warning: Could not enable PhaseNet perceptual loss. Falling back to base loss only. Reason: {exc}")
    
    # Create trainer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir) / timestamp
    log_dir = Path(args.log_dir) / timestamp
    
    config = vars(args)
    config['timestamp'] = timestamp
    config['num_params'] = num_params
    config['perceptual_type'] = perceptual_type
    config['perceptual_active'] = perceptual_loss_fn is not None
    
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
        beta=args.beta,
        recon_weight=args.recon_weight,
        perceptual_loss_fn=perceptual_loss_fn,
        perceptual_weight=perceptual_weight if perceptual_loss_fn is not None else 0.0,
        normalize_loss_terms=args.normalize_loss_terms,
        loss_norm_decay=args.loss_norm_decay,
        loss_norm_eps=args.loss_norm_eps,
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
