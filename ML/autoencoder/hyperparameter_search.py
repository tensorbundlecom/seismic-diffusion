"""
Hyperparameter search for VAE using Optuna.
Optimizes latent_dim, learning rate, batch size, weight decay, and beta.
"""
import os
import argparse
from pathlib import Path
from datetime import datetime
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import optuna
from optuna.trial import TrialState

from model import VariationalAutoencoder
from stft_dataset import SeismicSTFTDataset, collate_fn
from train import Trainer

# Suppress some warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


def create_datasets(args, seed):
    """Create train and validation datasets."""
    # Create dataset
    dataset = SeismicSTFTDataset(
        data_dir=args.data_dir,
        channels=args.channels,
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        nfft=args.nfft,
        normalize=True,
        log_scale=True,
    )
    
    # Split into train and validation (no test set needed for HPO)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset


def objective(trial, args, train_dataset, val_dataset, device):
    """
    Optuna objective function to optimize.
    
    Args:
        trial: Optuna trial object
        args: Command line arguments
        train_dataset: Training dataset
        val_dataset: Validation dataset
        device: Device to train on
        
    Returns:
        Best validation loss for this trial
    """
    # Suggest hyperparameters
    latent_dim = trial.suggest_categorical('latent_dim', [64, 128, 256, 512])
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    beta = trial.suggest_float('beta', 0.1, 10.0, log=True)
    
    # Print trial info
    print(f"\nTrial {trial.number}:")
    print(f"  latent_dim: {latent_dim}")
    print(f"  lr: {lr:.6f}")
    print(f"  batch_size: {batch_size}")
    print(f"  weight_decay: {weight_decay:.6f}")
    print(f"  beta: {beta:.4f}")
    
    # Create data loaders with suggested batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device.type == 'cuda' else False,
    )
    
    # Create model with suggested latent_dim
    model = VariationalAutoencoder(in_channels=3, latent_dim=latent_dim)
    model = model.to(device)
    
    # Create optimizer with suggested lr and weight_decay
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Create criterion
    criterion = nn.MSELoss()
    
    # Create temporary checkpoint and log directories for this trial
    # Use /tmp to avoid cluttering workspace during HPO
    checkpoint_dir = f"/tmp/optuna_trial_{trial.number}"
    log_dir = f"/tmp/optuna_logs_trial_{trial.number}"
    
    # Create config for this trial
    config = {
        'latent_dim': latent_dim,
        'lr': lr,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'beta': beta,
        'trial_number': trial.number,
    }
    
    # Create trainer using existing Trainer class
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
        test_loader=None,  # No test loader needed for HPO
        log_interval=100,  # Less frequent logging for HPO
        is_vae=True,
        beta=beta,
    )
    
    # Train with early stopping and pruning
    try:
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.max_epochs_per_trial):
            # Train one epoch
            train_loss = trainer.train_epoch(eval_interval=None)
            
            # Validate
            val_loss = trainer.validate()
            
            # Report intermediate value for Optuna pruning
            trial.report(val_loss, epoch)
            
            # Check if trial should be pruned
            if trial.should_prune():
                print(f"  Trial pruned at epoch {epoch + 1}")
                raise optuna.TrialPruned()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= args.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
            
            # Update trainer's epoch counter
            trainer.current_epoch = epoch + 1
        
        print(f"  Best val loss: {best_val_loss:.6f}")
        
        # Clean up temporary directories
        import shutil
        for temp_dir in [checkpoint_dir, log_dir]:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return best_val_loss
        
    except optuna.TrialPruned:
        # Clean up on pruning
        import shutil
        for temp_dir in [checkpoint_dir, log_dir]:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        raise
        
    except Exception as e:
        print(f"  Trial failed with error: {e}")
        # Clean up on error
        import shutil
        for temp_dir in [checkpoint_dir, log_dir]:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        return float('inf')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter search for seismic VAE")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='../../data/filtered_waveforms',
                        help='Path to filtered waveforms directory')
    parser.add_argument('--channels', type=str, nargs='+', default=["HH"],
                        help='Channel types to include (e.g., HH HN EH BH)')
    
    # STFT arguments
    parser.add_argument('--nperseg', type=int, default=256,
                        help='Length of each segment for STFT')
    parser.add_argument('--noverlap', type=int, default=192,
                        help='Number of points to overlap between segments')
    parser.add_argument('--nfft', type=int, default=256,
                        help='Length of the FFT used')
    
    # Training arguments
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--max_epochs_per_trial', type=int, default=5,
                        help='Maximum epochs to train each trial')
    parser.add_argument('--early_stopping_patience', type=int, default=1,
                        help='Patience for early stopping')
    
    # Optuna arguments
    parser.add_argument('--n_trials', type=int, default=15,
                        help='Number of Optuna trials to run')
    parser.add_argument('--study_name', type=str, default='vae_hpo',
                        help='Name of the Optuna study')
    parser.add_argument('--storage', type=str, default=None,
                        help='Database URL for Optuna storage (e.g., sqlite:///optuna.db). If None, uses in-memory storage.')
    parser.add_argument('--load_if_exists', action='store_true',
                        help='Load existing study if it exists')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Stop study after this many seconds (None = no timeout)')
    
    # Pruning arguments
    parser.add_argument('--pruner', type=str, default='median',
                        choices=['none', 'median', 'hyperband'],
                        help='Pruner algorithm to use')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='hpo_results',
                        help='Directory to save results')
    
    return parser.parse_args()


def main():
    """Main hyperparameter search function."""
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = output_dir / f"config_{timestamp}.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to {config_path}")
    
    # Create datasets once (reused across trials)
    print(f"\nLoading dataset from {args.data_dir}...")
    train_dataset, val_dataset = create_datasets(args, args.seed)
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    
    # Create pruner
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )
    elif args.pruner == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=args.max_epochs_per_trial,
            reduction_factor=3,
        )
    else:
        pruner = optuna.pruners.NopPruner()
    
    # Create or load study
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
        direction='minimize',
        pruner=pruner,
    )
    
    print(f"\nStarting hyperparameter optimization with {args.n_trials} trials...")
    print(f"Study name: {args.study_name}")
    print(f"Pruner: {args.pruner}")
    print(f"Max epochs per trial: {args.max_epochs_per_trial}")
    print(f"Early stopping patience: {args.early_stopping_patience}")
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, train_dataset, val_dataset, device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "="*80)
    print("Hyperparameter Search Complete!")
    print("="*80)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print(f"\nStatistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len(pruned_trials)}")
    print(f"  Number of complete trials: {len(complete_trials)}")
    
    if len(complete_trials) > 0:
        print(f"\nBest trial:")
        trial = study.best_trial
        print(f"  Value (validation loss): {trial.value:.6f}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best hyperparameters
        best_params_path = output_dir / f"best_params_{timestamp}.json"
        with open(best_params_path, 'w') as f:
            best_params = {
                'best_val_loss': trial.value,
                'params': trial.params,
                'trial_number': trial.number,
            }
            json.dump(best_params, f, indent=4)
        print(f"\nBest parameters saved to {best_params_path}")
        
        # Save study summary
        summary_path = output_dir / f"study_summary_{timestamp}.json"
        trials_data = []
        for t in complete_trials:
            trials_data.append({
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'duration': t.duration.total_seconds() if t.duration else None,
            })
        
        summary = {
            'study_name': args.study_name,
            'timestamp': timestamp,
            'n_trials': len(study.trials),
            'n_complete': len(complete_trials),
            'n_pruned': len(pruned_trials),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'trials': trials_data,
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"Study summary saved to {summary_path}")
        
        # Print top 5 trials
        print("\nTop 5 trials:")
        sorted_trials = sorted(complete_trials, key=lambda t: t.value)[:5]
        for i, t in enumerate(sorted_trials, 1):
            print(f"\n{i}. Trial {t.number}")
            print(f"   Val Loss: {t.value:.6f}")
            print(f"   Params: {t.params}")
    
    else:
        print("\nNo trials completed successfully.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
