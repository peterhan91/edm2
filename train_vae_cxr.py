#!/usr/bin/env python3
"""Fine-tune VAE for CXR images with comprehensive validation."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py
from tqdm import tqdm
import click
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from diffusers import AutoencoderKL
from torchvision.utils import save_image

#----------------------------------------------------------------------------

class H5VAEDataset(Dataset):
    """H5 dataset for VAE training."""
    def __init__(self, h5_path, split='train', train_ratio=0.9):
        self.h5_file = h5py.File(h5_path, 'r')
        self.images = self.h5_file['images']
        
        # Split dataset
        total_size = len(self.images)
        train_size = int(total_size * train_ratio)
        
        if split == 'train':
            self.indices = np.arange(train_size)
        else:
            self.indices = np.arange(train_size, total_size)
        
        print(f"{split} dataset: {len(self.indices)} images")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = np.array(self.images[real_idx])
        
        # Handle different input dtypes consistently with EDM2
        if img.dtype == np.uint8:
            # Standard case: uint8 [0, 255] -> float32 [-1, 1]
            img = img.astype(np.float32) / 127.5 - 1.0
        elif img.dtype in [np.float32, np.float64]:
            # Float images - check range
            if img.min() >= 0 and img.max() <= 255:
                # Float in [0, 255] range
                img = img.astype(np.float32) / 127.5 - 1.0
            elif img.min() >= -1 and img.max() <= 1:
                # Already normalized to [-1, 1]
                img = img.astype(np.float32)
            else:
                # Assume [0, 1] range
                img = img.astype(np.float32) * 2 - 1.0
        
        return torch.from_numpy(img)
    
    def close(self):
        self.h5_file.close()

#----------------------------------------------------------------------------

class VAEValidator:
    """Comprehensive VAE validation metrics."""
    def __init__(self, device):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    def compute_metrics(self, real_images, recon_images):
        """Compute reconstruction metrics."""
        metrics = {}
        
        # Convert to numpy for SSIM/PSNR
        real_np = ((real_images + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        recon_np = ((recon_images + 1) * 127.5).clamp(0, 255).cpu().numpy().astype(np.uint8)
        
        # Compute per-image metrics
        ssim_scores = []
        psnr_scores = []
        
        for i in range(len(real_images)):
            # Move channel dimension to last
            real_img = real_np[i].transpose(1, 2, 0)
            recon_img = recon_np[i].transpose(1, 2, 0)
            
            # SSIM
            ssim_score = ssim(real_img, recon_img, channel_axis=2, data_range=255)
            ssim_scores.append(ssim_score)
            
            # PSNR
            psnr_score = psnr(real_img, recon_img, data_range=255)
            psnr_scores.append(psnr_score)
        
        metrics['ssim'] = np.mean(ssim_scores)
        metrics['psnr'] = np.mean(psnr_scores)
        
        # LPIPS (perceptual distance)
        with torch.no_grad():
            lpips_score = self.lpips_fn(real_images, recon_images).mean()
            metrics['lpips'] = lpips_score.item()
        
        # MSE in latent space
        metrics['mse'] = F.mse_loss(recon_images, real_images).item()
        
        return metrics

#----------------------------------------------------------------------------

def train_epoch(vae, train_loader, optimizer, device, kl_weight=1e-6):
    """Train VAE for one epoch."""
    vae.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        images = batch.to(device)
        
        # Forward pass
        outputs = vae(images)
        recon_images = outputs.sample
        
        # Compute losses
        recon_loss = F.mse_loss(recon_images, images)
        
        # KL divergence
        posterior = outputs.latent_dist
        kl_loss = posterior.kl().mean()
        
        # Total loss
        loss = recon_loss + kl_weight * kl_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    return {
        'loss': total_loss / len(train_loader),
        'recon_loss': total_recon_loss / len(train_loader),
        'kl_loss': total_kl_loss / len(train_loader)
    }

#----------------------------------------------------------------------------

def validate(vae, val_loader, validator, device, save_dir=None, epoch=0):
    """Validate VAE with comprehensive metrics."""
    vae.eval()
    
    all_metrics = {'ssim': [], 'psnr': [], 'lpips': [], 'mse': []}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Validating")):
            images = batch.to(device)
            
            # Reconstruct images
            outputs = vae(images)
            recon_images = outputs.sample
            
            # Compute metrics
            metrics = validator.compute_metrics(images, recon_images)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
            
            # Save sample reconstructions
            if save_dir and i == 0:
                n_samples = min(8, len(images))
                comparison = torch.cat([
                    images[:n_samples],
                    recon_images[:n_samples]
                ])
                save_image(
                    comparison,
                    os.path.join(save_dir, f'reconstruction_epoch_{epoch}.png'),
                    nrow=n_samples,
                    normalize=True,
                    value_range=(-1, 1)
                )
    
    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return avg_metrics

#----------------------------------------------------------------------------

def save_latent_histogram(vae, val_loader, device, save_dir, epoch):
    """Analyze latent space distribution."""
    vae.eval()
    
    all_means = []
    all_stds = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyzing latents"):
            images = batch.to(device)
            
            # Encode to latent space
            posterior = vae.encode(images).latent_dist
            
            all_means.append(posterior.mean.cpu().numpy())
            all_stds.append(posterior.std.cpu().numpy())
    
    all_means = np.concatenate(all_means)
    all_stds = np.concatenate(all_stds)
    
    # Plot histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Mean distribution
    axes[0, 0].hist(all_means.flatten(), bins=100, alpha=0.7)
    axes[0, 0].set_title(f'Latent Means Distribution (μ={all_means.mean():.3f}, σ={all_means.std():.3f})')
    axes[0, 0].set_xlabel('Mean Value')
    
    # Std distribution
    axes[0, 1].hist(all_stds.flatten(), bins=100, alpha=0.7)
    axes[0, 1].set_title(f'Latent Stds Distribution (μ={all_stds.mean():.3f}, σ={all_stds.std():.3f})')
    axes[0, 1].set_xlabel('Std Value')
    
    # Channel-wise statistics
    channel_means = all_means.mean(axis=(0, 2, 3))
    channel_stds = all_stds.mean(axis=(0, 2, 3))
    
    axes[1, 0].bar(range(len(channel_means)), channel_means)
    axes[1, 0].set_title('Channel-wise Mean Values')
    axes[1, 0].set_xlabel('Channel')
    
    axes[1, 1].bar(range(len(channel_stds)), channel_stds)
    axes[1, 1].set_title('Channel-wise Std Values')
    axes[1, 1].set_xlabel('Channel')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'latent_analysis_epoch_{epoch}.png'))
    plt.close()
    
    return {
        'latent_mean': all_means.mean(),
        'latent_std': all_means.std(),
        'latent_std_mean': all_stds.mean()
    }

#----------------------------------------------------------------------------

@click.command()
@click.option('--data', help='H5 dataset path', required=True)
@click.option('--output-dir', help='Output directory', required=True)
@click.option('--base-model', help='Base VAE model', default='stabilityai/sd-vae-ft-mse')
@click.option('--batch-size', help='Batch size', type=int, default=4)
@click.option('--lr', help='Learning rate', type=float, default=1e-5)
@click.option('--epochs', help='Number of epochs', type=int, default=50)
@click.option('--kl-weight', help='KL divergence weight', type=float, default=1e-6)
@click.option('--validate-every', help='Validation frequency', type=int, default=5)
@click.option('--save-every', help='Save frequency', type=int, default=10)
@click.option('--resume', help='Resume from checkpoint', type=str)
def train_vae(data, output_dir, base_model, batch_size, lr, epochs, kl_weight, 
              validate_every, save_every, resume):
    """Fine-tune VAE for CXR images with comprehensive validation.
    
    Examples:
    
    \b
    # Fine-tune VAE on CXR dataset
    python train_vae_cxr.py \\
        --data datasets/cxr_512.h5 \\
        --output-dir training-runs/vae-cxr \\
        --batch-size 8 \\
        --lr 1e-5 \\
        --epochs 100
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(output_dir, 'training_log.txt')
    
    def log_metrics(metrics_dict, epoch=None):
        """Log metrics to file and console."""
        with open(log_file, 'a') as f:
            if epoch is not None:
                f.write(f"\nEpoch {epoch}:\n")
            for key, value in metrics_dict.items():
                f.write(f"  {key}: {value}\n")
    
    # Log configuration
    config = {
        'data': data,
        'batch_size': batch_size,
        'lr': lr,
        'epochs': epochs,
        'kl_weight': kl_weight,
        'base_model': base_model
    }
    log_metrics(config)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = H5VAEDataset(data, split='train')
    val_dataset = H5VAEDataset(data, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load VAE
    print(f"Loading VAE from {base_model}...")
    vae = AutoencoderKL.from_pretrained(base_model)
    vae = vae.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Setup validator
    validator = VAEValidator(device)
    
    # Resume if needed
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume)
        vae.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    best_lpips = float('inf')
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        train_metrics = train_epoch(vae, train_loader, optimizer, device, kl_weight)
        
        # Log training metrics
        train_log = {
            'train/loss': train_metrics['loss'],
            'train/recon_loss': train_metrics['recon_loss'],
            'train/kl_loss': train_metrics['kl_loss'],
            'train/lr': optimizer.param_groups[0]['lr']
        }
        log_metrics(train_log, epoch=epoch+1)
        
        # Validate
        if (epoch + 1) % validate_every == 0:
            val_metrics = validate(vae, val_loader, validator, device, output_dir, epoch)
            
            # Analyze latent space
            latent_metrics = save_latent_histogram(vae, val_loader, device, output_dir, epoch)
            
            # Log validation metrics
            val_log = {
                'val/ssim': val_metrics['ssim'],
                'val/psnr': val_metrics['psnr'],
                'val/lpips': val_metrics['lpips'],
                'val/mse': val_metrics['mse'],
                'latent/mean': latent_metrics['latent_mean'],
                'latent/std': latent_metrics['latent_std'],
                'latent/std_mean': latent_metrics['latent_std_mean']
            }
            log_metrics(val_log)
            
            print(f"Validation - SSIM: {val_metrics['ssim']:.4f}, PSNR: {val_metrics['psnr']:.2f}, "
                  f"LPIPS: {val_metrics['lpips']:.4f}, MSE: {val_metrics['mse']:.4f}")
            
            # Save best model
            if val_metrics['lpips'] < best_lpips:
                best_lpips = val_metrics['lpips']
                torch.save({
                    'epoch': epoch,
                    'model': vae.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'metrics': val_metrics
                }, os.path.join(output_dir, 'best_model.pt'))
                print(f"Saved best model (LPIPS: {best_lpips:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model': vae.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt'))
            
            # Save model in diffusers format
            vae.save_pretrained(os.path.join(output_dir, f'vae_epoch_{epoch}'))
        
        scheduler.step()
    
    # Final save
    vae.save_pretrained(os.path.join(output_dir, 'vae_final'))
    
    # Cleanup
    train_dataset.close()
    val_dataset.close()
    
    print(f"\nTraining completed! Best LPIPS: {best_lpips:.4f}")
    print(f"Models saved to {output_dir}")
    print(f"Training log saved to {log_file}")

if __name__ == '__main__':
    train_vae()