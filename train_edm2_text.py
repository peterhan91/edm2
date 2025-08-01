# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train text-conditioned diffusion models using the EDM2 formulation."""

import os
import json
import warnings
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')  # False warning printed by PyTorch 1.12

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix

def parse_nimg(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Configuration presets for text-conditioned models

config_presets_text = {
    'edm2-text-xs':  dnnlib.EasyDict(batch=2048, channels=128, channel_mult=[1,2,2,2], lr=0.0120, dropout=0.00),
    'edm2-text-s':   dnnlib.EasyDict(batch=2048, channels=192, channel_mult=[1,2,3,4], lr=0.0100, dropout=0.00),
    'edm2-text-m':   dnnlib.EasyDict(batch=2048, channels=256, channel_mult=[1,2,3,4], lr=0.0090, dropout=0.10),
    'edm2-text-l':   dnnlib.EasyDict(batch=2048, channels=320, channel_mult=[1,2,3,4,5], lr=0.0080, dropout=0.10),
    'edm2-text-xl':  dnnlib.EasyDict(batch=2048, channels=384, channel_mult=[1,2,3,4,5,6], lr=0.0070, dropout=0.10),
    'edm2-text-xxl': dnnlib.EasyDict(batch=2048, channels=512, channel_mult=[1,2,3,4,5,6], lr=0.0065, dropout=0.10),
}

#----------------------------------------------------------------------------

@click.command()
@click.pass_context

# Main options
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                          type=str, required=True)
@click.option('--data',          help='Path to the dataset (ZIP, H5, or directory)', metavar='PATH',       type=str, required=True)
@click.option('--text-emb',      help='Path to text embeddings (.npy file)', metavar='PATH',               type=str, required=True)
@click.option('--preset',        help='Configuration preset', metavar='STR',                               type=click.Choice(list(config_presets_text)), default=None, show_default=True)

# Performance options
@click.option('--gpus',          help='Number of GPUs to use', metavar='INT',                              type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                                   type=click.IntRange(min=1), default=2048, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU for memory', metavar='INT',                type=click.IntRange(min=1), default=None, show_default=True)
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',                   type=bool, default=True, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                            type=bool, default=False, show_default=True)

# I/O-related options
@click.option('--checkpoint',    help='Checkpoint interval', metavar='NIMG',                               type=parse_nimg, default='128Mi', show_default=True)
@click.option('--snapshot',      help='Snapshot interval', metavar='NIMG',                                 type=parse_nimg, default='8Mi', show_default=True)
@click.option('--status',        help='Status interval', metavar='NIMG',                                   type=parse_nimg, default='128Ki', show_default=True)
@click.option('--dry-run',       help='Print training options and exit',                                   is_flag=True)

# Hyperparameters
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                                    type=click.FloatRange(min=0, min_open=True), default=0.13, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                              type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)

# Text embedding options
@click.option('--text-dim',      help='Text embedding dimension (auto-detect if not specified)', metavar='INT', type=int, default=None)
@click.option('--window-size',   help='Window size for sampling images+captions', metavar='INT',           type=click.IntRange(min=1), default=3, show_default=True)

# Preconditioning
@click.option('--sigma-data',    help='Expected data std', metavar='FLOAT',                                type=click.FloatRange(min=0, min_open=True), default=0.5, show_default=True)
@click.option('--sigma-max',     help='Maximum sigma', metavar='FLOAT',                                    type=click.FloatRange(min=0, min_open=True), default=float('inf'), show_default=True)
@click.option('--sigma-min',     help='Minimum sigma', metavar='FLOAT',                                    type=click.FloatRange(min=0, min_open=True), default=0, show_default=True)

def cli(ctx, outdir, data, text_emb, preset, dry_run, **opts):
    """Train text-conditioned diffusion models using the improved EDM2 formulation.
    
    Examples:
    
    \b
    # Train text-conditioned XS model
    torchrun --standalone --nproc_per_node=8 train_edm2_text.py \\
        --outdir=training-runs/text-edm2-xs \\
        --data=datasets/images.zip \\
        --text-emb=datasets/text_embeddings.npy \\
        --preset=edm2-text-xs \\
        --batch-gpu=32
    
    \b
    # To resume training, run the same command again.
    """
    # Setup
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    
    # Initialize config
    c = dnnlib.EasyDict()
    
    # Dataset - auto-detect format
    if data.endswith('.h5'):
        c.dataset_kwargs = dnnlib.EasyDict(
            class_name='training.dataset_h5.H5Dataset',
            h5_path=data,
            text_emb_path=text_emb,
            use_labels=True,  # We always use text embeddings as labels
            xflip=opts['xflip'],
            resolution=None,  # Auto-detect from dataset
            max_size=None,    # Use all available data
        )
    else:
        c.dataset_kwargs = dnnlib.EasyDict(
            class_name='training.dataset_text.TextEmbeddingDataset',
            path=data,
            text_emb_path=text_emb,
            use_labels=True,  # We always use text embeddings as labels
            xflip=opts['xflip'],
            resolution=None,  # Auto-detect from dataset
            max_size=None,    # Use all available data
        )
    
    # Detect dataset properties
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.num_channels
        del dataset_obj  # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')
    
    # Encoder
    if dataset_channels == 3:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StandardRGBEncoder')
    elif dataset_channels == 8:
        c.encoder_kwargs = dnnlib.EasyDict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')
    
    # Auto-detect text embedding dimension if not specified
    if opts['text_dim'] is None:
        import numpy as np
        embeddings = np.load(text_emb, mmap_mode='r')
        opts['text_dim'] = embeddings.shape[1]
        if dist.get_rank() == 0:
            print(f'Auto-detected text embedding dimension: {opts["text_dim"]}')
    
    # Network architecture
    c.network_kwargs = dnnlib.EasyDict(
        class_name='training.networks_edm2_text.TextConditionedEDM2Precond',
        text_dim=opts['text_dim'],
        use_fp16=opts['fp16'],
        sigma_data=opts['sigma_data'],
        sigma_max=opts['sigma_max'],
        sigma_min=opts['sigma_min'],
    )
    
    # Apply preset
    if preset is not None:
        if preset not in config_presets_text:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        preset_cfg = config_presets_text[preset]
        c.network_kwargs.update(
            model_channels=preset_cfg.channels,
            channel_mult=preset_cfg.channel_mult,
            dropout=preset_cfg.get('dropout', opts['dropout'])
        )
        # Update other parameters from preset
        if opts['lr'] == 0.13:  # Use preset value if default
            opts['lr'] = preset_cfg.lr
        if opts['batch'] == 2048:  # Use preset value if default
            opts['batch'] = preset_cfg.batch
    
    # Training options
    c.batch_size = opts['batch']
    c.batch_gpu = opts['batch_gpu'] or (c.batch_size // (opts['gpus'] * dist.get_world_size()))
    c.total_nimg = 2048 << 20  # Train for 2B images (default)
    c.snapshot_nimg = opts['snapshot']
    c.checkpoint_nimg = opts['checkpoint']
    c.status_nimg = opts['status']
    
    # Learning rate schedule
    c.lr_kwargs = dnnlib.EasyDict(
        func_name='training.training_loop.learning_rate_schedule', 
        ref_lr=opts['lr'], 
        ref_batches=70000  # Default from EDM2
    )
    
    # No augmentation in base EDM2 implementation
    
    # Loss function
    c.loss_kwargs = dnnlib.EasyDict(
        class_name='training.training_loop.EDM2Loss',
        sigma_data=opts['sigma_data'],
        P_mean=-1.2,
        P_std=1.2,
    )
    
    # Performance-related options
    c.loss_scaling = 1
    c.cudnn_benchmark = True
    
    # Random seed
    c.seed = 0
    
    # Print options
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:    {outdir}')
    dist.print0(f'Dataset path:        {c.dataset_kwargs.path}')
    dist.print0(f'Text embeddings:     {c.dataset_kwargs.text_emb_path}')
    dist.print0(f'Number of GPUs:      {dist.get_world_size()}')
    dist.print0(f'Batch size:          {c.batch_size}')
    dist.print0(f'Mixed-precision:     {c.network_kwargs.use_fp16}')
    dist.print0()
    
    if dry_run:
        dist.print0('Dry run; exiting.')
        return
    
    # Launch training
    training_loop.training_loop(run_dir=outdir, **c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cli()