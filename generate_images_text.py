# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate images using text-conditioned diffusion models."""

import os
import re
import pickle
import numpy as np
import torch
import PIL.Image
import click
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc
from training import dataset_text

#----------------------------------------------------------------------------
# Load text embeddings for generation

def load_text_embeddings(embedding_source, num_samples=None, device='cuda', caption_column='impression'):
    """Load precomputed text embeddings from various sources.
    
    Args:
        embedding_source: Can be:
            - Path to .npy file with precomputed embeddings
            - Path to .json file with captions (will compute on-the-fly)
            - Path to .csv file with captions (will compute on-the-fly)
            - List of text strings (will compute on-the-fly)
        num_samples: Number of embeddings to load (None = all)
        device: Device to load embeddings to
        caption_column: Column name for CSV files
    
    Returns:
        torch.Tensor of shape [num_samples, embedding_dim]
    """
    if isinstance(embedding_source, str) and embedding_source.endswith('.npy'):
        # Load precomputed embeddings
        embeddings = np.load(embedding_source).astype(np.float32)
        if num_samples is not None:
            embeddings = embeddings[:num_samples]
        return torch.from_numpy(embeddings).to(device)
    
    elif isinstance(embedding_source, str) and embedding_source.endswith('.json'):
        # Load captions and compute embeddings on-the-fly
        import json
        with open(embedding_source, 'r') as f:
            captions = json.load(f)
        if isinstance(captions, dict):
            captions = list(captions.values())
        if num_samples is not None:
            captions = captions[:num_samples]
        return compute_text_embeddings(captions, device=device)
    
    elif isinstance(embedding_source, str) and embedding_source.endswith('.csv'):
        # Load captions from CSV and compute embeddings on-the-fly
        import pandas as pd
        df = pd.read_csv(embedding_source)
        captions = df[caption_column].fillna(" ").tolist()
        if num_samples is not None:
            captions = captions[:num_samples]
        return compute_text_embeddings(captions, device=device)
    
    elif isinstance(embedding_source, list):
        # Compute embeddings from list of strings
        if num_samples is not None:
            embedding_source = embedding_source[:num_samples]
        return compute_text_embeddings(embedding_source, device=device)
    
    else:
        raise ValueError(f'Unknown embedding source type: {type(embedding_source)}')

def compute_text_embeddings(texts, model='Salesforce/SFR-Embedding-Mistral', device='cuda'):
    """Compute text embeddings on-the-fly using SentenceTransformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError('Please install sentence-transformers: pip install sentence-transformers')
    
    encoder = SentenceTransformer(model, device=device)
    embeddings = encoder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return embeddings

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Modified EDM sampler for text conditioning

def edm_sampler_text(
    net, noise, text_embeddings=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Adjust noise levels based on what's supported by the network
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Denoiser with guidance support
    def denoise(x, t):
        Dx = net(x, t, labels=text_embeddings).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels=text_embeddings).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # Time step discretization following EDM paper
    t_steps = (sigma_max ** (1 / rho) + torch.arange(num_steps, dtype=dtype, device=noise.device) / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

    # Main sampling loop
    x = noise * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x

        # Increase noise temporarily
        gamma = min(S_churn / num_steps, 2 ** 0.5 - 1) if S_min <= t_cur <= S_max else 0
        t_hat = (t_cur ** 2 + gamma ** 2 * t_cur ** 2).sqrt()
        if gamma > 0:
            eps = randn_like(x) * S_noise
            x = x + eps * (t_hat ** 2 - t_cur ** 2).sqrt()

        # Euler step
        d_cur = (x - denoise(x, t_hat)) / t_hat
        x = x + (t_next - t_hat) * d_cur

        # Apply 2nd order correction
        if i < num_steps - 1:
            d_prime = (x - denoise(x, t_next)) / t_next
            x = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x

#----------------------------------------------------------------------------
# Main generation function for text-conditioned models

@torch.no_grad()
def generate_images_text(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    text_embeddings,                            # Text embeddings or source to load from.
    gnet                = None,                 # Guiding network. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = None,                 # List of random seeds. None = use indices as seeds.
    batch_gpu           = 1,                    # Maximum number of samples per GPU.
    sampler_fn          = edm_sampler_text,     # Sampler function.
    sampler_kwargs      = dict(),               # Keyword arguments for the sampler.
    device              = torch.device('cuda'), # Device to use.
    dist_kwargs         = dict(),               # Keyword arguments for torch_utils.distributed.
    caption_file        = None,                 # Optional caption file for saving alongside images.
):
    # Rank 0 goes first
    if dist.get_rank() != 0:
        torch.distributed.barrier()
        
    # Rank 0 prints status
    if dist.get_rank() == 0:
        print(f'Generating images...')
        if outdir is not None:
            print(f'Saving images to {outdir}...')
            os.makedirs(outdir, exist_ok=True)

    # Load networks
    if isinstance(net, str):
        if dist.get_rank() == 0:
            print(f'Loading network from {net}...')
        with dnnlib.util.open_url(net, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
            net = data['ema'].to(device)
            if encoder is None and 'encoder' in data:
                encoder = data['encoder'].to(device)

    if isinstance(gnet, str):
        if dist.get_rank() == 0:
            print(f'Loading guiding network from {gnet}...')
        with dnnlib.util.open_url(gnet, verbose=(dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder if needed
    if encoder is None:
        from training import encoders
        if hasattr(net, 'img_channels') and net.img_channels == 3:
            encoder = encoders.StandardRGBEncoder()
        else:
            encoder = encoders.StabilityVAEEncoder()
    
    # Initialize encoder
    if dist.get_rank() == 0:
        print(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    
    # Other ranks follow
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Load text embeddings
    if dist.get_rank() == 0:
        print(f'Loading text embeddings...')
    text_embeddings = load_text_embeddings(text_embeddings, device=device)
    num_samples = len(text_embeddings)
    
    if dist.get_rank() == 0:
        print(f'Loaded {num_samples} text embeddings of dimension {text_embeddings.shape[1]}')

    # Generate seeds if not provided
    if seeds is None:
        seeds = list(range(num_samples))
    elif len(seeds) < num_samples:
        seeds = seeds + list(range(len(seeds), num_samples))
    seeds = seeds[:num_samples]

    # Load captions if provided
    captions = None
    if caption_file is not None:
        import json
        with open(caption_file, 'r') as f:
            caption_data = json.load(f)
        if isinstance(caption_data, list):
            captions = caption_data[:num_samples]
        elif isinstance(caption_data, dict):
            captions = list(caption_data.values())[:num_samples]

    # Generate images in batches
    all_indices = list(range(num_samples))
    rank_indices = all_indices[dist.get_rank()::dist.get_world_size()]
    
    for batch_idx, start_idx in enumerate(range(0, len(rank_indices), batch_gpu)):
        end_idx = min(start_idx + batch_gpu, len(rank_indices))
        batch_indices = rank_indices[start_idx:end_idx]
        batch_size = len(batch_indices)
        batch_seeds = [seeds[idx] for idx in batch_indices]
        
        if dist.get_rank() == 0 and batch_idx % 10 == 0:
            print(f'Generating batch {batch_idx+1}/{(len(rank_indices) + batch_gpu - 1) // batch_gpu}...')
        
        # Get text embeddings for this batch
        batch_text_emb = text_embeddings[batch_indices]
        
        # Generate noise using seeds
        rnd = StackedRandomGenerator(device, batch_seeds)
        noise = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        
        # Generate latents
        latents = sampler_fn(net=net, noise=noise, text_embeddings=batch_text_emb, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs)
        
        # Decode to images
        images = encoder.decode(latents)
        
        # Save images
        if outdir is not None:
            for i, (idx, seed) in enumerate(zip(batch_indices, batch_seeds)):
                if subdirs:
                    subdir = f'{seed//1000*1000:06d}'
                    os.makedirs(os.path.join(outdir, subdir), exist_ok=True)
                    image_path = os.path.join(outdir, subdir, f'{seed:06d}.png')
                else:
                    image_path = os.path.join(outdir, f'{seed:06d}.png')
                
                image_np = images[i].permute(1, 2, 0).cpu().numpy()
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
                
                # Save caption if available
                if captions is not None and idx < len(captions):
                    caption_path = image_path.replace('.png', '.txt')
                    with open(caption_path, 'w') as f:
                        f.write(captions[idx])

    # Synchronize and print final status
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        print(f'Generated {num_samples} images.')

#----------------------------------------------------------------------------
# CLI interface

@click.command()
@click.pass_context
@click.option('--net', help='Network pickle filename or URL', metavar='PATH|URL', required=True)
@click.option('--embeddings', help='Text embeddings (.npy file, .json captions, or .csv)', metavar='PATH', required=True)
@click.option('--gnet', help='Guiding network pickle filename', metavar='PATH|URL', default=None)
@click.option('--outdir', help='Where to save the output images', metavar='DIR', required=True)
@click.option('--subdirs', help='Create subdirectory for every 1000 images', is_flag=True)
@click.option('--batch-gpu', help='Maximum images per GPU', metavar='INT', type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--num-steps', help='Number of sampling steps', metavar='INT', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma-min', help='Lowest noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma-max', help='Highest noise level', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho', help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance', help='Guidance strength [default: 1; no guidance]', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--S-churn', 'S_churn', help='Stochasticity strength', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S-min', 'S_min', help='Stoch. min noise level', metavar='FLOAT', type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S-max', 'S_max', help='Stoch. max noise level', metavar='FLOAT', type=click.FloatRange(min=0), default=float('inf'), show_default=True)
@click.option('--S-noise', 'S_noise', help='Stoch. noise inflation', metavar='FLOAT', type=float, default=1, show_default=True)
@click.option('--captions', help='Caption file for saving alongside images', metavar='PATH', default=None)
@click.option('--seeds', help='Random seeds (e.g. 0-99)', metavar='LIST', type=str, default='0-15')
def cli(ctx, **opts):
    """Generate images using text-conditioned diffusion models.
    
    Examples:
    
    \b
    # Generate images using precomputed embeddings
    python generate_images_text.py \\
        --net=models/text_edm2_model.pkl \\
        --embeddings=text_embeddings.npy \\
        --outdir=out
    
    \b
    # Generate with autoguidance
    python generate_images_text.py \\
        --net=models/text_edm2_large.pkl \\
        --gnet=models/text_edm2_small.pkl \\
        --embeddings=text_embeddings.npy \\
        --guidance=2.0 \\
        --outdir=out
    """
    opts = dnnlib.EasyDict(opts)
    
    # Set up guidance
    if opts.guidance != 1 and opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')
    
    # Parse seeds
    seeds = parse_int_list(opts.seeds)
    
    # Setup
    torch.multiprocessing.set_start_method('spawn')
    dist_kwargs = dict(device_id=0)
    
    # Generate
    generate_images_text(
        net=opts.net,
        text_embeddings=opts.embeddings,
        gnet=opts.gnet,
        outdir=opts.outdir,
        subdirs=opts.subdirs,
        seeds=seeds,
        batch_gpu=opts.batch_gpu,
        sampler_kwargs=dict(
            num_steps=opts.num_steps,
            sigma_min=opts.sigma_min,
            sigma_max=opts.sigma_max,
            rho=opts.rho,
            guidance=opts.guidance,
            S_churn=opts.S_churn,
            S_min=opts.S_min,
            S_max=opts.S_max,
            S_noise=opts.S_noise,
        ),
        caption_file=opts.captions,
        dist_kwargs=dist_kwargs,
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cli()