# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Text-conditioned diffusion model architecture extending EDM2."""

import numpy as np
import torch
from torch_utils import persistence
from . import networks_edm2

#----------------------------------------------------------------------------
# Text-conditioned U-Net extending the EDM2 architecture.

class TextConditionedUNet(networks_edm2.UNet):
    def __init__(self,
        img_resolution,                     # Image resolution.
        img_channels,                       # Image channels.
        text_dim,                           # Text embedding dimensionality or dict {'impression': 768, 'age': 32, 'sex': 32}.
        model_channels      = 192,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,3,4],    # Per-resolution multipliers for the number of channels.
        channel_mult_noise  = None,         # Multiplier for noise embedding dimensionality. None = use the last value of channel_mult.
        channel_mult_emb    = None,         # Multiplier for final embedding dimensionality. None = use the max value of channel_mult.
        num_blocks          = 3,            # Number of residual blocks per resolution.
        attn_resolutions    = [16,8],       # List of resolutions with self-attention.
        label_balance       = 0.5,          # Balance between noise embedding and text embedding, 0 = noise only, 1 = text only.
        concat_balance      = 0.5,          # Balance between skip connections and main path, 0 = skip only, 1 = main only.
        **block_kwargs,                     # Arguments for Block.
    ):
        # Initialize parent class with label_dim=0 to skip label embedding creation
        super().__init__(
            img_resolution=img_resolution,
            img_channels=img_channels,
            label_dim=0,  # We'll handle text embeddings separately
            model_channels=model_channels,
            channel_mult=channel_mult,
            channel_mult_noise=channel_mult_noise,
            channel_mult_emb=channel_mult_emb,
            num_blocks=num_blocks,
            attn_resolutions=attn_resolutions,
            label_balance=label_balance,
            concat_balance=concat_balance,
            **block_kwargs,
        )
        
        # Create text embedding projection(s)
        cemb = model_channels * channel_mult_emb if channel_mult_emb is not None else max([model_channels * m for m in channel_mult])
        
        if isinstance(text_dim, dict):
            # Multiple text embeddings with cross-attention
            self.text_dim = text_dim
            self.multi_text = True
            
            # Efficient cross-attention for combining multiple embeddings
            attn_dim = min(512, cemb)  # Reduce dimension for efficiency
            self.text_proj_k = networks_edm2.MPConv(4096, attn_dim, kernel=[])
            self.text_proj_v = networks_edm2.MPConv(4096, attn_dim, kernel=[])
            self.noise_proj_q = networks_edm2.MPConv(cemb, attn_dim, kernel=[])
            self.text_out = networks_edm2.MPConv(attn_dim, cemb, kernel=[])
            self.attn_scale = attn_dim ** -0.5
        else:
            # Single text embedding (backward compatibility)
            self.text_dim = text_dim
            self.emb_text = networks_edm2.MPConv(text_dim, cemb, kernel=[])
            self.multi_text = False
        
        self.label_balance = label_balance

    def forward(self, x, noise_labels, class_labels):
        # Noise embedding
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        
        # Handle text embeddings
        if self.multi_text and isinstance(class_labels, dict):
            # Cross-attention to combine multiple text embeddings
            # Stack embeddings: [batch, num_texts, 4096]
            text_keys = sorted(class_labels.keys())  # Ensure consistent ordering
            text_stack = torch.stack([class_labels[k] for k in text_keys], dim=1)
            batch_size, num_texts, _ = text_stack.shape
            
            # Project to K, V: [batch, num_texts, attn_dim]
            k = self.text_proj_k(text_stack.reshape(-1, 4096)).reshape(batch_size, num_texts, -1)
            v = self.text_proj_v(text_stack.reshape(-1, 4096)).reshape(batch_size, num_texts, -1)
            
            # Query from noise embedding: [batch, 1, attn_dim]
            q = self.noise_proj_q(emb).unsqueeze(1)
            
            # Attention scores: [batch, 1, num_texts]
            scores = torch.bmm(q, k.transpose(1, 2)) * self.attn_scale
            weights = torch.softmax(scores, dim=-1)
            
            # Weighted combination: [batch, attn_dim]
            attended = torch.bmm(weights, v).squeeze(1)
            text_emb = self.text_out(attended)
        else:
            # Single text embedding (backward compatibility)
            text_embeddings = class_labels
            text_emb = self.emb_text(text_embeddings)
        
        emb = networks_edm2.mp_sum(emb, text_emb, t=self.label_balance)
        emb = networks_edm2.mp_silu(emb)

        # Encoder
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder
        for name, block in self.dec.items():
            if 'block' in name:
                x = networks_edm2.mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)

        return x

#----------------------------------------------------------------------------
# Preconditioning wrapper for text-conditioned diffusion models.

@persistence.persistent_class
class TextConditionedEDM2Precond(torch.nn.Module):
    def __init__(self,
        img_resolution,         # Image resolution.
        img_channels,           # Image channels.
        text_dim,               # Text embedding dimensionality or dict.
        use_fp16        = True, # Run the model at FP16 precision?
        sigma_data      = 0.5,  # Expected standard deviation of the training data.
        logvar_channels = 128,  # Intermediate dimensionality for uncertainty estimation.
        **unet_kwargs,          # Keyword arguments for TextConditionedUNet.
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.text_dim = text_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.multi_text = isinstance(text_dim, dict)
        self.unet = TextConditionedUNet(
            img_resolution=img_resolution, 
            img_channels=img_channels, 
            text_dim=text_dim, 
            **unet_kwargs
        )
        self.logvar_fourier = networks_edm2.MPFourier(logvar_channels)
        self.logvar_linear = networks_edm2.MPConv(logvar_channels, 1, kernel=[])

    def forward(self, x, sigma, labels=None, force_fp32=False, return_logvar=False, **unet_kwargs):
        # For compatibility with training loop, accept 'labels' parameter name
        text_embeddings = labels
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        
        # Handle text embeddings based on type
        if self.multi_text:
            # Dict of embeddings
            text_embeddings = {
                name: emb.to(torch.float32) 
                for name, emb in text_embeddings.items()
            }
        else:
            # Single embedding
            text_embeddings = text_embeddings.to(torch.float32).reshape(-1, self.text_dim)
        
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        # Preconditioning weights (same as original EDM2)
        c_skip = (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)
        c_out = (sigma * self.sigma_data) / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4

        # Run the model
        x_in = (c_in * x).to(dtype)
        F_x = self.unet(x_in, c_noise, text_embeddings, **unet_kwargs)
        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        # Estimate uncertainty if requested
        if return_logvar:
            logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return D_x, logvar
        return D_x

    # For compatibility with sampling code that expects label_dim
    @property
    def label_dim(self):
        return 0  # We use text embeddings, not class labels
    
    # For compatibility with sampling code that expects sigma_min/max
    @property
    def sigma_min(self):
        return 0
    
    @property
    def sigma_max(self):
        return float('inf')