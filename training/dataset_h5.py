# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""H5 dataset support for EDM2."""

import os
import numpy as np
import h5py
import torch
from training.dataset import Dataset

#----------------------------------------------------------------------------

class H5Dataset(Dataset):
    def __init__(self,
        h5_path,                    # Path to H5 file
        text_emb_path=None,         # Path to text embeddings (.npy or dict of paths)
        resolution=None,            # Ensure specific resolution
        use_labels=True,            # Enable text embeddings
        max_size=None,              # Artificially limit dataset size
        xflip=False,                # Enable horizontal flips
        cache=False,                # Cache images in RAM
    ):
        self._h5_path = h5_path
        self._text_emb_path = text_emb_path
        self._resolution = resolution
        
        # Open H5 file
        self._h5_file = h5py.File(h5_path, 'r')
        self._images = self._h5_file['images']
        
        # Validate dataset
        assert len(self._images.shape) == 4, f"Expected 4D array (N,C,H,W), got {self._images.shape}"
        raw_shape = list(self._images.shape)
        
        # Check resolution
        if resolution is not None:
            assert raw_shape[2] == resolution and raw_shape[3] == resolution, \
                f"Resolution mismatch: expected {resolution}, got {raw_shape[2]}x{raw_shape[3]}"
        
        # Load text embeddings if provided
        self._text_embeddings = None
        self._multi_text = False
        if text_emb_path and use_labels:
            if isinstance(text_emb_path, dict):
                # Multiple text embeddings
                self._text_embeddings = {}
                self._multi_text = True
                for name, path in text_emb_path.items():
                    emb = np.load(path).astype(np.float32)
                    assert len(emb) == raw_shape[0], \
                        f"Text embedding count mismatch for {name}: {len(emb)} vs {raw_shape[0]}"
                    self._text_embeddings[name] = emb
            else:
                # Single text embedding
                self._text_embeddings = np.load(text_emb_path).astype(np.float32)
                assert len(self._text_embeddings) == raw_shape[0], \
                    f"Text embedding count mismatch: {len(self._text_embeddings)} vs {raw_shape[0]}"
        
        # Dataset name from filename
        name = os.path.splitext(os.path.basename(h5_path))[0]
        
        # Initialize parent class
        super().__init__(
            name=name,
            raw_shape=raw_shape,
            use_labels=use_labels,
            max_size=max_size,
            xflip=xflip,
            cache=cache
        )
        
        # Store xflip state for proper indexing
        self._xflip_idx = self._xflip

    def _load_raw_image(self, raw_idx):
        if self._cache and raw_idx in self._cached_images:
            return self._cached_images[raw_idx]
        
        # H5 datasets support direct indexing
        image = np.array(self._images[raw_idx])
        
        # Ensure correct dtype - ZIP pipeline returns uint8 for RGB images
        if image.dtype != np.uint8 and not self._is_vae_encoded():
            # Convert float images to uint8 range [0, 255]
            if image.dtype in [np.float32, np.float64]:
                if image.min() >= -1 and image.max() <= 1:
                    # Assume [-1, 1] range, convert to [0, 255]
                    image = ((image + 1) * 127.5).clip(0, 255).astype(np.uint8)
                elif image.min() >= 0 and image.max() <= 1:
                    # Assume [0, 1] range, convert to [0, 255]
                    image = (image * 255).clip(0, 255).astype(np.uint8)
                else:
                    # Assume already in [0, 255] range
                    image = image.clip(0, 255).astype(np.uint8)
        
        if self._cache:
            self._cached_images[raw_idx] = image
        
        return image
    
    def _is_vae_encoded(self):
        """Check if this dataset contains VAE-encoded latents."""
        # VAE latents are float32 and have 8 channels (4 mean + 4 std)
        return self._raw_shape[1] == 8 and self._images.dtype == np.float32

    def _load_raw_labels(self):
        if self._text_embeddings is not None and not self._multi_text:
            # Single embedding case
            return self._text_embeddings
        # For multi-text, return dummy labels (actual dict returned in get_label)
        return np.zeros([self._raw_shape[0], 0], dtype=np.float32)
    
    def get_label(self, idx):
        """Override to handle dict of embeddings."""
        if self._multi_text and self._text_embeddings is not None:
            # Return dict of embeddings for this index
            raw_idx = self._raw_idx[idx]
            return {name: emb[raw_idx].copy() for name, emb in self._text_embeddings.items()}
        else:
            # Use parent class implementation for single embedding
            return super().get_label(idx)

    def close(self):
        if hasattr(self, '_h5_file'):
            self._h5_file.close()

    @property
    def has_labels(self):
        return self._text_embeddings is not None