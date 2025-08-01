# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Dataset for text-conditioned diffusion models using precomputed embeddings."""

import os
import numpy as np
import zipfile
import json
import torch
from . import dataset

#----------------------------------------------------------------------------
# Dataset for images with precomputed text embeddings.

class TextEmbeddingDataset(dataset.ImageFolderDataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Resolution to use.
        text_emb_path   = None, # Path to precomputed text embeddings (.npy file).
        **super_kwargs          # Additional arguments for the base class.
    ):
        # Initialize parent class
        super().__init__(path=path, resolution=resolution, **super_kwargs)
        
        # Load precomputed text embeddings
        if text_emb_path is None:
            # Default: look for embeddings in the same directory as the dataset
            if self._type == 'zip':
                text_emb_path = self._path.replace('.zip', '_text_embeddings.npy')
            else:
                text_emb_path = os.path.join(self._path, 'text_embeddings.npy')
        
        if not os.path.exists(text_emb_path):
            raise ValueError(f'Text embeddings file not found: {text_emb_path}')
        
        self._text_embeddings = np.load(text_emb_path).astype(np.float32)
        
        # Validate embeddings shape
        if len(self._text_embeddings) != self._raw_shape[0]:
            raise ValueError(f'Number of text embeddings ({len(self._text_embeddings)}) does not match number of images ({self._raw_shape[0]})')
        
        # Store text embedding dimension
        self._text_dim = self._text_embeddings.shape[1]
        
    def _load_raw_labels(self):
        # Return text embeddings as labels
        return self._text_embeddings
    
    @property
    def label_shape(self):
        return [self._text_dim]
    
    @property
    def label_dim(self):
        return self._text_dim
    
    @property
    def has_labels(self):
        return True

#----------------------------------------------------------------------------
# Dataset metadata for text-conditioned models.

def create_text_dataset_json(
    image_path,         # Path to image directory or zip
    text_emb_path,      # Path to text embeddings npy file
    caption_path=None,  # Optional: Path to captions json for reference
    output_path=None,   # Where to save dataset.json
):
    """Create a dataset.json file for text-conditioned datasets."""
    
    # Load embeddings to get dimensions
    embeddings = np.load(text_emb_path)
    num_samples, text_dim = embeddings.shape
    
    # Create metadata
    metadata = {
        'labels': [f'text_embedding_{i}' for i in range(text_dim)],
        'num_samples': num_samples,
        'text_embedding_dim': text_dim,
        'text_embedding_file': os.path.basename(text_emb_path),
    }
    
    # Add caption info if available
    if caption_path and os.path.exists(caption_path):
        with open(caption_path, 'r') as f:
            captions = json.load(f)
        metadata['caption_file'] = os.path.basename(caption_path)
        metadata['sample_captions'] = captions[:5]  # Store first 5 as examples
    
    # Save metadata
    if output_path is None:
        if image_path.endswith('.zip'):
            output_path = image_path.replace('.zip', '_dataset.json')
        else:
            output_path = os.path.join(image_path, 'dataset.json')
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_path