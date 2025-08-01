#!/usr/bin/env python3
"""Create H5 datasets for EDM2 training with aspect ratio preservation."""

import os
import numpy as np
import h5py
import cv2
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import click
import torch
import pandas as pd
from typing import List, Union

#----------------------------------------------------------------------------

def load_data(filepath):
    """Load dataframe from CSV file."""
    dataframe = pd.read_csv(filepath)
    return dataframe

def get_image_paths_list(filepath): 
    """Get list of image paths from CSV file."""
    dataframe = load_data(filepath)
    image_paths = dataframe['Path']
    return image_paths

def preprocess(img, desired_size=512):
    """
    This function resizes and zero pads image.
    Preserves aspect ratio by resizing to fit within desired_size
    and padding with black (zero) pixels.
    """
    old_size = img.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    
    # create a new image and paste the resized on it
    # Always create RGB image (even for grayscale input)
    if img.mode == 'L':
        # For grayscale, create RGB background
        new_img = Image.new('RGB', (desired_size, desired_size), (0, 0, 0))
        # Convert grayscale to RGB before pasting
        img_rgb = Image.merge('RGB', (img, img, img))
        new_img.paste(img_rgb, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
    else:
        # For RGB input
        new_img = Image.new('RGB', (desired_size, desired_size), (0, 0, 0))
        new_img.paste(img, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
    return new_img

def img_to_hdf5(image_paths: List[Union[str, Path]], out_filepath: str, resolution=512, vae_encode=False, vae_model=None):
    """
    Convert directory of images into a .h5 file given paths to all 
    images. Preserves aspect ratios with zero padding.
    """
    dset_size = len(image_paths)
    failed_images = []
    
    # Setup VAE if needed
    vae = None
    use_custom_vae = False
    if vae_encode:
        if vae_model and os.path.exists(vae_model):
            # Use custom fine-tuned VAE
            from diffusers import AutoencoderKL
            print(f"Loading custom VAE from {vae_model}")
            vae = AutoencoderKL.from_pretrained(vae_model)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            vae = vae.to(device)
            use_custom_vae = True
        else:
            # Use default VAE through StabilityVAEEncoder
            from training.encoders import StabilityVAEEncoder
            vae_name = vae_model if vae_model else 'stabilityai/sd-vae-ft-mse'
            print(f"Using VAE: {vae_name}")
            vae = StabilityVAEEncoder(vae_name)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            vae.init(device)
            use_custom_vae = False
    
    # Determine dataset shape
    if vae_encode:
        # VAE latents: 8 channels (4 mean + 4 std)
        channels = 8
        dataset_shape = (dset_size, channels, resolution // 8, resolution // 8)
    else:
        # Always use 3 channels for RGB
        channels = 3
        dataset_shape = (dset_size, channels, resolution, resolution)
    
    with h5py.File(out_filepath, 'w') as h5f:
        # Create main dataset
        img_dset = h5f.create_dataset('images', shape=dataset_shape, dtype=np.float32 if vae_encode else np.uint8)
        
        # Store metadata
        h5f.attrs['resolution'] = resolution
        h5f.attrs['channels'] = channels
        h5f.attrs['vae_encoded'] = vae_encode
        
        for idx, path in enumerate(tqdm(image_paths, desc="Processing images")):
            try:
                # Read image using cv2
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"Failed to read {path}")
                
                # Convert to PIL Image object
                if len(img.shape) == 3:
                    # BGR to RGB conversion
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img, mode='RGB')
                else:
                    # Grayscale image
                    img_pil = Image.fromarray(img, mode='L')
                
                # Preprocess with aspect ratio preservation (returns RGB)
                img_pil = preprocess(img_pil, desired_size=resolution)
                
                # Convert back to numpy
                img = np.array(img_pil)
                
                if vae_encode:
                    # img is already RGB from preprocess function
                    # Convert to tensor and normalize
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                    
                    # Encode
                    with torch.no_grad():
                        if use_custom_vae:
                            # For custom VAE
                            latent = vae.encode(img_tensor.to(device) * 2 - 1).latent_dist
                            encoded = torch.cat([latent.mean, latent.std], dim=1)
                            img_dset[idx] = encoded[0].cpu().numpy()
                        else:
                            # Use StabilityVAEEncoder
                            encoded = vae.encode_pixels(img_tensor.to(device))
                            img_dset[idx] = encoded[0].cpu().numpy()
                else:
                    # Store as uint8 RGB image (already RGB from preprocess)
                    # Convert HWC to CHW format
                    img = img.transpose(2, 0, 1)
                    img_dset[idx] = img
                    
            except Exception as e:
                print(f"Failed to process {path}: {e}")
                failed_images.append((path, e))
        
        # Store image paths for reference
        str_dtype = h5py.special_dtype(vtype=str)
        path_dataset = h5f.create_dataset('image_paths', (len(image_paths),), dtype=str_dtype)
        for idx, path in enumerate(image_paths):
            path_dataset[idx] = str(path)
    
    print(f"\nDataset created: {out_filepath}")
    print(f"Total images: {len(image_paths)}")
    print(f"Failed images: {len(failed_images)}")
    if failed_images:
        print("Failed images:", failed_images[:5], "..." if len(failed_images) > 5 else "")
    print(f"Final shape: {dataset_shape}")
    print(f"File size: {os.path.getsize(out_filepath) / 1024**3:.2f} GB")

#----------------------------------------------------------------------------

@click.command()
@click.option('--source', help='Source CSV file with image paths OR directory with images', required=True)
@click.option('--output', help='Output H5 file path', required=True)
@click.option('--resolution', help='Target resolution', type=int, default=512)
@click.option('--vae-encode', help='Encode to VAE latents', is_flag=True)
@click.option('--vae-model', help='VAE model path (default: stabilityai/sd-vae-ft-mse)', type=str)
@click.option('--max-images', help='Maximum images to process', type=int)
def create_h5_dataset(source, output, resolution, vae_encode, vae_model, max_images):
    """Create H5 dataset from CSV file or directory of images.
    
    This script preserves aspect ratios by padding images with black pixels.
    Grayscale images are automatically converted to RGB by repeating channels.
    
    Examples:
    
    \b
    # Create dataset from CSV file
    python create_h5_dataset.py --source image_paths.csv --output dataset_512.h5
    
    \b
    # Create dataset from directory
    python create_h5_dataset.py --source /path/to/images --output dataset_512.h5
    
    \b
    # Create VAE-encoded dataset with custom VAE
    python create_h5_dataset.py --source image_paths.csv --output dataset_512_vae.h5 --vae-encode --vae-model /path/to/vae
    """
    
    # Determine if source is CSV or directory
    source_path = Path(source)
    
    if source_path.suffix.lower() == '.csv':
        # Load paths from CSV
        print(f"Loading image paths from CSV: {source}")
        image_paths = get_image_paths_list(source)
        image_paths = [Path(p) for p in image_paths]
    else:
        # Find all images in directory
        print(f"Searching for images in directory: {source}")
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.dcm', '.tif', '.tiff'}
        image_paths = []
        for ext in image_exts:
            image_paths.extend(source_path.rglob(f'*{ext}'))
            image_paths.extend(source_path.rglob(f'*{ext.upper()}'))
        image_paths = sorted(set(image_paths))  # Remove duplicates and sort
    
    if max_images:
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images")
    
    if len(image_paths) == 0:
        raise ValueError("No images found!")
    
    # Create dataset
    img_to_hdf5(
        image_paths=image_paths,
        out_filepath=output,
        resolution=resolution,
        vae_encode=vae_encode,
        vae_model=vae_model
    )

if __name__ == '__main__':
    create_h5_dataset()