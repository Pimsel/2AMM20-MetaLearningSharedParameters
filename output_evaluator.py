import os
import re
from PIL import Image
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


# Directories with the original and reconstructed images
original_dir = r"saved_input"   # Path to original images
reconstruct_dir = r"saved_output"   # Path to reconstructed images

# Get lists of all files in each directory
original_images = sorted(os.listdir(original_dir))
reconstructed_images = sorted(os.listdir(reconstruct_dir))


# Remove the '_input'/'_output' suffix to match files
reconstructed_images_matching = [
    re.sub(r'_output\.jpg$', '_input.jpg', img) for img in reconstructed_images
]

# Now pair original images with reconstructed images
image_pairs = [
    (os.path.join(original_dir, orig), os.path.join(reconstruct_dir, rec))
    for orig, rec in zip(original_images, reconstructed_images)
    if orig in reconstructed_images_matching
]

psnr_list = []
ssim_list = []
# Iterate over the pairs
for original_img_path, reconstructed_img_path in image_pairs:
    
    original_img = np.array(Image.open(original_img_path))
    reconstructed_img = np.array(Image.open(reconstructed_img_path))

    # Check if the shapes of the images are the same
    if original_img.shape != reconstructed_img.shape:
        raise ValueError(f"Image sizes do not match: {original_img.shape} vs {reconstructed_img.shape}")

    # Compute PSNR (higher is better)
    psnr_value = compare_psnr(original_img, reconstructed_img)
    psnr_list.append(psnr_value)
    
    # Compute SSIM (closer to 1 is better)
    ssim_value = compare_ssim(original_img, reconstructed_img, channel_axis=-1)
    ssim_list.append(ssim_value)

print(f'Average PSNR: {np.mean(psnr_list):.2f}        Median PSNR: {np.median(psnr_list):.2f}        Max PSNR: {np.max(psnr_list):.2f} img{psnr_list.index(np.max(psnr_list))+1}')
print(f'Average SSIM: {np.mean(ssim_list):.4f}        Median SSIM: {np.median(ssim_list):.4f}        Max SSIM: {np.max(ssim_list):.4f} img{ssim_list.index(np.max(ssim_list))+1}')