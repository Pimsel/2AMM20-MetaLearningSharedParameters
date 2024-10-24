import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor
import wandb
import os
from PIL import Image


# Class to load and prepare images from a directory
class Data(Dataset):
    def __init__(self, folder_path, transform=None):
        self.images_path = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB to handle different image modes
        
        if self.transform:
            image = self.transform(image)
        
        return image


def prepare_image_for_siren(image_tensor):
    '''
    Prepares an image tensor for input into a SIREN model by generating coordinates and target pixel values.

    Parameters:
    ----------
    image_tensor : torch.Tensor
        A 3D tensor representing an image, with shape [C, H, W] where:
        - C is the number of channels (typically 3 for RGB, 1 for grayscale),
        - H is the height of the image,
        - W is the width of the image.

    Returns:
    -------
    coords : torch.Tensor
        A 2D tensor containing the flattened [x, y] coordinates of each pixel, normalized to [-1, 1].
        Shape: [H * W, 2], where H * W is the total number of pixels.
    
    pixel_values : torch.Tensor
        A 2D tensor containing the flattened RGB values of each pixel, to be used as the target output.
        Shape: [H * W, 3], where H * W is the total number of pixels, and 3 represents RGB channels.
    '''

    # Get image dimensions
    B, C, H, W = image_tensor.shape
    
    # Create a mesh grid of pixel coordinates
    y_coords, x_coords = torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W))
    
    # Flatten the coordinate grids and normalize to [-1, 1]
    x_coords = (x_coords.flatten() / (W - 1)) * 2 - 1
    y_coords = (y_coords.flatten() / (H - 1)) * 2 - 1
    
    # Stack to create a list of [x, y] coordinates
    coords = torch.stack([x_coords, y_coords], dim=-1)  # Shape: [H * W, 2]
    
    # Flatten the image pixel values and use as the target
    pixel_values = image_tensor.view(C, H * W).permute(1, 0)  # Shape: [H * W, 3] (RGB values)
    
    return coords, pixel_values


def check_patience(best_loss, loss, wait, patience):
    '''
    Check if training should stop early based on a patience mechanism.

    Parameters:
    ----------
    best_loss : float
        The lowest loss achieved during training so far.

    loss : float
        The current loss in the current epoch or update.

    wait : int
        The counter for the number of consecutive updates without improvement.

    patience : int
        The maximum number of consecutive updates allowed without improvement before stopping.

    Returns:
    -------
    best_loss : float
        Updated best loss if there is an improvement; otherwise, unchanged.

    wait : int
        Updated counter for the number of updates without improvement.

    stop_training : bool
        Flag that indicates whether early stopping is triggered (True) or not (False).
    '''

    if loss < best_loss:
        best_loss = loss  # Update best loss
        wait = 0  # Reset patience counter
    else:
        wait += 1
        if wait >= patience:
            print(f'Stopping early: no improvement after {patience} meta-updates.')
            return best_loss, wait, True  # Early stopping flag
    
    return best_loss, wait, False