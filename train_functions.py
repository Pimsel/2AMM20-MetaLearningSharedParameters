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
    C, H, W = image_tensor.shape
    
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


def get_N_shared_layers(model, N):
    '''
    Split a model's parameters into shared and unique layers.

    Parameters:
    ----------
    model : torch.nn.Module
        The model whose parameters will be divided into shared and unique layers.

    N : int
        The number of initial layers to be considered as shared layers.

    Returns:
    -------
    shared_layers : list
        A list containing the parameters of the first `N` shared layers.

    unique_layers : list
        A list containing the parameters of the remaining layers after the shared layers.
    '''

    shared_layers = []
    unique_layers = []
    for i, (name, param) in enumerate(model.named_parameters()):
        if i < N:
            shared_layers.append(param)
        else:
            unique_layers.append(param)
    return shared_layers, unique_layers


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


# Training loop
def train_model(model, device, dataset_path, K, num_epochs, inner_loss_fn, outer_loss_fn, 
                inner_learning_rate, meta_learning_rate, batch_size, patience, 
                pretrained=False, shared_params=None, original_unique_params=None):
    '''
    Trains a neural network model with meta-learning by iteratively updating a shared set of parameters 
    across multiple tasks. It also validates the model's performance using a validation dataset.

    Parameters:
    -----------
    model : torch.nn.Module
        The neural network model to be trained, which contains shared and unique blocks of parameters.

    device : torch.device
        The device on which the model and data will be loaded, typically a CPU or GPU.

    dataset_path : str
        The file path to the directory containing the images for training and validation.

    K : int
        The number of inner-loop optimization steps performed for updating the task-specific parameters 
        (unique parameters) before updating the shared parameters.

    num_epochs : int
        The number of epochs (full passes through the training dataset) for training the model.

    inner_loss_fn : callable
        The loss function used for optimizing the task-specific parameters (unique parameters) in the inner loop.

    outer_loss_fn : callable
        The loss function used for optimizing the shared parameters in the outer loop based on the updated 
        task-specific parameters.

    inner_learning_rate : float
        The learning rate used by the inner optimizer to update the task-specific parameters during the inner-loop training.

    meta_learning_rate : float
        The learning rate used by the meta-optimizer to update the shared parameters in the outer loop during meta-updates.

    batch_size : int
        The number of images in each batch of data that is fed into the model for training in each epoch.

    patience : int
        The number of epochs to wait before early stopping if the validation loss does not improve.

    pretrained : bool
        Indicates whether to load a pre-trained model or train the model from scratch.

    shared_params : iterable, optional
        A collection of model parameters for the shared layers, passed when loading a pre-trained model. 
        If None, the model will initialize its own shared parameters during training.

    original_unique_params : iterable, optional
        A list of model parameters for the unique layers, specified when loading a pre-trained model where 
        specific unique parameters are pre-defined. If None, the model will initialize its own unique parameters.
    
    Returns:
    --------
    None
    '''

    # Prepare data
    train_dataset = Data(dataset_path, transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define shared and unique parameters if new model is used
    if not pretrained:
        shared_params = model.shared_block.parameters()
        original_unique_params = list(model.unique_block.parameters()) + list(model.output_layer.parameters())
    
    # Initialize outer loop optimizer
    meta_optimizer = optim.Adam(shared_params, lr=meta_learning_rate)
    
    # Initialize patience variables
    best_loss = 1.0
    wait = 0

    for epoch in range(num_epochs):
        total_outer_loss = 0
        for i1, image_batch in enumerate(train_dataloader):
            batch_coords, batch_targets = [], []
            for image in image_batch:
                coords, targets = prepare_image_for_siren(image)
                batch_coords.append(coords)
                batch_targets.append(targets)
            
            batch_coords = torch.stack(batch_coords).to(device)
            batch_targets = torch.stack(batch_targets).to(device)

            # Create a copy of the unique parameters for the inner loop
            unique_params = [param.to(device).requires_grad_(True) for param in original_unique_params]

            # Inner loop
            for k in range(K):
                # Create a temporary optimizer for the inner loop (2nd order MAML)
                inner_optimizer = optim.Adam(unique_params, lr=inner_learning_rate)

                output = model(batch_coords)
                inner_loss = inner_loss_fn(output, batch_targets)
        
                # Backward pass and update over only unique parameters
                inner_optimizer.zero_grad()
                inner_loss.backward(retain_graph=True)  # Retain graph for meta-update
                inner_optimizer.step()
                    
                # Log inner loss
                wandb.log({f'Inner loop loss': inner_loss.item()})
                
                print(f'\rProcessed batch {str(i1+1).ljust(2)}/{len(train_dataloader)}. Epoch {str(epoch+1).ljust(4)}/{num_epochs}', end='  ', flush=True)
            
            # Meta-update
            output = model(batch_coords)
            outer_loss = outer_loss_fn(output, batch_targets)
            total_outer_loss += outer_loss
            
            meta_optimizer.zero_grad()
            outer_loss.backward()
            meta_optimizer.step()

            wandb.log({f'Outer loop batch loss': outer_loss.item()})

        avg_outer_loss = total_outer_loss / len(train_dataloader)
        wandb.log({f'Outer loop average loss': avg_outer_loss.item()})

        # Check patience
        best_loss, wait, stop_training = check_patience(best_loss, avg_outer_loss.item(), wait, patience)
        if stop_training:
            print(f'Training stopped early at epoch {epoch+1}.')
            break

    wandb.finish()