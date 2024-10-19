import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import copy
import wandb
import os
from PIL import Image


# Create class to load images from a directory
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


# Function to split data into mini-batches
def batch_data(coords, targets, batch_size):
    num_samples = coords.shape[0]
    for i in range(0, num_samples, batch_size):
        yield coords[i:i + batch_size], targets[i:i + batch_size]


# Function to prepare image as input to the SIREN (return flattened coords and target pixels)
def prepare_image_for_siren(image_tensor):
    '''
    
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
    
    return coords, pixel_values, H, W


def check_patience(best_loss, loss, wait, patience):
    '''
    
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
def train_model(model, device, train_dataset_path, K, num_epochs, inner_loss_fn, outer_loss_fn, inner_learning_rate, meta_learning_rate, general_batch_size, mini_batch_size, patience):
    '''
    
    '''
    # Prepare data
    train_dataset_tensors = Data(train_dataset_path, transform=ToTensor())
    general_dataloader = DataLoader(train_dataset_tensors, batch_size=general_batch_size, shuffle=True)

    shared_params = model.shared_block.parameters()
    original_unique_params = list(model.unique_block.parameters()) + list(model.output_layer.parameters())
    ###original_unique_params = [param.clone().detach() for param in unique_params]
    
    
    meta_optimizer = optim.AdamW(shared_params, lr=meta_learning_rate)
    ###inner_optimizer = optim.NAdam(unique_params, lr=inner_learning_rate)
    
    best_loss = 1.0
    wait = 0

    for epoch in range(num_epochs):
        for i1, image_batch in enumerate(general_dataloader):
            total_outer_loss = 0

            for i2, image in enumerate(image_batch):
                # Reset the unique parameters to their original values at the start of each inner loop
                ###for param, original_param in zip(unique_params, original_unique_params):
                ###    param.data.copy_(original_param.data)
                for param in unique_params: #
                    param.requires_grad = True #
                
                coords, targets, height, width = prepare_image_for_siren(image)
                coords, targets = coords.to(device), targets.to(device)

                # Create a copy of the unique parameters for the inner loop #
                unique_params = [param.clone().detach().to(device).requires_grad_(True) for param in original_unique_params] #

                # Inner loop
                for k in range(K):
                    total_inner_loss = 0
                    num_mini_batches = 0

                    # Create a temporary optimizer for the inner loop (2nd order MAML) #
                    inner_optimizer = optim.NAdam(unique_params, lr=inner_learning_rate) #

                    for batch_coords, batch_targets in batch_data(coords, targets, mini_batch_size):
                        # Forward pass (batched)
                        output = model(batch_coords)
                        inner_loss = inner_loss_fn(output, batch_targets)

                        # Accumulate loss for the batch
                        total_inner_loss += inner_loss
                        num_mini_batches += 1
                    
                    # Compute the average loss over all batches for this K-iteration
                    avg_inner_loss = total_inner_loss / num_mini_batches
        
                    # Backward pass and update only after averaging the loss over all batches
                    inner_optimizer.zero_grad()
                    avg_inner_loss.backward(retain_graph=True)  # Retain graph for meta-update
                    inner_optimizer.step()
                    
                    # Log average inner loss
                    wandb.log({f'inner loop avg loss epoch {epoch+1}': avg_inner_loss.item()})
                    print(f'\rFinished inner loop iteration {str(k).ljust(2)} for image {str(i2+1).ljust(4)}/{len(image_batch)} of batch {str(i1+1).ljust(2)}/{len(general_dataloader)}. Epoch {str(epoch+1).ljust(4)}/{num_epochs}', end='  ', flush=True)
            
                # Meta-update
                output = model(coords)
                outer_loss = outer_loss_fn(output, targets, height, width)

                total_outer_loss += outer_loss

        avg_outer_loss = total_outer_loss / len(general_dataloader)

        # Meta-update (update shared block parameters) based on the average outer loss
        meta_optimizer.zero_grad()
        avg_outer_loss.backward()
        meta_optimizer.step()
            
        # Log the outer loss
        wandb.log({f'outer loss epoch {epoch+1}': avg_outer_loss.item()})

        # Check patience
        best_loss, wait, stop_training = check_patience(best_loss, avg_outer_loss.item(), wait, patience)
        if stop_training:
            print(f'Training stopped early at epoch {epoch+1}.')
            break

    wandb.finish()