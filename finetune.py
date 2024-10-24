import argparse, os
from types import SimpleNamespace
import torch
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import wandb
from model import SIREN
import loss_functions
from train_functions import Data, prepare_image_for_siren, check_patience
from PIL import Image
import numpy as np


def finetune(config):
    with wandb.init(project='SPO_finetuner_PC', config=config):
        config = wandb.config

        # General initializations
        dataset_path = r'C:\Users\pimde\OneDrive\Documents\GitHub\2AMM20-MetaLearningSharedParameters\celeba_thousand'
        dataset = Data(dataset_path, transform=ToTensor())
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loss_fn = loss_functions.MSE_loss()

        best_loss = 1.0
        wait = 0

        for i1, image in enumerate(dataloader):
            model = torch.load("saved_models/spo_model.pth")
            model = model.to(device)

            unique_params = list(model.unique_block.parameters()) + list(model.output_layer.parameters())
            optimizer = optim.Adam(unique_params, lr=config.learning_rate)

            coords, targets = prepare_image_for_siren(image)
            coords = coords.to(device)
            targets = targets.to(device)

            for epoch in range(config.epochs):
                output = model(coords)
                loss = loss_fn(output, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({f'loss image {i1+1}/{len(dataloader)}': loss.item()})
                print(f'\rImage {str(i1+1).ljust(3)}/{len(dataloader)}; epoch {str(epoch+1).ljust(4)}/{config.epochs} processed.', end='   ', flush=True)

                best_loss, wait, stop_training = check_patience(best_loss, loss.item(), wait, config.patience)

                if wait == 0:
                    torch.save(model, 'saved_models/finetuned_model.pth')
                if stop_training:
                    print(f'Training for image {i1+1} stopped early at epoch {epoch+1}.')
                    break

            model = torch.load("saved_models/finetuned_model.pth")
            model = model.to(device)
            output = model(coords)
            
            # Reconstruct and save the image
            save_reconstructed_image(coords, output, image, i1)

    
    wandb.finish()


def save_reconstructed_image(coords, output, original_image, image_index):
    """
    Convert the model's output back to an image format and save it as a file.
    """
    # Define the resolution of the image
    height, width = 218, 178  # CelebA's typical resolution is 178x218

    # Reshape the output to match the image shape
    image_array = output.view(height, width, 3).detach().cpu().numpy()
    
    # Convert the image from [-1, 1] range to [0, 255] for proper display and saving
    image_array = (image_array * 0.5 + 0.5) * 255.0
    image_array = image_array.astype(np.uint8)

    # Create and save the image using PIL
    img = Image.fromarray(image_array)
    output_dir = 'saved_output'  # Directory to save the reconstructed images
    os.makedirs(output_dir, exist_ok=True)
    img.save(os.path.join(output_dir, f'{image_index+1}_output.jpg'))


    # Save the original input image
    original_image = original_image[0]
    original_input_image_array = original_image.permute(1, 2, 0).detach().cpu().numpy()  # Convert to (H, W, C)
    
    # Convert from [0, 1] to [0, 255] for display
    original_input_image_array = (original_input_image_array * 255.0).astype(np.uint8)

    # Create and save the original input image
    original_input_img = Image.fromarray(original_input_image_array)
    original_input_dir = 'saved_input'
    os.makedirs(original_input_dir, exist_ok=True)
    original_input_img.save(os.path.join(original_input_dir, f'{image_index+1}_input.jpg'))


# Initialize default hyperparameters and parser functionality
default_config = SimpleNamespace(
    epochs=3000,
    patience=300,
    learning_rate=1e-3,
    batch_size=1,
)

def parse_args():
    argparser = argparse.ArgumentParser(description='Process hyperparameters')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='maximum epochs')
    argparser.add_argument('--patience', type=int, default=default_config.patience, help='patience')
    argparser.add_argument('--learning_rate', type=float, default=default_config.learning_rate, help='learning rate')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


if __name__ == '__main__':
    parse_args()
    finetune(default_config)