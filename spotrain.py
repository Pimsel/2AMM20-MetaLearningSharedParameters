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


def spotrain(config):
    with wandb.init(project='SPO_maintrainer_Snellius', config=config):
        config = wandb.config
        
        # General initializations
        dataset_path = os.path.expanduser('~/traindata/celeba_smaller')
        train_dataset = Data(dataset_path, transform=ToTensor())
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load pretrained SIREN
        model = torch.load("saved_models/pretrained_model.pth")
        model = model.to(device)

        shared_params = model.shared_block.parameters()
        original_unique_params = list(model.unique_block.parameters()) + list(model.output_layer.parameters())
        # Initialize outer loop optimizer
        meta_optimizer = optim.Adam(shared_params, lr=config.meta_learning_rate)
        loss_fn = loss_functions.MSE_loss()

        best_loss = 1.0
        wait = 0

        for epoch in range(config.epochs):
            for i1, image_batch in enumerate(train_dataloader):
                batch_total_loss = 0
                for image in image_batch:
                    coords, targets = prepare_image_for_siren(image)

                    # Create a copy of the unique parameters for the inner loop
                    unique_params = [param.to(device).requires_grad_(True) for param in original_unique_params]

                    # Inner loop
                    for k in range(config.K):
                        # Create a temporary optimizer for the inner loop (2nd order MAML)
                        inner_optimizer = optim.Adam(unique_params, lr=config.inner_learning_rate)

                        output = model(coords)
                        inner_loss = loss_fn(output, targets)
                
                        # Backward pass and update over only unique parameters
                        inner_optimizer.zero_grad()
                        inner_loss.backward(retain_graph=True)  # Retain graph for meta-update
                        inner_optimizer.step()
                            
                        # Log inner loss
                        wandb.log({f'inner_loss': inner_loss.item()})
                    
                    output = model(coords)
                    outer_loss = loss_fn(output, targets)
                    batch_total_loss += outer_loss
                
                avg_outer_loss = batch_total_loss / len(image_batch)

                meta_optimizer.zero_grad()
                avg_outer_loss.backward()
                meta_optimizer.step()

                wandb.log({f'avg_outer_loss': avg_outer_loss.item()})
            
            # Check patience
            best_loss, wait, stop_training = check_patience(best_loss, avg_outer_loss.item(), wait, config.patience)

            if wait == 0:
                torch.save(model.state_dict(), 'saved_models/spo_model_params.pth')
            if stop_training:
                print(f'Training stopped early at epoch {epoch+1}.')
                break
    
    wandb.finish()


# Initialize default hyperparameters and parser functionality
default_config = SimpleNamespace(
    epochs=100000,
    patience=20,
    batch_size=5,
    inner_learning_rate=5e-6,
    meta_learning_rate=5e-4,
    K=10
)

def parse_args():
    argparser = argparse.ArgumentParser(description='Process hyperparameters')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='maximum epochs')
    argparser.add_argument('--patience', type=int, default=default_config.patience, help='patience')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--inner_learning_rate', type=float, default=default_config.inner_learning_rate, help='inner learning rate')
    argparser.add_argument('--meta_learning_rate', type=float, default=default_config.meta_learning_rate, help='meta learning rate')
    argparser.add_argument('--K', type=int, default=default_config.K, help='inner loop iterations')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


if __name__ == '__main__':
    parse_args()
    spotrain(default_config)