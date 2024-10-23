import argparse
from types import SimpleNamespace
import torch
import torch.optim as optim
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import wandb
from model import SIREN
import loss_functions
from train_functions import Data, prepare_image_for_siren, check_patience


def pretrain(config):
    with wandb.init(project='SPO_pretrainer_test1', config=config):
        config = wandb.config
        
        # General initializations
        dataset_path = r'C:\Users\pimde\Documents\2AMM20\CelebA_small'
        train_dataset = Data(dataset_path, transform=ToTensor())
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Set SIREN initialization parameters
        shared_size = (config.shared_width, config.shared_depth)
        unique_size = (config.unique_width, config.unique_depth)  # model will already include 2 unique layers by default
        in_features = 2  # 2 input values for the coordinate pair (x,y)
        out_features = 3  # 3 output values for the RGB value
        model = SIREN(shared_size, unique_size, in_features, out_features, config.omega_0)
        model = model.to(device)
        
        loss_fn = loss_functions.MSE_loss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
    

        best_loss = 1.0
        wait = 0
    
        for epoch in range(config.epochs):
            for i, batch in enumerate(train_dataloader):
                running_loss = 0
                batch_coords, batch_targets = [], []
            
                for image in batch:
                    coords, targets = prepare_image_for_siren(image)
                    batch_coords.append(coords)
                    batch_targets.append(targets)
            
                batch_coords = torch.stack(batch_coords).to(device)
                batch_targets = torch.stack(batch_targets).to(device)
            
                output = model(batch_coords)
                batch_loss = loss_fn(output, batch_targets)
                running_loss += batch_loss

                print(f'\rEpoch {str(epoch+1).ljust(3)}: Processed batch {str(i+1).ljust(2)}/{len(train_dataloader)}.', end='   ', flush=True)
        
            avg_loss = running_loss / len(train_dataloader)
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()
        
            wandb.log({f'avg_loss': avg_loss.item()})
        
            best_loss, wait, stop_training = check_patience(best_loss, avg_loss.item(), wait, config.patience)

            if wait == 0:
                torch.save(model.state_dict(), 'saved_models/model_params.pth')
            if stop_training:
                print(f'Training stopped early at epoch {epoch+1}.')
                break
    
    wandb.finish()


# Initialize default hyperparameters and parser functionality
default_config = SimpleNamespace(
    epochs=10000,
    patience=50,
    learning_rate=1e-4,
    batch_size=5,
    shared_width=256,
    shared_depth=5,
    unique_width=128,
    unique_depth=1,
    omega_0=30
)

def parse_args():
    argparser = argparse.ArgumentParser(description='Process hyperparameters')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs, help='maximum epochs')
    argparser.add_argument('--patience', type=int, default=default_config.patience, help='patience')
    argparser.add_argument('--learning_rate', type=float, default=default_config.learning_rate, help='learning rate')
    argparser.add_argument('--batch_size', type=int, default=default_config.batch_size, help='batch size')
    argparser.add_argument('--shared_width', type=int, default=default_config.shared_width, help='shared width')
    argparser.add_argument('--shared_depth', type=int, default=default_config.shared_depth, help='shared depth')
    argparser.add_argument('--unique_width', type=int, default=default_config.unique_width, help='unique width')
    argparser.add_argument('--unique_depth', type=int, default=default_config.unique_depth, help='unique depth')
    argparser.add_argument('--omega_0', type=int, default=default_config.omega_0, help='omega_0')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


if __name__ == '__main__':
    parse_args()
    pretrain(default_config)