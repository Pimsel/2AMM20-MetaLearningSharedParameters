import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from model import SIREN
import loss_functions
from train_functions import train_model
import wandb


# Set SIREN initialization parameters
shared_size = (256, 5)
unique_size = (128, 1)  # plus 2 for transition layer and output layer
in_features = 2  # 2 input values for the coordinate pair (x,y)
out_features = 3  # 3 output values for the RGB value
omega_0 = 30  # 30 is the default as recommended by the SIREN paper

model = SIREN(shared_size, unique_size, in_features, out_features, omega_0)


# Define general parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device used: {device}')
model = model.to(device)
dataset_path = r'C:\Users\pimde\Documents\2AMM20\CelebA_small'
inner_loss_fn = loss_functions.MSE_loss()
outer_loss_fn = loss_functions.SSIM_loss()

# Define hyperparameters
K = 10  # nr of inner loop iterations
num_epochs = 4000  # nr of allowed meta-updates
inner_learning_rate = 1e-4
meta_learning_rate = 1e-3
general_batch_size = 15
mini_batch_size = 2048  # batch size of coordinate-value pairs to be processed per each of K iterations (image dimensions: 178x218)
patience = 10


# Initialize new W&B run
wandb.init(project='MetaLearnDataset-testrun3', config={
    'meta_learning_rate': meta_learning_rate,
    'inner_learning_rate': inner_learning_rate,
    'max_epochs': num_epochs,
    'meta-patience': patience,
    'K': K,
    'mini_batch_size': mini_batch_size,
    'omega_0': omega_0,
    'Inner loss function': 'MSE',
    'Outer loss function': 'SSIM'
})

torch.cuda.empty_cache()
#torch.autograd.set_detect_anomaly(True)
train_model(model, device, dataset_path, K, num_epochs, inner_loss_fn, outer_loss_fn, inner_learning_rate, meta_learning_rate, general_batch_size, mini_batch_size, patience)