import torch
import wandb
from model import SIREN
import loss_functions
from train_functions import train_model, get_N_shared_layers


# Set SIREN initialization parameters
# Relevant if no pretrained model is loaded
shared_size = (256, 5)
unique_size = (128, 1)  # model will already include 2 unique layers by default
in_features = 2  # 2 input values for the coordinate pair (x,y)
out_features = 3  # 3 output values for the RGB value
omega_0 = 30  # 30 is the default as recommended by the SIREN paper

pretrained_model_path = r'path_to_model.pth'  # Define if pretrained model is to be used

pretrained = False  # Set according to model used
if pretrained:
    model = torch.load(pretrained_model_path)
else:
    model = SIREN(shared_size, unique_size, in_features, out_features, omega_0)


# Define general parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
dataset_path = r'C:\Users\pimde\Documents\2AMM20\CelebA_small'
train_fraction = 0.8  # Fraction of data to be used for training
inner_loss_fn = loss_functions.MSE_loss()
outer_loss_fn = loss_functions.MSE_loss()

# Define hyperparameters
K = 10  # nr of inner loop iterations
num_epochs = 4000  # nr of allowed meta-updates
inner_learning_rate = 1e-4
meta_learning_rate = 1e-3
general_batch_size = 15
mini_batch_size = 4096  # batch size of coordinate-value pairs to be processed per each of K iterations (image dimensions: 178x218)
patience = 10


# Initialize new W&B run
wandb.init(project='MetaLearnDataset-testrun4', config={
    'pretrained model': pretrained,
    'meta_learning_rate': meta_learning_rate,
    'inner_learning_rate': inner_learning_rate,
    'max_epochs': num_epochs,
    'meta-patience': patience,
    'K': K,
    'mini_batch_size': mini_batch_size,
    'omega_0': omega_0,
    'Inner loss function': 'MSE',
    'Outer loss function': 'MSE'
})

torch.cuda.empty_cache()
#torch.autograd.set_detect_anomaly(True)

# Let user decide what nr of layers of the pretrained model is to be shared among the data
if pretrained:
    print('Pre-trained model architecture:')
    for name, layer in model.named_modules():
        print(f'{name}: {layer}')
    nr_shared_layers = int(input('State nr of layers to be shared over the data: '))

    shared_params, original_unique_params = get_N_shared_layers(model, nr_shared_layers)
    train_model(model, device, dataset_path, K, num_epochs, inner_loss_fn, outer_loss_fn, inner_learning_rate, meta_learning_rate, general_batch_size, mini_batch_size, patience, pretrained, shared_params, original_unique_params)

else:
    train_model(model, device, dataset_path, train_fraction, K, num_epochs, inner_loss_fn, outer_loss_fn, inner_learning_rate, meta_learning_rate, general_batch_size, mini_batch_size, patience)