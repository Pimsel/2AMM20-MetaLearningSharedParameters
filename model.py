import torch
import torch.nn as nn
import math

# Define a SineLayer as to be used in a SIREN
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super(SineLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_first = is_first
        self.omega_0 = omega_0

        # Linear transformation
        self.linear = nn.Linear(in_features, out_features)

        # Weight initialization following SIREN paper recommendations
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            # First layer: Weights initialized with larger variance
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        x = self.linear(x)
        return torch.sin(self.omega_0 * x)


# Define a SIREN model or similar that takes shared and unique parameters
class SIREN(nn.Module):
    def __init__(self, shared_size, unique_size, in_features=2, out_features=3, omega_0=30):
        super(SIREN, self).__init__()

        self.shared_size = shared_size
        self.unique_size = unique_size
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        
        # First block of shared layers
        shared_layers = []
        shared_layers.append(SineLayer(in_features, shared_size[0], is_first=True, omega_0=omega_0))  # First layer
        for _ in range(shared_size[1] - 1):  # Remaining layers
            shared_layers.append(SineLayer(shared_size[0], shared_size[0], omega_0=omega_0))
        self.shared_block = nn.Sequential(*shared_layers)
        
        # Second block of unique layers
        unique_layers = []
        unique_layers.append(SineLayer(shared_size[0], unique_size[0], omega_0=omega_0))  # First unique layer
        for _ in range(unique_size[1]):  # Remaining unique layers
            unique_layers.append(SineLayer(unique_size[0], unique_size[0], omega_0=omega_0))
        self.unique_block = nn.Sequential(*unique_layers)
        
        # Final output layer (linear transformation, no sine activation)
        self.output_layer = nn.Linear(unique_size[0], out_features)

    def forward(self, x):
        # Pass through the shared block
        x = self.shared_block(x)
        # Pass through the unique block
        x = self.unique_block(x)
        # Final output layer
        return self.output_layer(x)