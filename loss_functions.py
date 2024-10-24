import torch
from torch import nn
from pytorch_msssim import ssim


class MSE_loss(nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, predicted, target):
        # Compute Mean Squared Error
        return torch.mean((predicted - target) ** 2)

class SSIM_loss(nn.Module):
    def __init__(self):
        super(SSIM_loss, self).__init__()

    def forward(self, predicted_img, target_img, height, width):
        # Reshape the predicted_img and target_img back to [batch_size, channels, height, width]
        predicted_img = predicted_img.contiguous().reshape(1, 3, height, width)  # Assuming batch_size = 1
        target_img = target_img.contiguous().reshape(1, 3, height, width)
        
        # Compute SSIM
        ssim_score = ssim(predicted_img, target_img, data_range=1.0, size_average=True)
        
        return 1 - ssim_score