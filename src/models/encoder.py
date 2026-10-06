import torch
import torch.nn as nn
from src.logger import logging

## encoder class
class Encoder(nn.Module):
    """
    This encoder class responsible for training Convolution network
    """
    def __init__(self, in_channel: int, kernel_size: int = 1, padding: int = 1):
        super().__init__()
        """
        Parameters:
        in_channel  (int):   Number of b-values (input channels)
        """
        ## Encoder Net
        self.enet = nn.Sequential(
            ##first conv layer
            nn.Conv2d(in_channel, 128, kernel_size, padding),
            nn.BatchNorm2d(128),
            nn.GELU(),
            MCDropout(0.10),

            nn.Conv2d(128,256,kernel_size, padding),
            nn.BatchNorm2d(256),
            nn.GELU(),

            nn.Conv2d(256, 128, kernel_size, padding),
            nn.BatchNorm2d(128),
            nn.GELU(),
            MCDropout(0.20),

            nn.Conv2d(128, 128, kernel_size, padding),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 128, kernel_size, padding),
            nn.BatchNorm2d(128),
            nn.GELU(),
            MCDropout(0.30),

            nn.Conv2d(128, 64, kernel_size, padding),
            nn.GELU(),
            nn.Flatten()
        )