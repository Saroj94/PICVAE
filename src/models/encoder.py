import torch
import torch.nn as nn
from src.logger import logging

## encoder class
class Encoder(nn.Module):
    """
    Encoder: CNN neural network train on multi-b values MRI data to generate p(z|x)
    """
    def __init__(self, in_channel: int, 
                 output_enco: int = 64,
                 kernel_size: int = 1, 
                 padding: int = 1, 
                 dropout = (0.10, 0.20, 0.30), 
                 latent_dim: int = 64):
        super().__init__()
        """
        Parameters:
        in_channel  (int):   Number of b-values (input channels)
        """
        ## unpack the dropout values
        self.p1, self.p2, self.p3 = dropout

        ## Encoder Net
        self.encodernet = nn.Sequential(
            ##first conv layer
            nn.Conv2d(in_channel, 128, kernel_size, padding), nn.BatchNorm2d(128),nn.GELU(),MCDropout(self.p1),
            nn.Conv2d(128,256,kernel_size, padding),nn.BatchNorm2d(256),nn.GELU(),
            nn.Conv2d(256, 128, kernel_size, padding),nn.BatchNorm2d(128),nn.GELU(),MCDropout(self.p2),
            nn.Conv2d(128, 128, kernel_size, padding),nn.BatchNorm2d(128),nn.GELU(),
            nn.Conv2d(128, 128, kernel_size, padding),nn.BatchNorm2d(128),nn.GELU(),MCDropout(self.p3),
            nn.Conv2d(128, output_enco, kernel_size, padding),nn.GELU(),nn.Flatten()
        )

        ## encoder output
        encoder_output_dim = output_enco*3*3

        ## mean and variance for latent space from encodernet output
        self.mu = nn.Linear(encoder_output_dim, latent_dim)
        self.log_var = nn.Linear(encoder_output_dim, latent_dim)

    ## Encoder part
    def Encoder(self, patches):
            ## image vector
            img_vec = self.encodernet(patches)
            ## mean
            mu = self.mu(img_vec)
            ## variance
            log_var = self.log_var(img_vec)
            return mu, log_var

