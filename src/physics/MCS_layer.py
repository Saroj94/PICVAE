import torch 
import torch.nn as nn
import torch.nn.functional as F

## Monte Carlo dropout
class MCDropout(nn.Module):
    """ 
    Monte Carlo sampling will help to estimate the uncertainty
    """
    def __init__(self, dropout_p = 0.1):
        super().__init__()
        self.p = dropout_p
        self.mc_active = True

    def forwad(self, x):
        return F.dropout(x, self.p, training=self.training or self.mc_active)