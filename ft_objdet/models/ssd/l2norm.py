import torch.nn as nn
import torch

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_channels))
        self.reset_parameters(scale)
        
    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
        
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        return self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) * (x / norm)