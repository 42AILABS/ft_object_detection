import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leakyrelu(self.conv(x))