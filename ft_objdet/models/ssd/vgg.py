import torch.nn as nn
from .cnnblock import CNNBlock

class VGG(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.architecture = [
            # Backbone only (conv1~conv5_3)
            (3, 64, 1, 1), (3, 64, 1, 1), "M",
            (3, 128, 1, 1), (3, 128, 1, 1), "M",
            (3, 256, 1, 1), (3, 256, 1, 1), (3, 256, 1, 1), ("M", 2, 2, 0, True),
            (3, 512, 1, 1), (3, 512, 1, 1), (3, 512, 1, 1), "M",
            (3, 512, 1, 1), (3, 512, 1, 1), (3, 512, 1, 1), ("M", 3, 1, 1, False)
        ]
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        for layer in self.architecture:
            if isinstance(layer, tuple):
                if (layer[0] == "M"):
                    layers.append(nn.MaxPool2d(
                        kernel_size=layer[1], stride=layer[2], 
                        padding=layer[3], ceil_mode=layer[4]
                    ))
                else:
                    k, f, s, p = layer
                    layers.append(CNNBlock(
                        self.in_channels, f, 
                        kernel_size=k, stride=s, padding=p
                    ))
                    self.in_channels = f
            elif layer == "M":
                layers.append(nn.MaxPool2d(2, 2))
        return layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
