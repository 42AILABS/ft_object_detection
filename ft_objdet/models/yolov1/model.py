import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leakyrelu(self.conv(x))
    
class Darknet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.architecture = [
            # (kernel_size, fiters, stride, padding)
            (7, 64, 2, 3), "M",
            (3, 192, 1, 1), "M",
            (1, 128, 1, 0), (3, 256, 1, 1), (1, 256, 1, 0), (3, 512, 1, 1), "M",
            [(1, 256, 1, 0), (3, 512, 1, 1), 4],
            (1, 512, 1, 0), (3, 1024, 1, 1), "M",
            [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
            (3, 1024, 1, 1), (3, 1024, 2, 1),
            (3, 1024, 1, 1), (3, 1024, 1, 1)
        ]
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def _create_conv_layers(self):
        layers = []
        for layer in self.architecture:
            if isinstance(layer, tuple):
                layers += [
                    CNNBlock(
                        self.in_channels,
                        layer[1], # out_channels
                        kernel_size=layer[0],
                        stride=layer[2],
                        padding=layer[3])
                ]
                self.in_channels = layer[1]
            elif isinstance(layer, list):
                conv1, conv2, repeat = layer
                for _ in range(repeat):
                    layers += [
                        CNNBlock(self.in_channels, conv1[1],
                               kernel_size=conv1[0],
                               stride=conv1[2],
                               padding=conv1[3]),
                        CNNBlock(conv1[1], conv2[1],
                               kernel_size=conv2[0],
                               stride=conv2[2],
                               padding=conv2[3])
                    ]
                    self.in_channels = conv2[1]
            elif layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return (nn.Sequential(*layers))
    
    def forward(self, x):
        return (self.layers(x))

class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, split_size=7, num_boxes=2, num_classes=20):
        super().__init__()
        self.backbone = Darknet(in_channels)

        self.S = split_size
        self.B = num_boxes
        self.C = num_classes

        self.darknet = Darknet(in_channels)
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * self.S**2, 496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(496, self.S**2 * (self.C + self.B*5))  # 동적 차원 계산
        )

    def forward(self, x):
        x = self.darknet(x)
        # (batch, S, S, C+B*5) 형태로 변환
        return self.fcs(x).reshape(-1, self.S, self.S, self.C + self.B*5)