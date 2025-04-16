import torch.nn as nn
import torch
from .l2norm import L2Norm
from .vgg import VGG
from .cnnblock import CNNBlock
class SSD(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone = VGG()
        self.l2norm = L2Norm(512, 20)
        self.num_classes = num_classes
        
        # Extra layers
        self.extras = nn.ModuleList([
            CNNBlock(512, 1024, kernel_size=3, padding=6, dilation=6),
            CNNBlock(1024, 1024, kernel_size=1),
            CNNBlock(1024, 256, kernel_size=1),
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            CNNBlock(512, 128, kernel_size=1),
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            CNNBlock(256, 128, kernel_size=1),
            CNNBlock(128, 256, kernel_size=3)
        ])

        # Prediction layers
        self.loc = nn.ModuleList([
            nn.Conv2d(512, 4*4, 3, padding=1),   # 38x38
            nn.Conv2d(1024, 6*4, 3, padding=1),  # 19x19
            nn.Conv2d(512, 6*4, 3, padding=1),   # 10x10
            nn.Conv2d(256, 6*4, 3, padding=1),   # 5x5
            nn.Conv2d(256, 4*4, 3, padding=1),   # 3x3
            nn.Conv2d(256, 4*4, 3, padding=1)    # 1x1
        ])
        
        self.conf = nn.ModuleList([
            nn.Conv2d(512, 4*num_classes, 3, padding=1),
            nn.Conv2d(1024, 6*num_classes, 3, padding=1),
            nn.Conv2d(512, 6*num_classes, 3, padding=1),
            nn.Conv2d(256, 6*num_classes, 3, padding=1),
            nn.Conv2d(256, 4*num_classes, 3, padding=1),
            nn.Conv2d(256, 4*num_classes, 3, padding=1)
        ])

    def forward(self, x):
        sources = []
        
        # 1. Backbone 처리 (conv4_3)
        for k in range(12):  # conv4_3의 마지막 CNNBlock까지
            x = self.backbone.layers[k](x)
        sources.append(self.l2norm(x))  # [38x38x512]

        # 2. Backbone 처리 (conv5_3)
        for k in range(12, len(self.backbone.layers)):
            x = self.backbone.layers[k](x)
        
        # 3. Extras 처리
        for k in range(len(self.extras)):
            x = self.extras[k](x)
            if k in [1, 3, 5, 7]:  # 주요 피처맵 수집 지점
                sources.append(x)

        # 4. 예측 레이어
        loc = torch.cat([l(src).view(x.size(0),-1,4) 
                       for src, l in zip(sources, self.loc)], 1)
        conf = torch.cat([c(src).view(x.size(0),-1,self.num_classes) 
                        for src, c in zip(sources, self.conf)], 1)
        
        return loc, conf