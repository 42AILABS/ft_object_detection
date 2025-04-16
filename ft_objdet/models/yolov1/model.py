import torch.nn as nn
from .darknet import Darknet

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