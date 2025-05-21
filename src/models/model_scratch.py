import torch
import torch.nn as nn
import torch.nn.functional as F

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0., 6.) / 6.

class HSigmoid(nn.Module):
    def forward(self, x):
        return F.relu6(x + 3) / 6

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced_channels = in_channels // reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            HSigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size, stride, use_se, activation):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if activation == "RE":
            act = nn.ReLU(inplace=True)
        elif activation == "HS":
            act = HSwish()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act,
            SEBlock(hidden_dim) if use_se else nn.Identity(),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileNetV3SmallScratch(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HSwish()
        )

        self.blocks = nn.Sequential(
            InvertedResidual(16, 16, 16, 3, 2, True, "RE"),
            InvertedResidual(16, 72, 24, 3, 2, False, "RE"),
            InvertedResidual(24, 88, 24, 3, 1, False, "RE"),
            InvertedResidual(24, 96, 40, 5, 2, True, "HS"),
            InvertedResidual(40, 240, 40, 5, 1, True, "HS"),
            InvertedResidual(40, 240, 40, 5, 1, True, "HS"),
            InvertedResidual(40, 120, 48, 5, 1, True, "HS"),
            InvertedResidual(48, 144, 48, 5, 1, True, "HS"),
            InvertedResidual(48, 288, 96, 5, 2, True, "HS"),
            InvertedResidual(96, 576, 96, 5, 1, True, "HS"),
            InvertedResidual(96, 576, 96, 5, 1, True, "HS")
        )

        self.final_layers = nn.Sequential(
            nn.Conv2d(96, 576, 1, 1, 0, bias=False),
            nn.BatchNorm2d(576),
            HSwish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 1024, 1),
            HSwish(),
            nn.Dropout(0.2),
            nn.Conv2d(1024, num_classes, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final_layers(x)
        return x.view(x.size(0), -1)