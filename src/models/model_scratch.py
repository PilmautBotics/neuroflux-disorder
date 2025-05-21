"""Implementation of MobileNetV3-Small architecture from scratch.

This module implements a lightweight convolutional neural network based on the
MobileNetV3-Small architecture, as described in the paper:
"Searching for MobileNetV3" - https://arxiv.org/abs/1905.02244
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HSwish(nn.Module):
    """Hard Swish activation function.

    Applies the hard swish function element-wise:
    Hardswish(x) = x * max(0, min(6, x + 3)) / 6
    """

    def forward(self, x):
        """Apply hard swish activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying hard swish activation.
        """
        return x * F.hardtanh(x + 3, 0., 6.) / 6.


class HSigmoid(nn.Module):
    """Hard Sigmoid activation function.

    Applies the hard sigmoid function element-wise:
    Hardsigmoid(x) = min(max(0, x + 3), 6) / 6
    """

    def forward(self, x):
        """Apply hard sigmoid activation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying hard sigmoid activation.
        """
        return F.relu6(x + 3) / 6


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction factor. Defaults to 4.
    """

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
        """Apply channel attention.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying channel attention.
        """
        return x * self.se(x)


class InvertedResidual(nn.Module):
    """Inverted Residual block with optional SE and different activations.

    Args:
        in_channels (int): Number of input channels.
        hidden_dim (int): Number of channels in the expansion layer.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the depthwise convolution kernel.
        stride (int): Stride of the depthwise convolution.
        use_se (bool): Whether to include Squeeze-and-Excitation block.
        activation (str): Type of activation function ('RE' for ReLU or 'HS' for HSwish).
    """

    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size,
                 stride, use_se, activation):
        super().__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if activation == "RE":
            act = nn.ReLU(inplace=True)
        elif activation == "HS":
            act = HSwish()

        self.block = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act,
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                     kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act,
            # Squeeze-and-Excitation
            SEBlock(hidden_dim) if use_se else nn.Identity(),
            # Pointwise projection
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """Apply inverted residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying the inverted residual block.
        """
        out = self.block(x)
        if self.use_res_connect:
            return x + out
        return out


class MobileNetV3SmallScratch(nn.Module):
    """MobileNetV3-Small architecture implemented from scratch.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 5.
    """

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
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final_layers(x)
        return x.view(x.size(0), -1)