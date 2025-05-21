"""Custom loss functions for training neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in classification tasks.

    This implementation follows the paper "Focal Loss for Dense Object Detection"
    (https://arxiv.org/abs/1708.02002).

    Args:
        alpha (torch.Tensor, optional): Class weights for handling imbalanced datasets.
            Shape should be (num_classes,) or scalar. Defaults to None.
        gamma (float, optional): Focusing parameter that reduces the relative loss
            for well-classified examples. Defaults to 2.0.
        reduction (str, optional): Specifies the reduction to apply to the output.
            Options: 'none' | 'mean' | 'sum'. Defaults to 'mean'.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Compute the focal loss.

        Args:
            inputs (torch.Tensor): Raw logits from the model,
                shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth class indices,
                shape (batch_size,).

        Returns:
            torch.Tensor: Computed focal loss. If reduction is 'none',
                shape is (batch_size,). Otherwise, scalar.
        """
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss