
import torch
from torch import nn


class ToySegmentationModel(nn.Module):
    """A tiny CNN for synthetic pixel-level segmentation."""

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
