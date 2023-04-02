import torch
from torch import nn
from torch.nn import functional as F

class AdaNet_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding="same"),
                            )

    def forward(self, x):
        out = self.conv_block(x)
        return out

