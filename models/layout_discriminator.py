import torch
import torch.nn as nn


class LayoutDiscriminator2D(nn.Module):
    """
    Global 2D layout discriminator.

    Input:
        layout2d: (B, C, D, W)

    Output:
        scalar score per sample: (B,)
    """

    def __init__(self, in_channels: int = 5, nfc: int = 32):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, nfc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfc, nfc, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfc, nfc * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfc * 2, nfc * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(nfc * 2, nfc * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.head = nn.Linear(nfc * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h)
        h = h.flatten(1)
        out = self.head(h)
        return out.view(-1)