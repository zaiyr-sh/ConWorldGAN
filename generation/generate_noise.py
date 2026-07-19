# Code based on https://github.com/tamarott/SinGAN
import torch


def generate_spatial_noise(size, device):
    """Sample a standard-normal spatial noise tensor on ``device``."""

    return torch.randn(size, device=device)
