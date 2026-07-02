import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd=0, stride=1, activation="lrelu", norm_type="instance"):
        super().__init__()
        self.add_module("conv", nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=ker_size, stride=stride, padding=padd))
        # self.add_module("noise", NoiseInjection(out_channel))
        self.add_module(*get_norm_layer(out_channel, norm_type))
        self.add_module(*get_activation(activation))

def get_norm_layer(out_channel: int, type: str):
    normalization_layers = {
        "batch": nn.BatchNorm3d(out_channel, affine=True),
        "instance": nn.InstanceNorm3d(out_channel, affine=True),
        "group": nn.GroupNorm(16, out_channel, affine=True),
    }
    return type, normalization_layers[type]


def get_activation(name):
    activations = {
        "lrelu": nn.LeakyReLU(0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "prelu": nn.PReLU(num_parameters=1, init=0.25),
        "selu": nn.SELU(inplace=True),
        "relu": nn.ReLU(inplace=True),
    }
    return name, activations[name]

def upsample(x: torch.Tensor, size, mode: str = "trilinear") -> torch.Tensor:
    # size: (Y, Z, X)
    if mode == "nearest":
        return F.interpolate(x, size=size, mode="nearest")
    return F.interpolate(x, size=size, mode=mode, align_corners=True)

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, x):
        noise = torch.randn_like(x)
        return x + self.weight * noise