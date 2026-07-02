import torch.nn as nn
from .conv_block import ConvBlock

class Discriminator(nn.Module):
    def __init__(self, opt, in_channels=None):
        super().__init__()
        hidden_channel = int(opt.hidden_channel)

        k = int(opt.ker_size)
        p = int(opt.padd_size)
        stride = int(opt.stride_size)
        num_layer = int(opt.num_layer)
        activation = opt.activation

        if in_channels is None:
            in_channels = int(opt.repr_channels)

        self.head = ConvBlock(
            in_channel=in_channels,
            out_channel=hidden_channel,
            ker_size=k,
            padd=p,
            stride=stride,
            activation=activation,
            norm_type=opt.norm_layer
        )

        self.body = nn.Sequential()
        for i in range(num_layer - 2):
            block = ConvBlock(
                in_channel=hidden_channel,
                out_channel=hidden_channel,
                ker_size=k,
                padd=p,
                stride=stride,
                activation=activation,
                norm_type=opt.norm_layer
            )
            self.body.add_module(f"block{i+1}", block)

        self.tail = nn.Conv3d(hidden_channel, 1, kernel_size=k, padding=p, stride=stride)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x