import torch.nn as nn
from .conv_block import ConvBlock
import copy
from models.conv_block import upsample

def zero_pad_3d(p: int) -> nn.Module:
    # ConstantPad3d accepts (left,right, top,bottom, front,back)
    return nn.ConstantPad3d((p, p, p, p, p, p), 0.0)

class GrowingGenerator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.hidden_channel = int(opt.hidden_channel)
        self.repr_channels = int(opt.repr_channels)

        k = int(opt.ker_size)
        p = int(opt.padd_size)
        stride = int(opt.stride_size)
        activation = opt.activation

        self.n_blocks = max(1, int(opt.num_layer))

        self._pad = zero_pad_3d(1)
        pad_block = max(0, self.n_blocks - 1)
        self._pad_block = zero_pad_3d(pad_block)

        self.head = ConvBlock(in_channel=self.repr_channels, out_channel=self.hidden_channel, ker_size=k, padd=p, stride=stride, activation=activation, norm_type=opt.norm_layer)

        self.body = nn.ModuleList([])
        first_stage = nn.Sequential()
        for i in range(self.n_blocks):
            block = ConvBlock(in_channel=self.hidden_channel, out_channel=self.hidden_channel, ker_size=k, padd=p, stride=stride, activation=activation, norm_type=opt.norm_layer)
            first_stage.add_module(f"block{i+1}", block)
        self.body.append(first_stage)

        self.tail = nn.Conv3d(in_channels=self.hidden_channel, out_channels=self.repr_channels, kernel_size=k, padding=p, stride=stride)

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        assert isinstance(noise, (list, tuple)) and len(noise) >= 1
        assert len(noise) == len(noise_amp)
        assert len(noise) <= len(real_shapes)

        x = self.head(self._pad(noise[0]))
        x = upsample(x, size=(x.shape[2] + 2, x.shape[3] + 2, x.shape[4] + 2))
        x = self._pad_block(x)
        x_prev_out = self.body[0](x)

        for idx, stage in enumerate(self.body[1:], 1):
            tgt = real_shapes[idx]
            tgt_size = (int(tgt[2]), int(tgt[3]), int(tgt[4]))

            x_prev_out_1 = upsample(x_prev_out, size=tgt_size)

            extra = int(self.n_blocks) * 2
            x_prev_out_2 = upsample(
                x_prev_out,
                size=(tgt_size[0] + extra, tgt_size[1] + extra, tgt_size[2] + extra)
            )
            x_in = x_prev_out_2 + noise[idx] * noise_amp[idx]
            x_prev = stage(x_in)

            x_prev_out = x_prev + x_prev_out_1

        out = self.tail(self._pad(x_prev_out))
        return out