import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from vector_quantize_pytorch import ResidualVQ
import numpy as np
from local_attention import LocalMHA
from local_attention.transformer import FeedForward, DynamicPositionBias

# Generator

class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1)

    def forward(self, x):
        return self._conv_forward(F.pad(x, [self.causal_padding, 0]), self.weight, self.bias)


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.causal_padding = self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1 - self.stride[0]

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose1d')

        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)
        return F.conv_transpose1d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)[..., :-self.causal_padding]


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.dilation = dilation

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=7, dilation=dilation),
            nn.ELU(),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=out_channels // 2,
                         out_channels=out_channels // 2, dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels // 2,
                         out_channels=out_channels // 2, dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels // 2,
                         out_channels=out_channels // 2, dilation=9),
            nn.ELU(),
            CausalConv1d(in_channels=out_channels // 2, out_channels=out_channels,
                         kernel_size=2 * stride, stride=stride)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, out_channels, stride):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConvTranspose1d(in_channels=2 * out_channels,
                                  out_channels=out_channels,
                                  kernel_size=2 * stride, stride=stride),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9),

        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C, D, factors):

        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(in_channels=1, out_channels=C, kernel_size=7),
            nn.ELU(),
            EncoderBlock(out_channels=2 * C, stride=factors[0]),
            nn.ELU(),
            EncoderBlock(out_channels=4 * C, stride=factors[1]),
            nn.ELU(),
            EncoderBlock(out_channels=8 * C, stride=factors[2]),
            nn.ELU(),
            EncoderBlock(out_channels=16 * C, stride=factors[3]),
            nn.ELU(),
            CausalConv1d(in_channels=16 * C, out_channels=D, kernel_size=3)
        )


    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, C, D, factors):
        super().__init__()

        self.layers = nn.Sequential(
            CausalConv1d(in_channels=D, out_channels=16 * C, kernel_size=7),
            nn.ELU(),
            DecoderBlock(out_channels=8 * C, stride=factors[3]),
            nn.ELU(),
            DecoderBlock(out_channels=4 * C, stride=factors[2]),
            nn.ELU(),
            DecoderBlock(out_channels=2 * C, stride=factors[1]),
            nn.ELU(),
            DecoderBlock(out_channels=C, stride=factors[0]),
            nn.ELU()
            # ,CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)
        )

    def forward(self, x):
        return self.layers(x)

def exists(val):
    return val is not None

class LocalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        heads,
        window_size,
        dynamic_pos_bias = False,
        **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.layers = nn.ModuleList([])

        self.pos_bias = None
        if dynamic_pos_bias:
            self.pos_bias = DynamicPositionBias(dim = dim // 2, heads = heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalMHA(dim = dim, heads = heads, qk_rmsnorm = True, window_size = window_size, use_rotary_pos_emb = not dynamic_pos_bias, use_xpos = True, **kwargs),
                FeedForward(dim = dim)
            ]))

    def forward(self, x):
        w = self.window_size

        attn_bias = self.pos_bias(w, w * 2) if exists(self.pos_bias) else None

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        return x

class SoundStream(nn.Module):
    def __init__(self, C, D, n_q, codebook_size, factors):
        super().__init__()
        self.encoder = Encoder(C=C, D=D, factors=factors)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q, dim=D, codebook_size=codebook_size,
            kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=C, D=D, factors=factors)
        self.last_conv = CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)
    def forward(self, x):
        e = self.encoder(x)
        e = e.permute(0, 2, 1).contiguous()
        quantized, _, _ = self.quantizer(e)
        quantized = quantized.permute(0, 2, 1).contiguous()
        o = self.decoder(quantized)
        o = self.last_conv(o)
        return o

class SoundStream1(nn.Module):
    def __init__(self, C, D, n_q, codebook_size, factors):
        super().__init__()
        self.encoder = Encoder(C=C, D=D, factors=factors)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q, dim=D, codebook_size=codebook_size,
            kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=C, D=D, factors=factors)
        self.last_conv = CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)
    def forward(self, x):
        e = self.encoder(x)
        e = e.permute(0, 2, 1).contiguous()
        quantized, _, commit_loss = self.quantizer(e)
        commit_loss1 = commit_loss.sum()
        quantized = quantized.permute(0, 2, 1).contiguous()
        o = self.decoder(quantized)
        o = self.last_conv(o)
        return o,commit_loss1

class SoundStream2(nn.Module):
    def __init__(self, C, D, n_q, codebook_size, factors):
        super().__init__()
        self.encoder = Encoder(C=C, D=D, factors=factors)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q, dim=D, codebook_size=codebook_size,
            kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2
        )
        self.decoder = Decoder(C=C, D=D, factors=factors)
        attn_kwargs = dict(
            dim=D,
            dim_head=64,
            heads=8,
            depth=1,
            window_size=128,
            xpos_scale_base=None,
            dynamic_pos_bias=False,
            prenorm=True,
            causal=True
        )

        self.encoder_attn = LocalTransformer(**attn_kwargs)
        self.decoder_attn = LocalTransformer(**attn_kwargs)
        self.last_conv = CausalConv1d(in_channels=C, out_channels=1, kernel_size=7)

    def forward(self, x):
        e = self.encoder(x)
        e = e.permute(0, 2, 1).contiguous()
        e = self.encoder_attn(e)
        quantized, _, _ = self.quantizer(e)
        quantized = self.decoder_attn(quantized)
        quantized = quantized.permute(0, 2, 1).contiguous()
        o = self.decoder(quantized)
        o = self.last_conv(o)
        return o

if __name__ == "__main__":
    x = torch.randn(1, 1, 8000)
    S = SoundStream(32, 32, 2, 1024, [2, 4, 5, 8]) # C, D, n_q, codebook_size, factors
    y = S(x)
    print(y.shape)