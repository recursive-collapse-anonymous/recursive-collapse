"""Noise-prediction networks: two U-Nets for images and one MLP for the 2D spiral.

Every network has the signature forward(x, t) -> predicted noise, with t a
LongTensor of diffusion steps. Sampling is done by common.ddim_sample.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal embedding of the diffusion step."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class ResBlock(nn.Module):
    """Residual block with FiLM time conditioning (no normalization)."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch * 2)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.relu(self.conv1(x))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.relu(self.conv2(h))
        return h + self.shortcut(x)


class ResBlockNorm(nn.Module):
    """Residual block with GroupNorm, SiLU and FiLM time conditioning."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch * 2)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.shortcut(x)


class UNetSmall(nn.Module):
    """U-Net for 28x28 single-channel images (MNIST, Fashion-MNIST)."""

    def __init__(self, channels=1, base_dim=32, time_dim=64, time_embed_dim=32):
        super().__init__()
        self.time = TimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )
        d = base_dim
        self.down1 = ResBlock(channels, d, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResBlock(d, d * 2, time_dim)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = ResBlock(d * 2, d * 2, time_dim)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_block1 = ResBlock(d * 2 + d * 2, d * 2, time_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_block2 = ResBlock(d * 2 + d, d, time_dim)
        self.out = nn.Conv2d(d, channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time(t))

        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)
        mid = self.mid(self.pool2(x2), t_emb)

        u = torch.cat([self.up1(mid), x2], dim=1)
        u = self.up_block1(u, t_emb)
        u = torch.cat([self.up2(u), x1], dim=1)
        u = self.up_block2(u, t_emb)
        return self.out(u)


class UNetAttention(nn.Module):
    """U-Net for 32x32 RGB images (CIFAR-10), with self-attention at 4x4."""

    def __init__(self, channels=3, base_dim=8, time_dim=32, time_embed_dim=64, num_heads=4):
        super().__init__()
        d = base_dim

        self.time = TimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder: 32 -> 16 -> 8 -> 4
        self.down32_1 = ResBlockNorm(channels, d, time_dim)
        self.down32_2 = ResBlockNorm(d, d, time_dim)
        self.pool32 = nn.MaxPool2d(2)

        self.down16_1 = ResBlockNorm(d, d * 2, time_dim)
        self.down16_2 = ResBlockNorm(d * 2, d * 2, time_dim)
        self.pool16 = nn.MaxPool2d(2)

        self.down8_1 = ResBlockNorm(d * 2, d * 4, time_dim)
        self.down8_2 = ResBlockNorm(d * 4, d * 4, time_dim)
        self.pool8 = nn.MaxPool2d(2)

        # Bottleneck at 4x4
        self.mid1 = ResBlockNorm(d * 4, d * 4, time_dim)
        self.mid2 = ResBlockNorm(d * 4, d * 4, time_dim)
        self.attn = nn.MultiheadAttention(embed_dim=d * 4, num_heads=num_heads, batch_first=True)

        # Decoder: 4 -> 8 -> 16 -> 32
        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(d * 4, d * 4, 3, padding=1),
        )
        self.up8_1 = ResBlockNorm(d * 4 + d * 4, d * 4, time_dim)
        self.up8_2 = ResBlockNorm(d * 4, d * 4, time_dim)

        self.up16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(d * 4, d * 4, 3, padding=1),
        )
        self.up16_1 = ResBlockNorm(d * 4 + d * 2, d * 2, time_dim)
        self.up16_2 = ResBlockNorm(d * 2, d * 2, time_dim)

        self.up32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(d * 2, d * 2, 3, padding=1),
        )
        self.up32_1 = ResBlockNorm(d * 2 + d, d, time_dim)
        self.up32_2 = ResBlockNorm(d, d, time_dim)

        self.out = nn.Conv2d(d, channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time(t))

        x32 = self.down32_2(self.down32_1(x, t_emb), t_emb)
        x16 = self.pool32(x32)
        x16 = self.down16_2(self.down16_1(x16, t_emb), t_emb)
        x8 = self.pool16(x16)
        x8 = self.down8_2(self.down8_1(x8, t_emb), t_emb)
        x4 = self.pool8(x8)

        mid = self.mid2(self.mid1(x4, t_emb), t_emb)

        B, C, H, W = mid.shape
        residual = mid
        mid = mid.flatten(2).transpose(1, 2)
        mid, _ = self.attn(mid, mid, mid)
        mid = mid.transpose(1, 2).reshape(B, C, H, W) + residual

        u8 = torch.cat([self.up8(mid), x8], dim=1)
        u8 = self.up8_2(self.up8_1(u8, t_emb), t_emb)

        u16 = torch.cat([self.up16(u8), x16], dim=1)
        u16 = self.up16_2(self.up16_1(u16, t_emb), t_emb)

        u32 = torch.cat([self.up32(u16), x32], dim=1)
        u32 = self.up32_2(self.up32_1(u32, t_emb), t_emb)

        return self.out(u32)


class MLP2D(nn.Module):
    """Fully connected noise predictor for the 2D spiral."""

    def __init__(self, data_dim=2, hidden_dim=256, time_dim=64):
        super().__init__()
        self.data_dim = data_dim
        self.time_embed = TimeEmbedding(time_dim // 2)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim // 2, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embed(t))
        return self.net(torch.cat([x, t_emb], dim=1))


def build_model(name, **kwargs):
    models = {"unet-small": UNetSmall, "unet-attention": UNetAttention, "mlp2d": MLP2D}
    if name not in models:
        raise ValueError(f"unknown model {name!r}, expected one of {sorted(models)}")
    return models[name](**kwargs)
