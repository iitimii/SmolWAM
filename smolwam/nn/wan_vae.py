# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for SmolWAM from the Wan2.1 VAE as vendored in NVlabs/cosmos-policy
# (cosmos_policy/_src/predict2/tokenizers/wan2pt1.py and cosmos_policy/tokenizers/wan2pt1.py,
# commit 18a2accadf4e7a3531e56754102af5a24d2316da).

"""The Wan2.1 causal video VAE (encoder and decoder), loading the original `Wan2.1_VAE.pth`.

The VAE maps RGB video in [-1, 1] at H x W to 16-channel latents at H/8 x W/8. Time is compressed causally: the first
frame becomes one latent frame and every following group of four frames another, with features of earlier frames
carried across groups by a per-layer cache (`_CausalCache`). SmolWAM encodes every camera image as its own one-frame
video ("single-frame latents"); only the latent-adapter training step encodes multi-frame clips, to reproduce the
clip-mode latents Cosmos Policy was trained on.

Latents are normalised exactly as the Cosmos Policy interface does: `(mu - mean) * (1 / std)` with Wan's per-channel
statistics, which it stores (and inverts) in bfloat16. Cosmos Policy runs the whole VAE in bfloat16, and so did every
SmolWAM training run and evaluation, so bfloat16 is the default here too.

Removed relative to upstream: context-parallel plugins, the deterministic-seeding wrapper, benchmarking hooks and the
unused sampling / reparameterisation helpers.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

CACHE_T = 2  # frames of each causal-conv input kept for the next temporal chunk
TEMPORAL_WINDOW = 4  # pixel frames per latent frame after the first
LATENT_MEAN = (
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
)  # fmt: skip
LATENT_STD = (
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
)  # fmt: skip
_FIRST_CHUNK = "Rep"  # cache marker of a temporal upsampler that has only seen the first chunk


class _CausalCache:
    """The last frames of every causal convolution's input, carried from one temporal chunk to the next.

    Entries are keyed by the order in which the network visits its cached convolutions; `start_chunk` rewinds that
    counter before each chunk.
    """

    def __init__(self):
        self.store: dict[int, torch.Tensor | str] = {}
        self.idx = 0

    def start_chunk(self) -> None:
        self.idx = 0

    def conv(self, layer: CausalConv3d, x: torch.Tensor) -> torch.Tensor:
        prev = self.store.get(self.idx)
        new = x[:, :, -CACHE_T:].clone()
        if new.shape[2] < 2 and prev is not None:  # keep two frames: last of the previous chunk + this one
            new = torch.cat([prev[:, :, -1:], new], dim=2)
        out = layer(x, prev)
        self.store[self.idx] = new
        self.idx += 1
        return out


class CausalConv3d(nn.Conv3d):
    """3D convolution padded only on the past side in time (optionally with the cached previous frames)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            x = torch.cat([cache_x.to(x.device), x], dim=2)
            padding[4] -= cache_x.shape[2]
        return super().forward(F.pad(x, padding))


class RMS_norm(nn.Module):  # noqa: N801 (upstream name)
    """Channel-first RMS normalisation with a learned gain (no bias in the Wan VAE)."""

    def __init__(self, dim: int, images: bool = True):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones((dim, 1, 1) if images else (dim, 1, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma


class Upsample(nn.Upsample):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)  # nearest-exact upsampling has no bfloat16 kernel everywhere


class Resample(nn.Module):
    """2x spatial down/upsampling, optionally with 2x causal temporal down/upsampling (`time_conv`)."""

    def __init__(self, dim: int, mode: str):
        super().__init__()
        assert mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d")
        self.mode = mode
        if mode.startswith("upsample"):
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1)
            )
        else:
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        if mode == "upsample3d":
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample3d":
            self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x: torch.Tensor, cache: _CausalCache) -> torch.Tensor:
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            prev = cache.store.get(cache.idx)
            if prev is None:  # first chunk (the first frame) is not doubled in time
                cache.store[cache.idx] = _FIRST_CHUNK
            else:
                new = x[:, :, -CACHE_T:].clone()
                if new.shape[2] < 2:
                    first = isinstance(prev, str)
                    new = torch.cat([torch.zeros_like(new) if first else prev[:, :, -1:].to(new.device), new], dim=2)
                x = self.time_conv(x) if isinstance(prev, str) else self.time_conv(x, prev)
                cache.store[cache.idx] = new
                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0], x[:, 1]), 3).reshape(b, c, t * 2, h, w)
            cache.idx += 1
        t = x.shape[2]
        x = rearrange(self.resample(rearrange(x, "b c t h w -> (b t) c h w")), "(b t) c h w -> b c t h w", t=t)
        if self.mode == "downsample3d":
            prev = cache.store.get(cache.idx)
            if prev is None:  # first chunk: one frame, nothing to downsample in time
                cache.store[cache.idx] = x.clone()
            else:
                new = x[:, :, -1:].clone()
                x = self.time_conv(torch.cat([prev[:, :, -1:], x], 2))
                cache.store[cache.idx] = new
            cache.idx += 1
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(0.0),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, cache: _CausalCache) -> torch.Tensor:
        h = self.shortcut(x)
        for layer in self.residual:
            x = cache.conv(layer, x) if isinstance(layer, CausalConv3d) else layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention, applied to each frame independently."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.size()
        x = self.norm(rearrange(x, "b c t h w -> (b t) c h w"))
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3, -1).permute(0, 1, 3, 2).contiguous().chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = self.proj(x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w))
        return rearrange(x, "(b t) c h w-> b c t h w", t=t) + identity


def _run(layers: nn.Sequential, x: torch.Tensor, cache: _CausalCache) -> torch.Tensor:
    for layer in layers:
        if isinstance(layer, CausalConv3d):
            x = cache.conv(layer, x)
        elif isinstance(layer, (ResidualBlock, Resample)):
            x = layer(x, cache)
        else:
            x = layer(x)
    return x


class Encoder3d(nn.Module):
    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_downsample=(False, True, True),
    ):
        super().__init__()
        dims = [dim * u for u in [1, *dim_mult]]
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                downsamples.append(Resample(out_dim, "downsample3d" if temporal_downsample[i] else "downsample2d"))
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim)
        )
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, cache: _CausalCache) -> torch.Tensor:
        x = cache.conv(self.conv1, x)
        for part in (self.downsamples, self.middle, self.head):
            x = _run(part, x, cache)
        return x


class Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int,
        z_dim: int,
        dim_mult=(1, 2, 4, 4),
        num_res_blocks: int = 2,
        temporal_upsample=(True, True, False),
    ):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0]), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0])
        )
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            if i in (1, 2, 3):
                in_dim = in_dim // 2  # the previous upsampler halved the channels
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                upsamples.append(Resample(out_dim, "upsample3d" if temporal_upsample[i] else "upsample2d"))
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x: torch.Tensor, cache: _CausalCache) -> torch.Tensor:
        x = cache.conv(self.conv1, x)
        for part in (self.middle, self.upsamples, self.head):
            x = _run(part, x, cache)
        return x


class WanVAE(nn.Module):
    """Wan2.1 VAE (dim 96, 16 latent channels, 8x spatial and 4x causal temporal compression)."""

    def __init__(self, dim: int = 96, z_dim: int = 16):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder3d(dim, z_dim * 2)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim)

    @classmethod
    def from_file(cls, path: str | Path, device: str | torch.device = "cpu", dtype: torch.dtype = torch.bfloat16):
        """Load `Wan2.1_VAE.pth` (from Wan-AI/Wan2.1-T2V-1.3B) for inference."""
        with torch.device("meta"):
            vae = cls()
        vae.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), assign=True)
        return vae.to(device=device, dtype=dtype).eval().requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return self.conv1.weight.dtype

    def _stats(self, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and 1 / std as the Cosmos interface computes them: in bfloat16 whatever the VAE dtype."""
        mean = torch.tensor(LATENT_MEAN, dtype=torch.bfloat16, device=like.device)
        inv_std = 1.0 / torch.tensor(LATENT_STD, dtype=torch.bfloat16, device=like.device)
        shape = (1, self.z_dim, 1, 1, 1)
        return mean.view(shape).to(self.dtype), inv_std.view(shape).to(self.dtype)

    @torch.no_grad()
    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """(B, 3, T, H, W) video in [-1, 1] -> normalised latents (B, 16, 1 + (T - 1) // 4, H/8, W/8).

        Computed in the VAE's dtype and returned in the input dtype. A single image is a one-frame video (T = 1).
        """
        in_dtype, x = videos.dtype, videos.to(self.dtype)
        cache = _CausalCache()
        chunks = [x[:, :, :1]] + [x[:, :, i : i + TEMPORAL_WINDOW] for i in range(1, x.shape[2], TEMPORAL_WINDOW)]
        out = []
        for chunk in chunks:
            cache.start_chunk()
            out.append(self.encoder(chunk, cache))
        mu = self.conv1(torch.cat(out, 2)).chunk(2, dim=1)[0]
        mean, inv_std = self._stats(mu)
        return ((mu - mean) * inv_std).to(in_dtype)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalised latents (B, 16, T, h, w) -> video (B, 3, 1 + 4 (T - 1), 8h, 8w) in about [-1, 1], unclamped."""
        in_dtype, z = latents.dtype, latents.to(self.dtype)
        mean, inv_std = self._stats(z)
        x = self.conv2(z / inv_std + mean)
        cache = _CausalCache()
        out = []
        for i in range(x.shape[2]):
            cache.start_chunk()
            out.append(self.decoder(x[:, :, i : i + 1], cache))
        return torch.cat(out, 2).to(in_dtype)
