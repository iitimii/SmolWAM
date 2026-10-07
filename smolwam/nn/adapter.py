"""Latent adapter: single-frame Wan VAE latents <-> the clip-mode latents the Cosmos trunk was trained on.

Cosmos Policy encodes every observation inside a 33-frame clip (each image repeated four times, after blank frames),
so its trunk expects latents produced with the VAE's causal temporal cache. SmolWAM encodes each camera image once,
as a one-frame video, which is far cheaper but lands in a different latent distribution: the teacher's action error
rises from 0.0089 to 0.038 on single-frame latents. A small per-camera residual conv net (0.5 M parameters) maps
single-frame latents to clip-mode latents and restores 0.0092. An inverse adapter maps predicted (clip-mode) future
latents back to single-frame latents so they can be decoded frame by frame. Both are trained once on paired
encodings of the same images (`smolwam.train.adapter`) and stay frozen afterwards.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from smolwam.layout import LATENT_CH


class LatentAdapter(nn.Module):
    """Per-camera affine map plus a zero-initialised residual conv net, on (B, 16, 28, 28) latents."""

    def __init__(self, width: int = 96, n_cams: int = 2, init_scale: float = 1.8):
        super().__init__()
        self.scale = nn.Parameter(torch.full((n_cams, LATENT_CH, 1, 1), init_scale))
        self.shift = nn.Parameter(torch.zeros(n_cams, LATENT_CH, 1, 1))
        self.cam_emb = nn.Parameter(torch.zeros(n_cams, width, 1, 1))
        self.inp = nn.Conv2d(LATENT_CH, width, 3, 1, 1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(8, width),
                    nn.SiLU(),
                    nn.Conv2d(width, width, 3, 1, 1),
                    nn.GroupNorm(8, width),
                    nn.SiLU(),
                    nn.Conv2d(width, width, 3, 1, 1),
                )
                for _ in range(3)
            ]
        )
        self.out = nn.Conv2d(width, LATENT_CH, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
        """z: (B, 16, 28, 28) latents; cam: (B,) camera index (0 = wrist, 1 = primary)."""
        base = z * self.scale[cam] + self.shift[cam]
        h = self.inp(z) + self.cam_emb[cam]
        for blk in self.blocks:
            h = h + blk(h)
        return base + self.out(F.silu(h))


def inverse_adapter(width: int = 96, n_cams: int = 2) -> LatentAdapter:
    """The clip -> single-frame direction starts from the inverse of the forward map's initial scale."""
    return LatentAdapter(width, n_cams, init_scale=0.55)


def load_adapter_state(path: str | Path) -> dict[str, torch.Tensor]:
    """Adapter weights keyed `adapter.*` / `adapter_inv.*` (the names inside `SmolWAMModel`), from the safetensors
    file written by `smolwam.train.adapter` or from a research-code pickle ({"fwd": ..., "inv": ...})."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return {
        **{f"adapter.{k}": v for k, v in ck["fwd"].items()},
        **{f"adapter_inv.{k}": v for k, v in ck["inv"].items()},
    }
