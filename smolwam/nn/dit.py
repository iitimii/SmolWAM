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
# Adapted for SmolWAM from NVlabs/cosmos-policy (cosmos_policy/_src/predict2/networks/minimal_v4_dit.py and
# minimal_v1_lvg_dit.py, commit 18a2accadf4e7a3531e56754102af5a24d2316da).

"""The Cosmos-Predict2 video DiT trunk, reduced to the parts SmolWAM runs.

This is a pure-PyTorch port of the network inside Cosmos Policy 2B (`MinimalV1LVGDiT`): patch embedding, 3D RoPE,
AdaLN-LoRA transformer blocks (self-attention, cross-attention to the text context, GELU MLP), the timestep
embedding and the final layer. It has no dependency on transformer_engine, megatron or flash-attention, so it runs on
CUDA, Apple MPS and the CPU. Parameter and buffer names match upstream, so Cosmos Policy checkpoints (`net.*`) and
SmolWAM checkpoints (`trunk.*`) load unchanged.

Removed relative to upstream: context parallelism, FSDP, selective activation checkpointing, flex/NATTEN/i4 attention
backends, the KV cache, image-context cross-attention, extra absolute position embeddings and FPS-modulated RoPE.

One change of interface: the AdaLN modulation is applied per *token group* rather than per latent frame. SmolWAM
assembles a variable number of tokens from each latent frame ("slot"), and every token takes the shift/scale/gate of
the frame it came from (`token_group`). With one group per frame and every frame complete this is exactly upstream.
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn


def _no_autocast(device: torch.device):
    """Upstream computes the timestep embedding and the AdaLN modulation outside autocast (full precision on CUDA)."""
    if not torch.amp.is_autocast_available(device.type):  # e.g. the meta device used to count FLOPs
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, enabled=False)


class RMSNorm(nn.Module):
    """RMSNorm with transformer_engine semantics: statistics and gain in fp32, output in the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def apply_rotary_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Non-interleaved RoPE (transformer_engine, `bshd` layout), computed in fp32.

    t: (B, S, H, D) queries or keys; freqs: (S, 1, 1, D) rotation angles.
    """
    freqs = freqs[: t.shape[1]].transpose(0, 1).float()  # (1, S, 1, D)
    x = t.float()
    x1, x2 = x.chunk(2, dim=-1)
    out = x * freqs.cos() + torch.cat((-x2, x1), dim=-1) * freqs.sin()
    return out.to(t.dtype)


class Attention(nn.Module):
    """Multi-head attention with RMS-normalised queries and keys; RoPE on self-attention only."""

    def __init__(self, query_dim: int, context_dim: int | None, n_heads: int, head_dim: int):
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner_dim = n_heads * head_dim
        self.n_heads, self.head_dim = n_heads, head_dim
        self.q_proj = nn.Linear(query_dim, inner_dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner_dim, bias=False)
        self.output_proj = nn.Linear(inner_dim, query_dim, bias=False)

    def compute_qkv(
        self, x: torch.Tensor, context: torch.Tensor | None = None, rope: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns q, k, v of shape (B, S, H, Dh)."""
        context = x if context is None else context
        q, k, v = (
            rearrange(t, "b s (h d) -> b s h d", h=self.n_heads, d=self.head_dim)
            for t in (self.q_proj(x), self.k_proj(context), self.v_proj(context))
        )
        q, k = self.q_norm(q), self.k_norm(k)
        if self.is_selfattn and rope is not None:
            q, k = apply_rotary_pos_emb(q, rope), apply_rotary_pos_emb(k, rope)
        return q, k, v

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None, rope: torch.Tensor | None = None):
        q, k, v = (rearrange(t, "b s h d -> b h s d") for t in self.compute_qkv(x, context, rope))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.output_proj(rearrange(out, "b h s d -> b s (h d)"))


class GPT2FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d_model, d_ff, bias=False)
        self.layer2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.activation(self.layer1(x)))


class VideoRopePosition3DEmb(nn.Module):
    """3D RoPE over (frame, row, column) with NTK-aware extrapolation factors.

    Cosmos Policy uses h/w extrapolation 3.0 and t extrapolation 1.0, with FPS modulation off. Building the trunk with
    the defaults (1.0 everywhere, FPS modulation on) still loads every weight without error, but shifts the output by
    half its standard deviation: the zero-shot action error goes from 0.009 to 0.40.
    """

    def __init__(
        self,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
    ):
        super().__init__()
        dim_h = head_dim // 6 * 2
        dim_t = head_dim - 2 * dim_h
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.register_buffer("dim_spatial_range", torch.arange(0, dim_h, 2)[: dim_h // 2].float() / dim_h)
        self.register_buffer("dim_temporal_range", torch.arange(0, dim_t, 2)[: dim_t // 2].float() / dim_t)
        self.max_h, self.max_w = len_h, len_w
        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def forward(self, T: int, H: int, W: int) -> torch.Tensor:
        """Rotation angles for a T x H x W token grid in (t, h, w) raster order: (T*H*W, 1, 1, head_dim), fp32."""
        assert H <= self.max_h and W <= self.max_w
        h_freqs = 1.0 / ((10000.0 * self.h_ntk_factor) ** self.dim_spatial_range.float())
        w_freqs = 1.0 / ((10000.0 * self.w_ntk_factor) ** self.dim_spatial_range.float())
        t_freqs = 1.0 / ((10000.0 * self.t_ntk_factor) ** self.dim_temporal_range.float())
        half_emb_h = torch.outer(self.seq[:H], h_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_freqs)
        half_emb_t = torch.outer(self.seq[:T], t_freqs)
        emb = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )
        return rearrange(emb, "t h w d -> (t h w) 1 1 d").float()


class Timesteps(nn.Module):
    """Sinusoidal timestep features (cos, sin), computed in fp32 and returned in the input dtype."""

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self, timesteps_B_T: torch.Tensor) -> torch.Tensor:
        in_dtype = timesteps_B_T.dtype
        timesteps = timesteps_B_T.flatten().float()
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - 0.0)
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        return rearrange(emb.to(dtype=in_dtype), "(b t) d -> b t d", b=timesteps_B_T.shape[0])


class TimestepEmbedding(nn.Module):
    """AdaLN-LoRA timestep embedding: returns the raw features and a (B, T, 3D) LoRA term shared by all blocks."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return sample, self.linear_2(self.activation(self.linear_1(sample)))


class PatchEmbed(nn.Module):
    """(B, C, T, H, W) -> (B, T, H/p, W/p, D) with a linear map of each p x p patch."""

    def __init__(self, patch: int, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            Rearrange("b c (t r) (h m) (w n) -> b t h w (c r m n)", r=1, m=patch, n=patch),
            nn.Linear(in_channels * patch * patch, out_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _adaln(hidden: int, lora_dim: int, chunks: int) -> nn.Sequential:
    return nn.Sequential(
        nn.SiLU(), nn.Linear(hidden, lora_dim, bias=False), nn.Linear(lora_dim, chunks * hidden, bias=False)
    )


def _modulate(norm: nn.Module, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return norm(x) * (1 + scale.to(x.dtype)) + shift.to(x.dtype)


class Block(nn.Module):
    """Self-attention, cross-attention and MLP, each with its own AdaLN-LoRA (shift, scale, gate) modulation."""

    def __init__(self, dim: int, context_dim: int, num_heads: int, mlp_ratio: float, adaln_lora_dim: int):
        super().__init__()
        head_dim = dim // num_heads
        self.layer_norm_self_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, None, num_heads, head_dim)
        self.layer_norm_cross_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, context_dim, num_heads, head_dim)
        self.layer_norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(dim, int(dim * mlp_ratio))
        self.adaln_modulation_self_attn = _adaln(dim, adaln_lora_dim, 3)
        self.adaln_modulation_cross_attn = _adaln(dim, adaln_lora_dim, 3)
        self.adaln_modulation_mlp = _adaln(dim, adaln_lora_dim, 3)

    def modulation(self, emb: torch.Tensor, lora: torch.Tensor, token_group: torch.Tensor, dtype: torch.dtype):
        """Per-group (shift, scale, gate) for the three sub-layers, gathered to tokens: 3 tuples of (B, S, D)."""
        out = []
        with _no_autocast(emb.device):
            for mod in (self.adaln_modulation_self_attn, self.adaln_modulation_cross_attn, self.adaln_modulation_mlp):
                out.append(tuple(t[:, token_group].to(dtype) for t in (mod(emb) + lora).chunk(3, dim=-1)))
        return out

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        rope: torch.Tensor,
        emb: torch.Tensor,
        lora: torch.Tensor,
        token_group: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, S, D) tokens; context: (B, N, ctx_dim); rope: (S, 1, 1, Dh); emb: (B, G, D) and lora: (B, G, 3D)
        timestep embeddings per group; token_group: (S,) group index of every token."""
        (sh, sc, g), (sh_c, sc_c, g_c), (sh_m, sc_m, g_m) = self.modulation(emb, lora, token_group, x.dtype)
        x = x + g.to(x.dtype) * self.self_attn(_modulate(self.layer_norm_self_attn, x, sh, sc), None, rope)
        x = x + g_c.to(x.dtype) * self.cross_attn(_modulate(self.layer_norm_cross_attn, x, sh_c, sc_c), context)
        x = x + g_m.to(x.dtype) * self.mlp(_modulate(self.layer_norm_mlp, x, sh_m, sc_m))
        return x


class FinalLayer(nn.Module):
    """AdaLN-modulated LayerNorm and a linear map from each token to its p x p x C latent patch."""

    def __init__(self, hidden_size: int, patch: int, out_channels: int, adaln_lora_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch * patch * out_channels, bias=False)
        self.adaln_modulation = _adaln(hidden_size, adaln_lora_dim, 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor, lora: torch.Tensor, token_group: torch.Tensor):
        with _no_autocast(emb.device):
            shift, scale = (self.adaln_modulation(emb) + lora[:, :, : 2 * self.hidden_size]).chunk(2, dim=-1)
            shift, scale = shift[:, token_group], scale[:, token_group]
        return self.linear(_modulate(self.layer_norm, x, shift, scale))


class CosmosDiT(nn.Module):
    """The Cosmos-Predict2 2B DiT (`MinimalV1LVGDiT`), with the hyper-parameters of Cosmos Policy as defaults.

    Input channels: 16 latent channels + 1 condition mask (1 = conditioning frame) + 1 padding mask (always 0).
    RoPE uses h/w extrapolation 3.0 and no FPS modulation, exactly as Cosmos Policy (see VideoRopePosition3DEmb: any
    other setting loads the same weights without error and silently breaks the network). SmolWAM drives the blocks
    directly (see `smolwam.model`), so this module only owns the weights and the shared embeddings; it has no
    video-level forward.
    """

    def __init__(
        self,
        num_blocks: int = 28,
        model_dim: int = 2048,
        num_heads: int = 16,
        ctx_dim: int = 1024,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        latent_channels: int = 16,
        patch: int = 2,
        max_img_hw: int = 240,
        max_frames: int = 128,
        rope_hw_extrapolation: float = 3.0,
    ):
        super().__init__()
        self.patch = patch
        self.x_embedder = PatchEmbed(patch, latent_channels + 2, model_dim)
        self.pos_embedder = VideoRopePosition3DEmb(
            head_dim=model_dim // num_heads,
            len_h=max_img_hw // patch,
            len_w=max_img_hw // patch,
            len_t=max_frames,
            h_extrapolation_ratio=rope_hw_extrapolation,
            w_extrapolation_ratio=rope_hw_extrapolation,
            t_extrapolation_ratio=1.0,
        )
        self.t_embedder = nn.Sequential(Timesteps(model_dim), TimestepEmbedding(model_dim))
        self.blocks = nn.ModuleList(
            [Block(model_dim, ctx_dim, num_heads, mlp_ratio, adaln_lora_dim) for _ in range(num_blocks)]
        )
        self.final_layer = FinalLayer(model_dim, patch, latent_channels, adaln_lora_dim)
        self.t_embedding_norm = RMSNorm(model_dim, eps=1e-6)

    def embed_timesteps(self, timesteps_B_G: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, G) rectified-flow timesteps -> normalised embedding (B, G, D) and AdaLN-LoRA term (B, G, 3D)."""
        with _no_autocast(timesteps_B_G.device):
            emb, lora = self.t_embedder(timesteps_B_G)
            return self.t_embedding_norm(emb), lora

    def unpatchify(self, x_B_T_H_W_M: torch.Tensor) -> torch.Tensor:
        """(B, T, H/p, W/p, p*p*C) final-layer outputs -> (B, C, T, H, W) latents."""
        return rearrange(
            x_B_T_H_W_M, "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)", p1=self.patch, p2=self.patch, t=1
        )
