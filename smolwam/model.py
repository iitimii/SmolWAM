"""The SmolWAM network: Cosmos Policy's pretrained DiT trunk behind a rebuilt, single-pass inference interface.

One forward pass maps (camera latents, instruction features, robot state) to a 16-step action chunk and the latents of
both camera views 16 steps ahead (the world head). Relative to Cosmos Policy 2B (the teacher):

  * vision: each camera image is encoded once as a single-frame Wan VAE latent and mapped by the frozen latent
    adapter into the clip-mode latent space the trunk was trained on (`smolwam.nn.adapter`);
  * language: cached BERT-base token features, projected to the trunk's context width and zero-padded to 512
    positions, because the zero positions act as the trunk's attention sink; zero-initialised vision-to-language
    layers (`TextUpdate`) let the instruction attend to the scene;
  * depth: 12 of the 28 pretrained blocks (`SmolWAMConfig.trunk_blocks`);
  * tokens: the teacher's latent-slot sequence is kept, with its RoPE positions, but the proprio, future-proprio and
    value slots are dropped and the action slot keeps 28 learned query tokens to which an MLP of the robot state is
    added (1,008 tokens instead of 1,764);
  * readout: the trunk runs once at the first sampling noise level with a fixed noise draw, and actions and future
    latents are read from the rectified-flow clean estimate.

Rectified-flow preconditioning (Cosmos-Predict2, verified against the teacher): t = sigma / (1 + sigma); the network
input on a generated slot is (1 - t) x_t, which at the first sampling step is t * eps; the clean estimate is
x0 = net_in - t * F, with F the network output. Conditioning slots get t = 0 and their raw latents.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from smolwam.config import SmolWAMConfig
from smolwam.layout import (
    ACTION_DIM,
    ACTION_SLOT,
    CHUNK,
    FUTURE_IMAGE_SLOTS,
    GEN_SLOTS,
    IMAGE_SLOTS,
    LATENT_CH,
    LATENT_HW,
    PATCH,
    PROPRIO_DIM,
    SLOTS,
    TOKENS_PER_SLOT,
    VECTOR_SLOTS,
    fixed_noise,
    patchify_tokens,
    tile_vector,
    untile_tokens,
    vector_readout_index,
)
from smolwam.nn.adapter import LatentAdapter, inverse_adapter
from smolwam.nn.dit import CosmosDiT

TEXT_INIT_SCALE = 0.125 / 0.58  # projected BERT features (std ~0.58) start at the scale of T5-XXL tokens (std 0.125)


class TextUpdate(nn.Module):
    """Vision-to-language cross-attention + FFN on the instruction context (TurboVLA-style bidirectional interaction;
    the pretrained blocks only do language-to-vision). Output projections are zero-initialised, so the pretrained
    function is unchanged at initialisation."""

    def __init__(self, ctx_dim: int, vis_dim: int, inner: int, heads: int, ffn: int):
        super().__init__()
        self.heads = heads
        self.ln_q = nn.LayerNorm(ctx_dim)
        self.ln_kv = nn.LayerNorm(vis_dim)
        self.q = nn.Linear(ctx_dim, inner, bias=False)
        self.k = nn.Linear(vis_dim, inner, bias=False)
        self.v = nn.Linear(vis_dim, inner, bias=False)
        self.o = nn.Linear(inner, ctx_dim, bias=False)
        self.ln_f = nn.LayerNorm(ctx_dim)
        self.ffn = nn.Sequential(nn.Linear(ctx_dim, ffn), nn.GELU(), nn.Linear(ffn, ctx_dim))
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    def forward(self, ctx: torch.Tensor, vis: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.q(self.ln_q(ctx)), "b n (h d) -> b h n d", h=self.heads)
        kv = self.ln_kv(vis)
        k = rearrange(self.k(kv), "b s (h d) -> b h s d", h=self.heads)
        v = rearrange(self.v(kv), "b s (h d) -> b h s d", h=self.heads)
        ctx = ctx + self.o(rearrange(F.scaled_dot_product_attention(q, k, v), "b h n d -> b n (h d)"))
        ctx = ctx + self.ffn(self.ln_f(ctx))
        return ctx * text_mask.unsqueeze(-1).to(ctx.dtype)


def pad_context(ctx: torch.Tensor, n_total: int) -> torch.Tensor:
    """Zero-pad the cross-attention context to `n_total` positions. The trunk's cross-attention keys have no bias, so
    zero positions give zero keys: Cosmos Policy relies on its ~500 zero-padded T5 positions as an attention sink."""
    if ctx.shape[1] >= n_total:
        return ctx
    return torch.cat([ctx, ctx.new_zeros(ctx.shape[0], n_total - ctx.shape[1], ctx.shape[2])], 1)


class SmolWAMModel(nn.Module):
    """The SmolWAM network (trunk + text projection + V->L layers + state MLP + queries + latent adapters)."""

    def __init__(self, cfg: SmolWAMConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.model_dim
        self.trunk = CosmosDiT(len(cfg.trunk_blocks), D, cfg.num_heads, cfg.ctx_dim, cfg.mlp_ratio, cfg.adaln_lora_dim)
        # generated slots: fixed noise + learned additive queries (zero-init: the teacher's input at initialisation)
        self.register_buffer("eps_gen", fixed_noise(cfg.noise_seed, len(GEN_SLOTS)))
        self.register_buffer("blank", torch.zeros(LATENT_CH, LATENT_HW, LATENT_HW))  # VAE latent of a black frame
        self.gen_query = nn.Parameter(torch.zeros(len(GEN_SLOTS), TOKENS_PER_SLOT, D))
        self.text_proj = nn.Sequential(nn.LayerNorm(cfg.text_dim), nn.Linear(cfg.text_dim, cfg.ctx_dim))
        with torch.no_grad():
            self.text_proj[1].weight.mul_(TEXT_INIT_SCALE)
            self.text_proj[1].bias.zero_()
        n = len(cfg.trunk_blocks)
        steps = torch.linspace(0, n - 1, cfg.n_bidir + 1, device="cpu")[:-1].tolist() if cfg.n_bidir > 0 else []
        self.bidir_at = sorted({int(round(i)) for i in steps})  # after which kept blocks the V->L layers run
        self.bidir = nn.ModuleList(
            [TextUpdate(cfg.ctx_dim, D, cfg.bidir_inner, cfg.bidir_heads, cfg.bidir_ffn) for _ in self.bidir_at]
        )
        self.state_mlp = nn.Sequential(nn.Linear(PROPRIO_DIM, D), nn.GELU(), nn.Linear(D, D))
        nn.init.zeros_(self.state_mlp[2].weight)
        nn.init.zeros_(self.state_mlp[2].bias)
        self.adapter = LatentAdapter(cfg.adapter_width, len(IMAGE_SLOTS)).requires_grad_(False)
        self.adapter_inv = inverse_adapter(cfg.adapter_width, len(IMAGE_SLOTS)).requires_grad_(False)
        self.slot_tokens = tuple(cfg.slot_tokens)
        self.grad_checkpointing = False
        self._rope_cache: dict = {}
        self._readout_cache: dict = {}

    # ------------------------------------------------------------------------------------------------ layout
    def set_slot_tokens(self, slot_tokens: tuple[int, ...]) -> None:
        """Change the token layout (the training curriculum starts in the teacher's layout)."""
        self.slot_tokens = tuple(int(x) for x in slot_tokens)

    def slot_ntok(self) -> dict[int, int]:
        """Tokens kept per slot index: vector slots from `slot_tokens`, image slots always 196."""
        return {
            i: self.slot_tokens[VECTOR_SLOTS.index(name)] if name in VECTOR_SLOTS else TOKENS_PER_SLOT
            for i, name in enumerate(SLOTS)
        }

    def num_tokens(self) -> int:
        return sum(self.slot_ntok().values())

    def rope(self, device: torch.device) -> torch.Tensor:
        """RoPE angles of the kept tokens, at their positions in the teacher's full sequence: (S, 1, 1, head_dim)."""
        dtype = self.trunk.pos_embedder.seq.dtype
        key = (self.slot_tokens, str(device), dtype)
        if key not in self._rope_cache:
            hw = LATENT_HW // PATCH
            full = self.trunk.pos_embedder(len(SLOTS), hw, hw).reshape(len(SLOTS), hw * hw, 1, 1, -1)
            nt = self.slot_ntok()
            self._rope_cache = {key: torch.cat([full[s][: nt[s]] for s in range(len(SLOTS)) if nt[s] > 0]).contiguous()}
        return self._rope_cache[key]

    def readout_index(self, n_tokens: int, d: int, device: torch.device) -> torch.Tensor:
        key = (n_tokens, d, str(device))
        if key not in self._readout_cache:
            self._readout_cache[key] = vector_readout_index(n_tokens, d).to(device)
        return self._readout_cache[key]

    # ------------------------------------------------------------------------------------------------ latent spaces
    @torch.no_grad()
    def _adapt(self, adapter: LatentAdapter, lats: torch.Tensor) -> torch.Tensor:
        B, C = lats.shape[:2]
        cam = torch.arange(C, device=lats.device).repeat(B)
        z = adapter(lats.float().reshape(B * C, LATENT_CH, LATENT_HW, LATENT_HW), cam)
        return z.reshape(B, C, LATENT_CH, LATENT_HW, LATENT_HW).to(lats.dtype)

    def to_trunk_space(self, lats: torch.Tensor) -> torch.Tensor:
        """Single-frame VAE latents (B, n_cams, 16, 28, 28) -> clip-mode latents the trunk expects."""
        return self._adapt(self.adapter, lats)

    def to_single_space(self, lats: torch.Tensor) -> torch.Tensor:
        """Trunk-space latents (B, n_cams, 16, 28, 28), e.g. predicted futures -> single-frame VAE latents."""
        return self._adapt(self.adapter_inv, lats)

    # ------------------------------------------------------------------------------------------------ forward
    def context(self, text_feat: torch.Tensor, text_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        ctx = self.text_proj(text_feat) if self.cfg.text_projection else text_feat
        return (ctx * text_mask.unsqueeze(-1).to(text_feat.dtype)).to(dtype)

    def forward(
        self, latents: torch.Tensor, text_feat: torch.Tensor, text_mask: torch.Tensor, proprio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """One pass of the network.

        latents: (B, 2, 16, 28, 28) trunk-space latents of the wrist and primary cameras; text_feat: (B, N, text_dim);
        text_mask: (B, N) bool; proprio: (B, 9) normalised robot state.
        Returns actions (B, 16, 7) normalised, fut_latents (B, 2, 16, 28, 28) in trunk space, and fut_proprio (B, 9)
        and value (B,) when those slots are in the layout.
        """
        B, dev = latents.shape[0], latents.device
        pdtype = self.trunk.x_embedder.proj[1].weight.dtype
        hw = LATENT_HW // PATCH
        nt = self.slot_ntok()
        t = self.cfg.t_gen
        n_slots = len(SLOTS)
        # latent volume of network inputs: conditioning slots raw, generated slots t * eps
        x = torch.zeros(B, LATENT_CH, n_slots, LATENT_HW, LATENT_HW, device=dev, dtype=torch.float32)
        x[:, :, SLOTS.index("blank")] = self.blank.float()
        x[:, :, SLOTS.index("proprio")] = tile_vector(proprio.float())
        for c, sl in enumerate(IMAGE_SLOTS):
            x[:, :, sl] = latents[:, c].float()
        for k, sl in enumerate(GEN_SLOTS):
            x[:, :, sl] = t * self.eps_gen[k].float()
        cond_mask = torch.zeros(B, 1, n_slots, LATENT_HW, LATENT_HW, device=dev)
        cond_mask[:, :, :ACTION_SLOT] = 1.0
        xin = torch.cat([x, cond_mask, torch.zeros_like(cond_mask)], 1).to(pdtype)  # + padding-mask channel (0)
        emb = self.trunk.x_embedder(xin).reshape(B, n_slots, hw * hw, -1)
        state = self.state_mlp(proprio.to(pdtype)).unsqueeze(1)
        toks, groups, offsets, pos = [], [], {}, 0
        for sl in range(n_slots):
            n = nt[sl]
            if n == 0:
                continue
            e = emb[:, sl, :n]
            if sl in GEN_SLOTS:
                e = e + self.gen_query[GEN_SLOTS.index(sl), :n].unsqueeze(0).to(e.dtype)
                if sl in (ACTION_SLOT, SLOTS.index("fut_proprio")):
                    e = e + state
            toks.append(e)
            groups.append(torch.full((n,), sl, device=dev, dtype=torch.long))
            offsets[sl] = (pos, pos + n)
            pos += n
        h = torch.cat(toks, 1)
        token_group = torch.cat(groups)
        rope = self.rope(dev)
        timesteps = torch.tensor([0.0] * ACTION_SLOT + [t] * len(GEN_SLOTS), device=dev, dtype=torch.float32)
        emb_t, lora_t = self.trunk.embed_timesteps(timesteps.unsqueeze(0).expand(B, -1).to(pdtype))
        ctx = self.context(text_feat, text_mask, pdtype)
        img_tokens = slice(offsets[IMAGE_SLOTS[0]][0], offsets[IMAGE_SLOTS[-1]][1])
        bidir = {blk: k for k, blk in enumerate(self.bidir_at)}
        for i, block in enumerate(self.trunk.blocks):
            ctx_in = pad_context(ctx, self.cfg.ctx_pad_to)
            if self.grad_checkpointing and self.training:
                h = checkpoint(block, h, ctx_in, rope, emb_t, lora_t, token_group, use_reentrant=False)
            else:
                h = block(h, ctx_in, rope, emb_t, lora_t, token_group)
            if i in bidir:
                ctx = self.bidir[bidir[i]](ctx, h[:, img_tokens], text_mask)
        o = self.trunk.final_layer(h, emb_t, lora_t, token_group).float()  # (B, S, 64)

        def x0(sl: int) -> torch.Tensor:
            """One-step rectified-flow estimate on a generated slot: x0 = net_in - t * F (token output order)."""
            k = GEN_SLOTS.index(sl)
            a, b = offsets[sl]
            eps_tok = patchify_tokens(self.eps_gen[k : k + 1].float())[:, : b - a]
            return t * eps_tok - t * o[:, a:b]

        out = {}
        n_act = CHUNK * ACTION_DIM
        xa = x0(ACTION_SLOT)
        out["actions"] = untile_tokens(xa, self.readout_index(xa.shape[1], n_act, dev), n_act).reshape(
            B, CHUNK, ACTION_DIM
        )
        futs = [self.trunk.unpatchify(x0(sl).reshape(B, 1, hw, hw, -1))[:, :, 0] for sl in FUTURE_IMAGE_SLOTS]
        out["fut_latents"] = torch.stack(futs, 1)
        sl_fp = SLOTS.index("fut_proprio")
        if nt[sl_fp] > 0:
            xp = x0(sl_fp)
            out["fut_proprio"] = untile_tokens(xp, self.readout_index(xp.shape[1], PROPRIO_DIM, dev), PROPRIO_DIM)
        if nt[SLOTS.index("value")] > 0:
            out["value"] = x0(SLOTS.index("value")).mean(dim=(1, 2))
        return out

    # ------------------------------------------------------------------------------------------------ weights
    def load_cosmos_trunk(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        """Copy the kept blocks and the shared trunk modules from a Cosmos Policy checkpoint (keys `net.*`).
        Returns the trunk keys that were not found."""
        keep = {j: i for i, j in enumerate(self.cfg.trunk_blocks)}
        sd = {}
        for k, v in state_dict.items():
            if not k.startswith("net.") or k.endswith("_extra_state") or k.startswith("net.accum_"):
                continue
            k = k[len("net.") :]
            if k.startswith("blocks."):
                j = int(k.split(".")[1])
                if j not in keep:
                    continue
                k = f"blocks.{keep[j]}" + k[len(f"blocks.{j}") :]
            sd[k] = v
        return self.trunk.load_state_dict(sd, strict=False).missing_keys

    def num_params(self) -> int:
        """Network parameters, excluding the frozen latent adapters (counted with the vision encoder)."""
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("adapter"))
