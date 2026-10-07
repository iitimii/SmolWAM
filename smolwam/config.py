"""SmolWAM network configuration. The defaults describe the released model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from smolwam.layout import COMPACT_SLOT_TOKENS, NOISE_SEED, SIGMA_MAX, TEACHER_SLOT_TOKENS

RELEASED_BLOCKS = (0, 1, 2, 19, 20, 21, 22, 23, 24, 25, 26, 27)


@dataclass
class SmolWAMConfig:
    """Architecture of a SmolWAM network.

    trunk_blocks: indices of the Cosmos Policy DiT blocks kept, in order (12 of 28). Pruning keeps the first and last
        blocks: blocks 3-22 are individually near-identity, but removing block 1, 26 or 27 fails every episode.
    text_*, ctx_pad_to: the instruction is encoded by a small text encoder (BERT-base, <= 32 tokens), projected to the
        trunk's 1,024-d cross-attention context and zero-padded to `ctx_pad_to` positions. The trunk's cross-attention
        keys have no bias, so Cosmos Policy's ~500 zero-padded T5 positions act as an attention sink; truncating the
        context to 32 or 128 positions makes the teacher fail every episode, so the padding stays.
    text_projection: False feeds a precomputed 1,024-d context straight into cross-attention (Cosmos Policy's T5-XXL
        embeddings; only used to run the teacher in this code for distillation targets).
    n_bidir, bidir_*: zero-initialised vision-to-language layers that let the instruction attend to the image tokens.
    slot_tokens: tokens kept for the vector slots (blank, proprio, action, fut_proprio, value); 0 drops the slot.
        Image slots always keep 196 tokens. Training starts in the teacher's layout and switches to this one.
    sigma_gen, noise_seed: generated slots are read out with the one-step rectified-flow estimate at this noise level,
        from a fixed noise draw.
    """

    trunk_blocks: tuple[int, ...] = RELEASED_BLOCKS
    model_dim: int = 2048
    num_heads: int = 16
    ctx_dim: int = 1024
    mlp_ratio: float = 4.0
    adaln_lora_dim: int = 256
    text_encoder: str = "bert-base-uncased"
    text_dim: int = 768
    max_text_tokens: int = 32
    ctx_pad_to: int = 512
    text_projection: bool = True
    n_bidir: int = 6
    bidir_inner: int = 256
    bidir_heads: int = 4
    bidir_ffn: int = 1024
    slot_tokens: tuple[int, ...] = COMPACT_SLOT_TOKENS
    adapter_width: int = 96
    sigma_gen: float = SIGMA_MAX
    noise_seed: int = NOISE_SEED

    @property
    def t_gen(self) -> float:
        """Rectified-flow timestep of the generated slots: t = sigma / (1 + sigma)."""
        return self.sigma_gen / (1.0 + self.sigma_gen)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["trunk_blocks"] = list(self.trunk_blocks)
        d["slot_tokens"] = list(self.slot_tokens)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> SmolWAMConfig:
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown SmolWAMConfig fields: {sorted(unknown)}")
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    @classmethod
    def cosmos_teacher(cls) -> SmolWAMConfig:
        """Cosmos Policy 2B itself, run by SmolWAM's single-pass code: all 28 blocks, its own T5-XXL context and
        token layout. Used to compute the distillation targets (`smolwam.train.kd_targets`)."""
        return cls(
            trunk_blocks=tuple(range(28)),
            text_encoder="t5-xxl (precomputed)",
            text_dim=1024,
            max_text_tokens=512,
            text_projection=False,
            n_bidir=0,
            slot_tokens=TEACHER_SLOT_TOKENS,
        )
