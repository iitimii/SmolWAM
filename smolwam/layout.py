"""The latent-frame ("slot") layout SmolWAM inherits from Cosmos Policy on LIBERO, and helpers that depend only on it.

Cosmos Policy treats every input and output as a 16 x 28 x 28 latent frame of one sequence. On LIBERO the nine slots
are, in order:

    blank, proprio, wrist, primary | action, fut_proprio, fut_wrist, fut_primary, value
    ------ conditioning ---------- | ---------------- generated ------------------------

Conditioning slots hold clean latents (rectified-flow time 0). Generated slots start from noise; the trunk predicts
them in one pass. Vectors (proprio, the 16 x 7 action chunk, value) are written into a slot by tiling them over the
whole 16 x 28 x 28 volume (`tile_vector`) and read back by averaging all copies (`untile_tokens`). Each slot is
patchified into 14 x 14 = 196 tokens.

SmolWAM keeps this sequence (the pretrained trunk depends on the token positions) but may keep only some tokens of
the vector slots (`slot_tokens`): the released model keeps the blank frame, the two camera frames, 28 action query
tokens and the two future camera frames, 1,008 tokens in all, against the teacher's 1,764.
"""

from __future__ import annotations

import torch
from einops import rearrange

SLOTS = ("blank", "proprio", "wrist", "primary", "action", "fut_proprio", "fut_wrist", "fut_primary", "value")
CAMERAS = ("wrist", "primary")  # in the order of their slots
VECTOR_SLOTS = ("blank", "proprio", "action", "fut_proprio", "value")  # order of `slot_tokens`
TEACHER_SLOT_TOKENS = (196, 196, 196, 196, 196)  # Cosmos Policy's own layout: every slot complete (1,764 tokens)
COMPACT_SLOT_TOKENS = (196, 0, 28, 0, 0)  # the released layout: no proprio / future-proprio / value slots

CHUNK = 16  # actions per query, all executed open-loop
ACTION_DIM = 7  # end-effector delta pose (6) + gripper
PROPRIO_DIM = 9  # gripper joint positions (2) + end-effector position (3) + quaternion (4)
IMAGE_SIZE = 224
LATENT_CH = 16
LATENT_HW = IMAGE_SIZE // 8
PATCH = 2
TOKENS_PER_SLOT = (LATENT_HW // PATCH) ** 2  # 196

ACTION_SLOT = SLOTS.index("action")
COND_SLOTS = tuple(range(ACTION_SLOT))
GEN_SLOTS = tuple(range(ACTION_SLOT, len(SLOTS)))
IMAGE_SLOTS = tuple(SLOTS.index(c) for c in CAMERAS)
FUTURE_IMAGE_SLOTS = tuple(SLOTS.index("fut_" + c) for c in CAMERAS)

# Rectified flow (Cosmos-Predict2): timestep t = sigma / (1 + sigma). Conditioning slots get t = 0 and raw latents.
# Generated slots are read out at the first sampling step, sigma = 80, from a fixed noise draw.
SIGMA_MAX = 80.0
NOISE_SEED = 1234


def fixed_noise(seed: int = NOISE_SEED, n_slots: int = len(GEN_SLOTS)) -> torch.Tensor:
    """(n_slots, 16, 28, 28) unit Gaussian noise for the generated slots, drawn once and kept fixed."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_slots, LATENT_CH, LATENT_HW, LATENT_HW, generator=g, device="cpu")


def tile_vector(v: torch.Tensor) -> torch.Tensor:
    """(B, d) -> (B, 16, 28, 28) by repeating the vector over the flattened volume (Cosmos Policy's injection)."""
    B, d = v.shape
    n = LATENT_CH * LATENT_HW * LATENT_HW
    return v.repeat(1, n // d + 1)[:, :n].reshape(B, LATENT_CH, LATENT_HW, LATENT_HW)


def patchify_tokens(x_B_C_H_W: torch.Tensor) -> torch.Tensor:
    """(B, 16, 28, 28) latent -> (B, 196, 64) tokens in the trunk's final-layer output order (p1 p2 C)."""
    return rearrange(x_B_C_H_W, "B C (H p1) (W p2) -> B (H W) (p1 p2 C)", p1=PATCH, p2=PATCH)


def vector_readout_index(n_tokens: int, d: int) -> torch.Tensor:
    """(n_tokens, 64) index into a tiled d-vector for each output element of the first `n_tokens` tokens of a slot.

    Latent element e = c * 784 + row * 28 + col holds v[e mod d]; output o = p1 * 32 + p2 * 16 + c of token
    k = (hp, wp) is the latent element (c, 2 hp + p1, 2 wp + p2).
    """
    idx = torch.zeros(n_tokens, 4 * LATENT_CH, dtype=torch.long)
    for k in range(n_tokens):
        hp, wp = divmod(k, LATENT_HW // PATCH)
        for p1 in range(PATCH):
            for p2 in range(PATCH):
                for c in range(LATENT_CH):
                    e = c * LATENT_HW * LATENT_HW + (2 * hp + p1) * LATENT_HW + (2 * wp + p2)
                    idx[k, p1 * 32 + p2 * 16 + c] = e % d
    return idx


def untile_tokens(x_tok: torch.Tensor, idx: torch.Tensor, d: int) -> torch.Tensor:
    """Average every copy of a tiled d-vector over token outputs x_tok (B, n, 64) -> (B, d)."""
    B, dev = x_tok.shape[0], x_tok.device
    ones = torch.ones(idx.numel(), device=dev, dtype=x_tok.dtype)
    summed = torch.zeros(B, d, device=dev, dtype=x_tok.dtype).index_add_(1, idx.reshape(-1), x_tok.reshape(B, -1))
    counts = torch.zeros(d, device=dev, dtype=x_tok.dtype).index_add_(0, idx.reshape(-1), ones)
    return summed / counts
