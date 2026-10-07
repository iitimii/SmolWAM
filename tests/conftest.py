import pytest
import torch

from smolwam.config import SmolWAMConfig
from smolwam.layout import ACTION_DIM, CAMERAS, CHUNK, LATENT_CH, LATENT_HW, PROPRIO_DIM


@pytest.fixture
def tiny_config() -> SmolWAMConfig:
    """Same structure as the released network (sequence, layouts, adapters, V->L layers), tiny widths."""
    return SmolWAMConfig(
        trunk_blocks=(0, 1),
        model_dim=64,
        num_heads=2,
        ctx_dim=32,
        adaln_lora_dim=16,
        text_dim=24,
        max_text_tokens=8,
        ctx_pad_to=16,
        n_bidir=2,
        bidir_inner=16,
        bidir_heads=2,
        bidir_ffn=32,
        adapter_width=16,
    )


def fake_batch(cfg: SmolWAMConfig, batch_size: int = 2, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(batch_size, cfg.max_text_tokens, dtype=torch.bool)
    mask[:, :5] = True
    lat_shape = (batch_size, len(CAMERAS), LATENT_CH, LATENT_HW, LATENT_HW)
    return {
        "lats": torch.randn(*lat_shape, generator=g),
        "fut_lats": torch.randn(*lat_shape, generator=g),
        "proprio": torch.rand(batch_size, PROPRIO_DIM, generator=g) * 2 - 1,
        "fut_proprio": torch.rand(batch_size, PROPRIO_DIM, generator=g) * 2 - 1,
        "actions": torch.rand(batch_size, CHUNK, ACTION_DIM, generator=g) * 2 - 1,
        "value": torch.rand(batch_size, generator=g) * 2 - 1,
        "text_feat": torch.randn(batch_size, cfg.max_text_tokens, cfg.text_dim, generator=g) * mask.unsqueeze(-1),
        "text_mask": mask,
    }
