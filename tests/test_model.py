import json
from pathlib import Path

import torch
from conftest import fake_batch

from smolwam.config import SmolWAMConfig
from smolwam.layout import (
    ACTION_DIM,
    CAMERAS,
    CHUNK,
    COMPACT_SLOT_TOKENS,
    LATENT_CH,
    LATENT_HW,
    PROPRIO_DIM,
    TEACHER_SLOT_TOKENS,
    patchify_tokens,
    tile_vector,
    untile_tokens,
    vector_readout_index,
)
from smolwam.model import SmolWAMModel
from smolwam.train.train import compute_losses

DATA = Path(__file__).parent / "data"


def test_forward_compact_and_teacher_layouts(tiny_config):
    torch.manual_seed(0)
    model = SmolWAMModel(tiny_config).eval()
    b = fake_batch(tiny_config)
    with torch.no_grad():
        out = model(model.to_trunk_space(b["lats"]), b["text_feat"], b["text_mask"], b["proprio"])
    assert model.num_tokens() == 1008
    assert set(out) == {"actions", "fut_latents"}
    assert out["actions"].shape == (2, CHUNK, ACTION_DIM)
    assert out["fut_latents"].shape == (2, len(CAMERAS), LATENT_CH, LATENT_HW, LATENT_HW)
    model.set_slot_tokens(TEACHER_SLOT_TOKENS)
    with torch.no_grad():
        out = model(model.to_trunk_space(b["lats"]), b["text_feat"], b["text_mask"], b["proprio"])
    assert model.num_tokens() == 1764
    assert out["fut_proprio"].shape == (2, PROPRIO_DIM) and out["value"].shape == (2,)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_new_modules_start_as_identity(tiny_config):
    """Zero-initialised V->L layers, state MLP and queries: the pretrained function is unchanged at initialisation."""
    torch.manual_seed(0)
    model = SmolWAMModel(tiny_config).eval()
    b = fake_batch(tiny_config)
    lats = model.to_trunk_space(b["lats"])
    with torch.no_grad():
        ref = model(lats, b["text_feat"], b["text_mask"], b["proprio"])
        other = model(lats, b["text_feat"], b["text_mask"], torch.zeros_like(b["proprio"]) + 0.3)
    # in the compact layout the state enters only through the zero-initialised state MLP
    assert torch.equal(ref["actions"], other["actions"])


def test_one_training_step_reduces_loss(tiny_config):
    torch.manual_seed(0)
    model = SmolWAMModel(tiny_config).train()
    b = fake_batch(tiny_config)
    b["lats"], b["fut_lats"] = model.to_trunk_space(b["lats"]), model.to_trunk_space(b["fut_lats"])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    losses = []
    for _ in range(4):
        loss, _ = compute_losses(model(b["lats"], b["text_feat"], b["text_mask"], b["proprio"]), b)
        opt.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.adapter.inp.weight.grad is None  # the latent adapters stay frozen
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_gradient_checkpointing_matches(tiny_config):
    torch.manual_seed(0)
    model = SmolWAMModel(tiny_config).train()
    b = fake_batch(tiny_config)
    lats = model.to_trunk_space(b["lats"])
    grads = []
    for ckpt in (False, True):
        model.grad_checkpointing = ckpt
        model.zero_grad()
        model(lats, b["text_feat"], b["text_mask"], b["proprio"])["actions"].sum().backward()
        grads.append(model.trunk.blocks[0].mlp.layer1.weight.grad.clone())
    torch.testing.assert_close(grads[0], grads[1])


def test_released_parameter_names_and_shapes():
    """The released checkpoint loads by name: the full-size network must keep every name and shape."""
    with torch.device("meta"):
        sd = SmolWAMModel(SmolWAMConfig()).state_dict()
    expected = json.loads((DATA / "release_state_dict.json").read_text())
    assert {k: list(v.shape) for k, v in sd.items()} == expected
    assert SmolWAMConfig().slot_tokens == COMPACT_SLOT_TOKENS


def test_vector_tiling_roundtrip():
    v = torch.randn(3, CHUNK * ACTION_DIM)
    tokens = patchify_tokens(tile_vector(v))
    for n_tokens in (28, 196):
        out = untile_tokens(tokens[:, :n_tokens], vector_readout_index(n_tokens, v.shape[1]), v.shape[1])
        torch.testing.assert_close(out, v)
