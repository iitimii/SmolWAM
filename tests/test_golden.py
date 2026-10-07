"""Regression test of the released weights: on the bundled LIBERO observation, the actions and predicted future
latents must match those of the original research implementation (fp32 network, bf16 VAE).

Reference values exist for the CPU and for Apple MPS; the test runs on MPS when available (seconds) and otherwise on
the CPU (several minutes: bf16 convolutions are slow there). Uses $SMOLWAM_MODEL (a local release directory or repo
id) or timiiowolabi/SmolWAM from the Hub, and skips when neither is reachable."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
GOLDEN = json.loads((Path(__file__).parent / "data" / "golden.json").read_text())


def released_model() -> str:
    from smolwam.hub import resolve_model_dir, wan_vae_file

    model = os.environ.get("SMOLWAM_MODEL", "timiiowolabi/SmolWAM")
    try:
        resolve_model_dir(model)
        wan_vae_file()
    except Exception as e:  # offline, or the repo is not reachable
        pytest.skip(f"released weights not available: {e}")
    return model


@pytest.mark.slow
def test_golden_actions():
    from smolwam import SmolWAM

    sys.path.insert(0, str(ROOT / "examples"))
    from quickstart import load_sample

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    golden = GOLDEN[device]
    policy = SmolWAM.from_pretrained(released_model(), device=device, dtype="fp32", vae_dtype="bf16")
    obs, instruction = load_sample()
    assert instruction == GOLDEN["instruction"]
    actions, future = policy.predict_action(obs, instruction, return_future=True)
    np.testing.assert_allclose(actions, np.array(golden["actions"]), atol=1e-4, rtol=0)
    np.testing.assert_allclose(future.cpu().flatten()[:16].numpy(), golden["future_latents_head"], atol=1e-4, rtol=0)
    np.testing.assert_allclose(float(future.abs().mean()), golden["future_latents_abs_mean"], rtol=1e-4)
