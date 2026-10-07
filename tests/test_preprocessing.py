import json
from pathlib import Path

import numpy as np
import pytest
import torch

from smolwam.nn.wan_vae import WanVAE
from smolwam.preprocessing import Normalizer, center_crop_resize, preprocess_env_frame

SAMPLE = Path(__file__).parents[1] / "examples" / "sample_obs"


def test_center_crop_matches_torchvision():
    tf = pytest.importorskip("torchvision.transforms.functional")
    rng = np.random.default_rng(0)
    for _ in range(20):
        img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
        t = tf.center_crop(torch.from_numpy(img).permute(2, 0, 1), int(224 * 0.9**0.5))
        ref = tf.resize(t, [224, 224], antialias=True).permute(1, 2, 0).numpy()
        assert np.array_equal(center_crop_resize(img), ref)


def test_env_frame_preprocessing_shape():
    from PIL import Image

    out = preprocess_env_frame(np.asarray(Image.open(SAMPLE / "primary.png").convert("RGB")))
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


def test_normalizer_roundtrip():
    stats = {"actions": {"min": [-1.0, -0.5], "max": [1.0, 0.25]}, "proprio": {"min": [0.0], "max": [2.0]}}
    norm = Normalizer(stats)
    x = np.array([[0.3, -0.1]], dtype=np.float32)
    np.testing.assert_allclose(norm.unnormalize(norm.normalize(x, "actions"), "actions"), x, atol=1e-6)
    assert norm.normalize(np.array([2.0], np.float32), "proprio")[0] == 1.0
    assert json.loads(json.dumps(norm.to_dict())) == stats


def test_vae_shapes_single_frame_and_clip():
    torch.manual_seed(0)
    vae = WanVAE().eval()
    for frames, latent_frames in ((1, 1), (5, 2), (9, 3)):  # the VAE takes 1 + 4k frames
        z = vae.encode(torch.rand(1, 3, frames, 32, 32) * 2 - 1)
        assert z.shape == (1, 16, latent_frames, 4, 4)
        assert vae.decode(z).shape == (1, 3, 1 + 4 * (latent_frames - 1), 32, 32)
