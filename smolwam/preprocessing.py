"""Observation preprocessing and action / proprio normalisation, shared by the policy and the data pipeline.

Images follow Cosmos Policy's deterministic test-time transform: JPEG round trip (quality 95, as the training data was
stored as JPEG), resize to 224 px, centre crop covering 90 % of the area, and resize back to 224 px (bilinear,
antialiased). Training frames come from the dataset's JPEGs and skip the round trip.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from smolwam.layout import IMAGE_SIZE

CROP_AREA = 0.9


def decode_jpeg(data) -> np.ndarray:
    """JPEG bytes (or a uint8 array of them) -> (H, W, 3) uint8 RGB."""
    return np.asarray(Image.open(io.BytesIO(np.asarray(data).tobytes())).convert("RGB"))


def encode_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def resize_lanczos(img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return np.asarray(Image.fromarray(img).resize((size, size), Image.LANCZOS))


def center_crop_resize(img: np.ndarray, area: float = CROP_AREA) -> np.ndarray:
    """Centre crop covering `area` of a square image, resized back to its size (bilinear, antialiased).

    Bit-identical to torchvision's `center_crop` followed by `resize(..., antialias=True)` on a uint8 tensor.
    """
    h = img.shape[0]
    crop = int(h * area**0.5)
    top = int(round((h - crop) / 2.0))
    x = torch.from_numpy(np.ascontiguousarray(img[top : top + crop, top : top + crop])).permute(2, 0, 1).contiguous()
    x = F.interpolate(x[None].float(), size=[h, h], mode="bilinear", align_corners=False, antialias=True)[0]
    return torch.round(x).to(torch.uint8).permute(1, 2, 0).numpy()


def preprocess_env_frame(img: np.ndarray) -> np.ndarray:
    """Upright camera frame from the simulator (H, W, 3) uint8 -> 224 px model input, uint8."""
    img = np.asarray(Image.open(io.BytesIO(encode_jpeg(np.ascontiguousarray(img)))).convert("RGB"))
    return center_crop_resize(resize_lanczos(img))


def preprocess_dataset_frame(jpeg) -> np.ndarray:
    """Stored dataset JPEG -> 224 px model input, uint8."""
    return center_crop_resize(resize_lanczos(decode_jpeg(jpeg)))


def frames_to_tensor(frames: np.ndarray, device: str | torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, 1, H, W) one-frame videos in [-1, 1], scaled in `dtype` on `device`."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).to(device).permute(0, 3, 1, 2).to(dtype) / 127.5 - 1.0
    return x.unsqueeze(2)


def load_norm_stats(path: str | Path) -> dict:
    """Min / max of actions and proprio from a LIBERO-Cosmos-Policy `dataset_statistics.json`."""
    stats = json.loads(Path(path).read_text())
    return {k: {"min": stats[f"{k}_min"], "max": stats[f"{k}_max"]} for k in ("actions", "proprio")}


class Normalizer:
    """Min-max normalisation of actions and proprio to [-1, 1] with the training-set statistics."""

    def __init__(self, stats: dict):
        """stats: {"actions": {"min": [...], "max": [...]}, "proprio": {"min": [...], "max": [...]}}."""
        self.stats = {k: {b: np.asarray(v[b], dtype=np.float32) for b in ("min", "max")} for k, v in stats.items()}

    def normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        lo, hi = self.stats[key]["min"], self.stats[key]["max"]
        return 2.0 * (x - lo) / (hi - lo) - 1.0

    def unnormalize(self, x: np.ndarray, key: str) -> np.ndarray:
        lo, hi = self.stats[key]["min"], self.stats[key]["max"]
        return 0.5 * (x + 1.0) * (hi - lo) + lo

    def to_dict(self) -> dict:
        return {k: {b: v[b].tolist() for b in ("min", "max")} for k, v in self.stats.items()}
