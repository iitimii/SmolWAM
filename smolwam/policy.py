"""The deployable SmolWAM policy: camera images + robot state + instruction -> a 16-step action chunk.

    from smolwam import SmolWAM

    policy = SmolWAM.from_pretrained("timiiowolabi/SmolWAM")       # or a local release directory
    actions = policy.predict_action(obs, "put the wine bottle on the rack")             # (16, 7) numpy
    actions, future = policy.predict_action(obs, instruction, return_future=True)      # + predicted latents
    frames = policy.decode_future(future)        # {"wrist_image": (224, 224, 3), "primary_image": ...} uint8

`obs` holds upright RGB frames from the wrist and third-person ("primary") cameras, any square size (LIBERO renders
256 px), and the 9-d robot state: gripper joint positions (2), end-effector position (3) and quaternion (4). The
returned actions are end-effector deltas (6) and a gripper command in {-1, +1}, in the units of the LIBERO demos.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from smolwam.config import SmolWAMConfig
from smolwam.hub import DEFAULT_MODEL, load_release, resolve_model_dir, wan_vae_file
from smolwam.layout import CAMERAS
from smolwam.model import SmolWAMModel
from smolwam.nn.wan_vae import WanVAE
from smolwam.preprocessing import Normalizer, frames_to_tensor, preprocess_env_frame
from smolwam.text import TextCache, TextEncoder

DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def resolve_dtype(dtype: str | torch.dtype | None, device: str | torch.device) -> torch.dtype:
    """bf16 on CUDA, fp32 elsewhere (on Apple silicon fp32 is also the faster choice) unless asked otherwise."""
    if dtype is None:
        return torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    return DTYPES[dtype] if isinstance(dtype, str) else dtype


def synchronize(device: str | torch.device) -> None:
    """Wait for queued kernels so wall-clock timings are real (CUDA and MPS run asynchronously)."""
    kind = torch.device(device).type
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def load_model(cfg: SmolWAMConfig, state_dict: dict[str, torch.Tensor], device, dtype: torch.dtype) -> SmolWAMModel:
    """Build the network without initialising weights, load them, and cast. The latent adapters stay in fp32."""
    with torch.device("meta"):
        model = SmolWAMModel(cfg)
    model.load_state_dict(state_dict, assign=True)
    model = model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    model.adapter.float()
    model.adapter_inv.float()
    return model


class SmolWAM:
    """SmolWAM policy. Use `SmolWAM.from_pretrained`; `last_timing` holds the seconds spent in the last query."""

    def __init__(
        self,
        model: SmolWAMModel,
        vae: WanVAE,
        normalizer: Normalizer,
        text_cache: TextCache,
        device: str | torch.device,
        dtype: torch.dtype,
    ):
        self.model, self.vae, self.normalizer, self.text_cache = model, vae, normalizer, text_cache
        self.device, self.dtype = torch.device(device), dtype
        self.config = model.cfg
        self.text_encoder: TextEncoder | None = None
        self.last_timing: dict[str, float] = {}

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path = DEFAULT_MODEL,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        vae_dtype: str | torch.dtype | None = None,
    ) -> SmolWAM:
        """Load a release from the Hugging Face Hub (repo id) or a local directory.

        device: cuda > mps > cpu by default. dtype: network weights and compute, bf16 on CUDA and fp32 elsewhere by
        default. vae_dtype: bf16 by default, as in training and every reported result; fp32 by default on the CPU,
        where bf16 convolutions are about 100x slower (latents then differ from training by bf16 rounding).
        """
        device = device or default_device()
        dtype = resolve_dtype(dtype, device)
        config, state_dict, text_cache = load_release(resolve_model_dir(name_or_path))
        model = load_model(SmolWAMConfig.from_dict(config["model"]), state_dict, device, dtype)
        if vae_dtype is None:
            vae_dtype = torch.float32 if torch.device(device).type == "cpu" else DTYPES[config["vae"]["dtype"]]
        vae = WanVAE.from_file(wan_vae_file(config["vae"]), device, resolve_dtype(vae_dtype, device))
        return cls(model, vae, Normalizer(config["norm_stats"]), text_cache, device, dtype)

    # ------------------------------------------------------------------------------------------------ inputs
    def encode_instruction(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached features for the LIBERO instructions; anything else is encoded live (BERT is loaded on first use)."""
        if instruction in self.text_cache:
            feats, mask = self.text_cache[instruction]
            return feats.to(self.device), mask.to(self.device)
        if self.text_encoder is None:
            self.text_encoder = TextEncoder(self.config.text_encoder, self.config.max_text_tokens, self.device)
        feats, mask = self.text_encoder([instruction])
        return feats[0], mask[0]

    @torch.no_grad()
    def encode_observation(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """-> trunk-space camera latents (1, 2, 16, 28, 28) and normalised proprio (9,), both on the device."""
        t0 = time.perf_counter()
        frames = np.stack([preprocess_env_frame(obs[f"{cam}_image"]) for cam in CAMERAS])
        t1 = time.perf_counter()
        z = self.vae.encode(frames_to_tensor(frames, self.device, self.dtype))[:, :, 0].float().unsqueeze(0)
        z = self.model.to_trunk_space(z)
        synchronize(self.device)
        t2 = time.perf_counter()
        proprio = self.normalizer.normalize(np.asarray(obs["proprio"]).astype(np.float32), "proprio")
        self.last_timing = {"preprocess": t1 - t0, "vae": t2 - t1}
        return z, torch.from_numpy(proprio).to(self.device)

    # ------------------------------------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_action(self, obs: dict, instruction: str, return_future: bool = False):
        """One policy query. Returns the (16, 7) action chunk; with `return_future`, also the predicted latents of
        both cameras 16 steps ahead, (2, 16, 28, 28) in the trunk's latent space (see `decode_future`)."""
        t0 = time.perf_counter()
        latents, proprio = self.encode_observation(obs)
        feats, mask = self.encode_instruction(instruction)
        t1 = time.perf_counter()
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            out = self.model(
                latents.to(self.dtype), feats[None].to(self.dtype), mask[None], proprio[None].to(self.dtype)
            )
        actions = out["actions"][0].float().cpu().numpy()
        synchronize(self.device)
        t2 = time.perf_counter()
        self.last_timing.update(model=t2 - t1, total=t2 - t0)
        actions = self.normalizer.unnormalize(actions, "actions")
        actions[:, -1] = np.where(actions[:, -1] > 0, 1.0, -1.0)  # binary gripper command
        if return_future:
            return actions, out["fut_latents"][0].float()
        return actions

    @torch.no_grad()
    def decode_future(self, future_latents: torch.Tensor) -> dict[str, np.ndarray]:
        """Predicted trunk-space latents (2, 16, 28, 28) -> 224 px RGB forecasts of each camera, uint8."""
        single = self.model.to_single_space(future_latents[None].float().to(self.device))[0]
        x = self.vae.decode(single.unsqueeze(2))[:, :, 0].float()
        imgs = ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        return {f"{cam}_image": imgs[i] for i, cam in enumerate(CAMERAS)}
