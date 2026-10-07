"""The Cosmos Policy 2B teacher (nvidia/Cosmos-Policy-LIBERO-Predict2-2B).

Two uses:
  * distillation targets (`smolwam.train.kd_targets`) only need the teacher's weights and its precomputed T5-XXL
    instruction embeddings, run by SmolWAM's own single-pass code (`SmolWAMConfig.cosmos_teacher`); no extra package;
  * evaluating the teacher itself (`eval_libero --policy cosmos`) runs the upstream implementation and needs the
    `cosmos_policy` package from https://github.com/NVlabs/cosmos-policy (Linux + CUDA), imported lazily here.
"""

from __future__ import annotations

import os
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

TEACHER_REPO = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
TEACHER_CHECKPOINT = "Cosmos-Policy-LIBERO-Predict2-2B.pt"
TEACHER_T5 = "libero_t5_embeddings.pkl"
TEACHER_STATS = "libero_dataset_statistics.json"
TEACHER_EXPERIMENT = "cosmos_predict2_2b_480p_libero__inference_only"


def teacher_file(filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=TEACHER_REPO, filename=filename))


def load_teacher_state_dict() -> dict[str, torch.Tensor]:
    """The teacher checkpoint (keys `net.*`, 3.9 GB), memory-mapped."""
    return torch.load(teacher_file(TEACHER_CHECKPOINT), map_location="cpu", mmap=True, weights_only=False)


def load_t5_embeddings() -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Cosmos Policy's cached T5-XXL embeddings: {instruction: (features (512, 1024), mask (512,))}. Positions past the
    instruction are exactly zero, which is what makes them an attention sink."""
    with open(teacher_file(TEACHER_T5), "rb") as f:
        raw = pickle.load(f)
    out = {}
    for instruction, emb in raw.items():
        feats = emb.float().reshape(512, 1024)
        out[instruction] = (feats, feats.abs().sum(-1) > 0)
    return out


def _import_cosmos_utils():
    try:
        from cosmos_policy.experiments.robot import cosmos_utils
    except ImportError as e:
        raise ImportError(
            "Evaluating the Cosmos Policy teacher needs the upstream package (Linux + CUDA): install "
            "https://github.com/NVlabs/cosmos-policy following its README, then rerun."
        ) from e
    return cosmos_utils


@contextmanager
def _cwd(path: Path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class CosmosPolicy:
    """Cosmos Policy 2B with the upstream LIBERO inference code, behind SmolWAM's `predict_action` interface."""

    def __init__(self, num_steps: int = 5, seed: int = 195):
        cosmos_utils = _import_cosmos_utils()
        import cosmos_policy
        from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

        repo_root = Path(cosmos_policy.__file__).resolve().parent.parent  # its config paths are relative to the repo
        with _cwd(repo_root):
            self.model, _ = load_model_from_checkpoint(
                experiment_name=TEACHER_EXPERIMENT,
                s3_checkpoint_dir=cosmos_utils.download_hf_checkpoint(TEACHER_REPO),
                config_file="cosmos_policy/config/config.py",
                load_ema_to_reg=False,
            )
        self.model = self.model.eval().to("cuda")
        self.stats = cosmos_utils.load_dataset_stats(f"{TEACHER_REPO}/{TEACHER_STATS}")
        cosmos_utils.init_t5_text_embeddings_cache(f"{TEACHER_REPO}/{TEACHER_T5}")
        self.cfg = SimpleNamespace(
            suite="libero",
            use_proprio=True,
            normalize_proprio=True,
            use_wrist_image=True,
            num_wrist_images=1,
            use_third_person_image=True,
            num_third_person_images=1,
            use_jpeg_compression=True,
            trained_with_image_aug=True,
            chunk_size=16,
            unnormalize_actions=True,
            use_variance_scale=False,
        )
        self.num_steps, self.seed = num_steps, seed
        self.get_action = cosmos_utils.get_action
        self.last_timing: dict[str, float] = {}

    def predict_action(self, obs: dict, instruction: str) -> np.ndarray:
        t0 = time.perf_counter()
        ret = self.get_action(
            self.cfg,
            self.model,
            self.stats,
            obs,
            instruction,
            seed=self.seed,
            randomize_seed=False,
            num_denoising_steps_action=self.num_steps,
            generate_future_state_and_value_in_parallel=False,
        )
        torch.cuda.synchronize()
        self.last_timing = {"total": time.perf_counter() - t0}
        return np.stack(ret["actions"])
