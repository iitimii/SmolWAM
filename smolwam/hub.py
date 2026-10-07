"""Release format, Hugging Face download and `save_pretrained`.

A SmolWAM release is a directory (or Hugging Face model repo) with three files, none of them pickles:

    config.json             network config (incl. the text encoder), action / proprio normalisation statistics,
                            VAE source and version
    model.safetensors       every network weight in fp32, incl. the latent adapters, the blank-frame latent and
                            the fixed readout noise
    text_cache.safetensors  BERT features of the 40 LIBERO instructions (instruction strings in the metadata)

The Wan2.1 VAE is not redistributed: `Wan2.1_VAE.pth` is fetched from Wan-AI/Wan2.1-T2V-1.3B on first use (or read
from the path in the SMOLWAM_WAN_VAE environment variable).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from smolwam import __version__
from smolwam.config import SmolWAMConfig
from smolwam.text import TextCache

DEFAULT_MODEL = "timiiowolabi/SmolWAM"
CONFIG_FILE, WEIGHTS_FILE, TEXT_CACHE_FILE = "config.json", "model.safetensors", "text_cache.safetensors"
RELEASE_FILES = (CONFIG_FILE, WEIGHTS_FILE, TEXT_CACHE_FILE)
WAN_VAE = {"repo_id": "Wan-AI/Wan2.1-T2V-1.3B", "filename": "Wan2.1_VAE.pth", "dtype": "bfloat16"}


def resolve_model_dir(name_or_path: str | Path) -> Path:
    """A local release directory, or a Hugging Face repo id downloaded to the local cache."""
    path = Path(name_or_path).expanduser()
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(repo_id=str(name_or_path), allow_patterns=list(RELEASE_FILES)))
    missing = [f for f in RELEASE_FILES if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{name_or_path} is not a SmolWAM release: missing {', '.join(missing)}")
    return path


def wan_vae_file(vae: dict | None = None) -> Path:
    """Local path of `Wan2.1_VAE.pth`: $SMOLWAM_WAN_VAE if set, else downloaded from the Hugging Face Hub."""
    if os.environ.get("SMOLWAM_WAN_VAE"):
        return Path(os.environ["SMOLWAM_WAN_VAE"])
    from huggingface_hub import hf_hub_download

    vae = vae or WAN_VAE
    return Path(hf_hub_download(repo_id=vae["repo_id"], filename=vae["filename"]))


def load_release(model_dir: Path) -> tuple[dict, dict[str, torch.Tensor], TextCache]:
    config = json.loads((model_dir / CONFIG_FILE).read_text())
    return config, load_file(str(model_dir / WEIGHTS_FILE)), TextCache.load(model_dir / TEXT_CACHE_FILE)


def save_pretrained(
    out_dir: str | Path,
    model_config: SmolWAMConfig,
    state_dict: dict[str, torch.Tensor],
    norm_stats: dict,
    text_cache: TextCache,
    extra: dict | None = None,
) -> Path:
    """Write a release directory. `norm_stats`: {"actions"|"proprio": {"min": [...], "max": [...]}}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "smolwam_version": __version__,
        "model": model_config.to_dict(),
        "norm_stats": norm_stats,
        "vae": WAN_VAE,
        **(extra or {}),
    }
    (out / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
    tensors = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(tensors, str(out / WEIGHTS_FILE), metadata={"format": "pt", "smolwam_version": __version__})
    text_cache.save(out / TEXT_CACHE_FILE)
    return out
