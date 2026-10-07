"""Convert a training checkpoint into a release directory (see `smolwam.hub` for the format).

    python -m smolwam.export --ckpt runs/smolwam/best.pt --out release/   # data/ paths of experiments/train_smolwam.sh

Checkpoints written by `smolwam.train.train` hold every weight. Checkpoints of the original research code hold the
network without the frozen latent adapters (pass `--adapter`), store their config under `cfg` together with flags of
ablations that were dropped from this code (they must have the released values), and keep their text cache and
adapter as torch pickles; all of these are read here. Every tensor is written bit-identically, in fp32, and checked
after writing. Training counters (`trunk.accum_*`) and transformer_engine metadata (`*._extra_state`) are dropped.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch
from safetensors.torch import load_file

from smolwam.config import SmolWAMConfig
from smolwam.hub import CONFIG_FILE, RELEASE_FILES, WEIGHTS_FILE, save_pretrained
from smolwam.layout import ACTION_DIM, CHUNK, PROPRIO_DIM
from smolwam.model import SmolWAMModel
from smolwam.nn.adapter import load_adapter_state
from smolwam.preprocessing import load_norm_stats
from smolwam.text import TextCache

# research-code config flags that only served ablations: the value the released recipe used
RESEARCH_FIXED_FLAGS = {
    "use_adapter": True,
    "state_in_queries": True,
    "world_head": True,
    "text_name": "bert",
    "suite": "libero",
    "chunk": CHUNK,
    "action_dim": ACTION_DIM,
    "proprio_dim": PROPRIO_DIM,
}
RESEARCH_DROPPED = {"attn_backend", "grad_ckpt", "text_scale_init", "head_dim"}
TEXT_ENCODERS = {"bert": "bert-base-uncased"}


def config_from_checkpoint(ck: dict) -> SmolWAMConfig:
    if "config" in ck:
        return SmolWAMConfig.from_dict(ck["config"])
    cfg = dict(ck["cfg"])
    for k, v in RESEARCH_FIXED_FLAGS.items():
        if k in cfg and cfg.pop(k) != v:
            raise ValueError(f"checkpoint was trained with {k}={ck['cfg'][k]!r}; this code only supports {v!r}")
    if cfg.get("head_dim", cfg["model_dim"] // cfg["num_heads"]) * cfg["num_heads"] != cfg["model_dim"]:
        raise ValueError("width-pruned attention heads are not supported")
    for k in RESEARCH_DROPPED:
        cfg.pop(k, None)
    cfg["text_encoder"] = TEXT_ENCODERS[ck["cfg"]["text_name"]]
    return SmolWAMConfig.from_dict(cfg)


def load_text_cache(path: Path) -> TextCache:
    if path.suffix == ".safetensors":
        return TextCache.load(path)
    ck = torch.load(path, map_location="cpu", weights_only=True)  # research code: {"name", "cache": {s: (feat, mask)}}
    instructions = list(ck["cache"])
    feats = torch.stack([ck["cache"][s][0] for s in instructions])
    mask = torch.stack([ck["cache"][s][1] for s in instructions])
    return TextCache(instructions, feats, mask, TEXT_ENCODERS[ck["name"]])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def export(ckpt: Path, stats: Path, text_cache: Path, out: Path, adapter: Path | None = None) -> Path:
    ck = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    cfg = config_from_checkpoint(ck)
    sd = {k: v for k, v in ck["model"].items() if not (k.endswith("_extra_state") or k.startswith("trunk.accum_"))}
    if adapter is not None:
        sd.update(load_adapter_state(adapter))
    with torch.device("meta"):
        expected = SmolWAMModel(cfg).state_dict()
    missing, unexpected = sorted(set(expected) - set(sd)), sorted(set(sd) - set(expected))
    if missing or unexpected:
        raise ValueError(f"state dict mismatch: missing {missing[:8]} unexpected {unexpected[:8]} (adapter missing?)")
    for k, v in sd.items():
        if v.shape != expected[k].shape or v.dtype != torch.float32:
            raise ValueError(f"{k}: {tuple(v.shape)} {v.dtype}, expected {tuple(expected[k].shape)} float32")
    training = {"step": ck.get("step"), **{k: v for k, v in (ck.get("val") or {}).items() if k.startswith("val_")}}
    cache = load_text_cache(text_cache)
    save_pretrained(out, cfg, sd, load_norm_stats(stats), cache, extra={"training": training})
    written = load_file(str(out / WEIGHTS_FILE))
    assert written.keys() == sd.keys() and all(torch.equal(written[k], sd[k]) for k in sd), "weights changed on write"
    reread = TextCache.load(out / "text_cache.safetensors")
    assert reread.instructions == cache.instructions and torch.equal(reread.features, cache.features)
    assert torch.equal(reread.mask, cache.mask)
    print(
        f"exported {len(sd)} tensors ({sum(v.numel() for v in sd.values()) / 1e6:.1f} M values), "
        f"{len(cache.instructions)} cached instructions -> {out}"
    )
    for name in RELEASE_FILES:
        print(f"  {name:24s} {(out / name).stat().st_size:>13,d} B  sha256 {sha256(out / name)}")
    return out / CONFIG_FILE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, required=True, help="training checkpoint (.pt)")
    ap.add_argument(
        "--stats",
        type=Path,
        default=Path("data/LIBERO-Cosmos-Policy/success_only/dataset_statistics.json"),
        help="dataset_statistics.json of the training data",
    )
    ap.add_argument(
        "--text_cache", type=Path, default=Path("data/text_cache.safetensors"), help="cached instruction features"
    )
    ap.add_argument("--adapter", type=Path, default=None, help="latent adapter, if the checkpoint does not hold it")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    export(args.ckpt, args.stats, args.text_cache, args.out, args.adapter)


if __name__ == "__main__":
    main()
