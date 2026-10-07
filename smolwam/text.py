"""Instruction features: a small text encoder (BERT-base) in place of Cosmos Policy's 4.8 B T5-XXL.

The full token sequence is kept (no pooling), padded or truncated to `max_tokens`; padded positions are zeroed and
masked. LIBERO has 40 instructions, so their features are computed once and shipped with the model
(`text_cache.safetensors`); the policy only loads BERT for instructions outside that cache.

    python -m smolwam.text --bank data/bank --out data/text_cache.safetensors   # cache the bank's instructions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

DEFAULT_TEXT_ENCODER = "bert-base-uncased"
DEFAULT_MAX_TOKENS = 32


class TextEncoder:
    """Live instruction encoder: list of strings -> (features (B, N, dim) fp32, mask (B, N) bool)."""

    def __init__(self, name: str = DEFAULT_TEXT_ENCODER, max_tokens: int = DEFAULT_MAX_TOKENS, device="cpu"):
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging

        self.name, self.max_tokens, self.device = name, max_tokens, device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # the checkpoint's unused pretraining heads (cls.*) are expected, not an error
        try:
            self.model = AutoModel.from_pretrained(name).to(device).eval()
        finally:
            hf_logging.set_verbosity(verbosity)

    @torch.no_grad()
    def __call__(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.tokenizer(
            instructions, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_tokens
        ).to(self.device)
        mask = ids["attention_mask"].bool()
        feats = self.model(**ids).last_hidden_state * mask.unsqueeze(-1)
        return feats.float(), mask

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


class TextCache:
    """Precomputed instruction features, stored as safetensors with the instruction strings in the metadata."""

    def __init__(self, instructions: list[str], features: torch.Tensor, mask: torch.Tensor, encoder: str):
        self.index = {s: i for i, s in enumerate(instructions)}
        self.features, self.mask, self.encoder = features, mask, encoder

    @property
    def instructions(self) -> list[str]:
        return list(self.index)

    def __contains__(self, instruction: str) -> bool:
        return instruction in self.index

    def __getitem__(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        i = self.index[instruction]
        return self.features[i], self.mask[i]

    def save(self, path: str | Path) -> None:
        meta = {"instructions": json.dumps(self.instructions), "encoder": self.encoder}
        save_file({"features": self.features.contiguous(), "mask": self.mask.contiguous()}, str(path), metadata=meta)

    @classmethod
    def load(cls, path: str | Path) -> TextCache:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            meta = f.metadata()
        tensors = load_file(str(path))
        return cls(json.loads(meta["instructions"]), tensors["features"], tensors["mask"], meta["encoder"])

    @classmethod
    def build(cls, instructions: list[str], encoder: TextEncoder) -> TextCache:
        feats, mask = encoder(instructions)
        return cls(instructions, feats.cpu(), mask.cpu(), encoder.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache the instruction features of a latent bank.")
    ap.add_argument("--bank", required=True, help="bank directory (its meta.json lists the instructions)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default=DEFAULT_TEXT_ENCODER)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    instructions = json.loads((Path(args.bank) / "meta.json").read_text())["instructions"]
    cache = TextCache.build(instructions, TextEncoder(args.encoder, args.max_tokens, args.device))
    cache.save(args.out)
    print(f"cached {len(instructions)} instructions ({args.encoder}, {args.max_tokens} tokens) -> {args.out}")


if __name__ == "__main__":
    main()
