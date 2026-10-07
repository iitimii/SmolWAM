"""Training samples from the latent bank (`smolwam.train.bank`) and the cached teacher targets.

One sample is one stored step t of a demonstration: the current single-frame latents of both cameras, the latents
16 steps later (the world-head target), the normalised 16-step action chunk, normalised proprio now and 16 steps
later, the Monte-Carlo value at the future step, the instruction features and, when a KD directory is given, the
teacher's one-step predictions for the same inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from smolwam.layout import CAMERAS, CHUNK
from smolwam.preprocessing import Normalizer
from smolwam.text import TextCache


def action_chunk(actions: np.ndarray, t: int, chunk: int = CHUNK) -> np.ndarray:
    """actions[t : t + chunk], padded with the last action past the end of the episode."""
    if len(actions) - t >= chunk:
        return actions[t : t + chunk]
    return np.concatenate([actions[t:], np.tile(actions[-1], (chunk - (len(actions) - t), 1))], axis=0)


def mc_return(num_steps: int, t: int, gamma: float = 0.99) -> float:
    """Monte-Carlo return at step t of a successful episode (sparse terminal reward 1), rescaled to [-1, 1]."""
    return 2.0 * gamma ** (num_steps - 1 - t) - 1.0


class BankDataset(Dataset):
    """Rows of the latent bank, split by episode into train / val (`val_frac` of the episodes, at least one)."""

    def __init__(
        self,
        bank_dir: str | Path,
        text_cache: TextCache,
        normalizer: Normalizer,
        split: str = "train",
        val_frac: float = 0.03,
        seed: int = 0,
        kd_dir: str | Path | None = None,
        max_rows: int | None = None,
    ):
        self.dir = Path(bank_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.fut_rows = CHUNK // self.meta["stride"]
        self.lat = [np.load(self.dir / f"lat_{c}.npy", mmap_mode="r") for c in CAMERAS]
        self.proprio_full = np.load(self.dir / "proprio_full.npy", mmap_mode="r")
        self.actions_full = np.load(self.dir / "actions_full.npy", mmap_mode="r")
        self.row_step = np.load(self.dir / "row_step.npy")
        self.row_ep = np.load(self.dir / "row_ep.npy")
        self.ep_start_full = np.array(self.meta["ep_start_full"], dtype=np.int64)
        self.ep_len = np.array(self.meta["ep_len"], dtype=np.int64)
        self.ep_instr = np.array(self.meta["ep_instr"])
        row_start = np.array(self.meta["row_start"], dtype=np.int64)
        self.row_end = np.concatenate([row_start[1:], [self.meta["num_rows"]]])
        n_ep = len(self.ep_len)
        val_eps = set(np.random.RandomState(seed).permutation(n_ep)[: max(1, int(val_frac * n_ep))].tolist())
        eps = [e for e in range(n_ep) if (e in val_eps) == (split == "val")]
        self.rows = np.concatenate([np.arange(row_start[e], self.row_end[e]) for e in eps])
        self.kd = None
        if kd_dir is not None:
            kd_dir = Path(kd_dir)
            self.kd_index = {int(r): i for i, r in enumerate(np.load(kd_dir / "rows.npy"))}
            self.kd = {k: np.load(kd_dir / f"kd_{k}.npy", mmap_mode="r") for k in ("act", "fut")}
            self.rows = np.array([r for r in self.rows if int(r) in self.kd_index], dtype=np.int64)
        if max_rows is not None:
            self.rows = self.rows[:max_rows]
        self.normalizer = normalizer
        instructions = self.meta["instructions"]
        self.text_feat = torch.stack([text_cache[s][0] for s in instructions])
        self.text_mask = torch.stack([text_cache[s][1] for s in instructions])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        r = int(self.rows[i])
        e, t = int(self.row_ep[r]), int(self.row_step[r])
        r_fut = max(min(r + self.fut_rows, self.row_end[e] - 1), r)  # clamped to the episode's last stored row
        t_fut = int(self.row_step[r_fut])
        s0, n = self.ep_start_full[e], int(self.ep_len[e])
        norm = self.normalizer.normalize
        acts = action_chunk(np.asarray(self.actions_full[s0 : s0 + n]), t)
        out = {
            "lats": torch.from_numpy(np.stack([np.asarray(lat[r], dtype=np.float32) for lat in self.lat])),
            "fut_lats": torch.from_numpy(np.stack([np.asarray(lat[r_fut], dtype=np.float32) for lat in self.lat])),
            "proprio": torch.from_numpy(norm(np.asarray(self.proprio_full[s0 + t]), "proprio")),
            "fut_proprio": torch.from_numpy(norm(np.asarray(self.proprio_full[s0 + t_fut]), "proprio")),
            "actions": torch.from_numpy(norm(acts, "actions").astype(np.float32)),
            "value": torch.tensor(mc_return(n, t_fut), dtype=torch.float32),
            "text_feat": self.text_feat[self.ep_instr[e]],
            "text_mask": self.text_mask[self.ep_instr[e]],
        }
        if self.kd is not None:
            k = self.kd_index[r]
            out["kd_act"] = torch.from_numpy(np.array(self.kd["act"][k], dtype=np.float32))
            out["kd_fut"] = torch.from_numpy(np.array(self.kd["fut"][k], dtype=np.float32))
        return out
