"""Build the latent bank from the LIBERO-Cosmos-Policy demonstrations (nvidia/LIBERO-Cosmos-Policy, `success_only`).

Every `stride`-th step of every demonstration (the offset rotates with the episode index) is preprocessed exactly as
at test time (224 px, centre crop) and encoded once per camera as a single-frame Wan VAE latent. The bank keeps those
latents (fp16), the full-rate proprio and actions (so a row at step t can fetch its 16-step chunk and its future at
t + 16), the 224 px frames as JPEG (for the latent-adapter pairs), the episode index and the instructions.

    python -m smolwam.train.bank --data data/LIBERO-Cosmos-Policy/success_only --out data/bank   # ~68.8k rows

Output: meta.json, lat_{wrist,primary}.npy, proprio_full.npy, actions_full.npy, row_step.npy, row_ep.npy,
frames.h5, blank_latent.npy (latent of an all-black frame, Cosmos Policy's placeholder first frame).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from smolwam.envs.libero import SUITES
from smolwam.hub import wan_vae_file
from smolwam.layout import ACTION_DIM, CAMERAS, IMAGE_SIZE, LATENT_CH, LATENT_HW, PROPRIO_DIM
from smolwam.nn.wan_vae import WanVAE
from smolwam.preprocessing import encode_jpeg, frames_to_tensor, preprocess_dataset_frame

CAMERA_KEYS = {"wrist": "obs/eye_in_hand_rgb_jpeg", "primary": "obs/agentview_rgb_jpeg"}


def instruction_from_filename(path: Path) -> str:
    """Cosmos Policy's rule: the words of `<SCENE>_<instruction>_demo.hdf5` after the last word containing SCENE."""
    words = path.name[: -len("_demo.hdf5")].split("_")
    command = ""
    for w in words:
        command = "" if "SCENE" in w else command + w + " "
    return command[:-1]


def list_episodes(data_dir: Path, suites: list[str]) -> list[dict]:
    episodes = []
    for suite in suites:
        for f in sorted((data_dir / f"{suite}_regen").glob("*.hdf5")):
            with h5py.File(f, "r") as h:
                for k in sorted(h["data"].keys(), key=lambda x: int(x.split("_")[1])):
                    episodes.append(
                        {
                            "suite": SUITES.index(suite),
                            "file": str(f),
                            "demo": f"data/{k}",
                            "instruction": instruction_from_filename(f),
                            "num_steps": int(h[f"data/{k}/actions"].shape[0]),
                        }
                    )
    return episodes


class StepDataset(Dataset):
    """(episode, step) rows at steps o, o + stride, ... with offset o = episode index mod stride."""

    def __init__(self, episodes: list[dict], stride: int):
        self.episodes, self.index, self.row_start = episodes, [], []
        for e, ep in enumerate(episodes):
            self.row_start.append(len(self.index))
            self.index += [(e, t) for t in range(e % stride, ep["num_steps"], stride)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        e, t = self.index[i]
        ep = self.episodes[e]
        with h5py.File(ep["file"], "r") as h:
            frames = {c: preprocess_dataset_frame(h[ep["demo"]][CAMERA_KEYS[c]][t]) for c in CAMERAS}
        out = {"idx": i}
        for c in CAMERAS:
            out[c] = frames[c]
            out[c + "_jpeg"] = np.frombuffer(encode_jpeg(frames[c]), dtype=np.uint8)
        return out


def collate(items: list[dict]) -> dict:
    out = {"idx": np.array([it["idx"] for it in items])}
    for c in CAMERAS:
        out[c] = np.stack([it[c] for it in items])
        out[c + "_jpeg"] = [it[c + "_jpeg"] for it in items]
    return out


@torch.no_grad()
def encode_frames(vae: WanVAE, frames: np.ndarray) -> np.ndarray:
    """(N, 224, 224, 3) uint8 -> (N, 16, 28, 28) single-frame latents, fp32."""
    device = vae.conv1.weight.device
    return vae.encode(frames_to_tensor(frames, device, torch.float32))[:, :, 0].float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/LIBERO-Cosmos-Policy/success_only")
    ap.add_argument("--out", default="data/bank")
    ap.add_argument("--stride", type=int, default=4, help="encode every k-th step")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    episodes = list_episodes(Path(args.data), list(SUITES))
    ep_len = np.array([e["num_steps"] for e in episodes], dtype=np.int64)
    ep_start_full = np.concatenate([[0], np.cumsum(ep_len)[:-1]]).astype(np.int64)
    ds = StepDataset(episodes, args.stride)
    n_rows = len(ds)
    instructions = sorted({e["instruction"] for e in episodes})
    meta = {
        "cams": list(CAMERAS),
        "num_rows": n_rows,
        "num_steps_full": int(ep_len.sum()),
        "stride": args.stride,
        "num_episodes": len(episodes),
        "suites": list(SUITES),
        "instructions": instructions,
        "row_start": ds.row_start,
        "ep_start_full": ep_start_full.tolist(),
        "ep_len": ep_len.tolist(),
        "ep_suite": [e["suite"] for e in episodes],
        "ep_instr": [instructions.index(e["instruction"]) for e in episodes],
        "ep_file": [Path(e["file"]).name for e in episodes],
        "ep_demo": [e["demo"] for e in episodes],
        "transform": "resize 224 + centre crop 0.9 (Cosmos test-time transform), single-frame Wan2.1 VAE (bf16)",
    }
    (out / "meta.json").write_text(json.dumps(meta))
    print(f"{len(episodes)} episodes, {n_rows} rows, {len(instructions)} instructions", flush=True)
    np.save(out / "row_step.npy", np.array([t for _, t in ds.index], dtype=np.int32))
    np.save(out / "row_ep.npy", np.array([e for e, _ in ds.index], dtype=np.int32))
    proprio = np.zeros((int(ep_len.sum()), PROPRIO_DIM), np.float32)
    actions = np.zeros((int(ep_len.sum()), ACTION_DIM), np.float32)
    for e, ep in enumerate(episodes):
        with h5py.File(ep["file"], "r") as h:
            s = slice(ep_start_full[e], ep_start_full[e] + ep["num_steps"])
            proprio[s] = h[ep["demo"]]["robot_states"][:]
            actions[s] = h[ep["demo"]]["actions"][:]
    np.save(out / "proprio_full.npy", proprio)
    np.save(out / "actions_full.npy", actions)

    vae = WanVAE.from_file(wan_vae_file(), args.device)
    black = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    np.save(out / "blank_latent.npy", encode_frames(vae, black)[0])
    shape = (n_rows, LATENT_CH, LATENT_HW, LATENT_HW)
    lat = {
        c: np.lib.format.open_memmap(out / f"lat_{c}.npy", mode="w+", dtype=np.float16, shape=shape) for c in CAMERAS
    }
    with h5py.File(out / "frames.h5", "w") as frames_h5:
        jpegs = {c: frames_h5.create_dataset(f"{c}_jpeg", (n_rows,), dtype=h5py.vlen_dtype(np.uint8)) for c in CAMERAS}
        loader = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, collate_fn=collate)
        t0, done = time.time(), 0
        for bi, b in enumerate(loader):
            for c in CAMERAS:
                lat[c][b["idx"]] = encode_frames(vae, b[c]).astype(np.float16)
                for j, i in enumerate(b["idx"]):
                    jpegs[c][i] = b[c + "_jpeg"][j]
            done += len(b["idx"])
            if bi % 50 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"[{done}/{n_rows}] {rate:.1f} rows/s, eta {(n_rows - done) / rate / 60:.1f} min", flush=True)
    for m in lat.values():
        m.flush()
    print("bank complete:", out)


if __name__ == "__main__":
    main()
