"""Train the latent adapters (single-frame <-> clip-mode Wan VAE latents) on paired encodings of bank frames.

    python -m smolwam.train.adapter pairs --bank data/bank --n 6000 --out data/latent_pairs.npz   # ~20 min on an A100
    python -m smolwam.train.adapter fit --pairs data/latent_pairs.npz --out data/adapter.safetensors

The clip reproduces how Cosmos Policy encodes an observation: a blank frame, four blank frames (the proprio
placeholder), then each camera image repeated four times, encoded as one 13-frame video with the causal temporal
cache. Its latent frames 2 and 3 are the wrist and primary latents the trunk saw in training; the bank's
single-frame latents of the same images are the adapter's inputs. The released adapter reaches a held-out L1 of 0.088
(an affine map alone: 0.48).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from smolwam.hub import wan_vae_file
from smolwam.layout import CAMERAS, LATENT_CH, LATENT_HW
from smolwam.nn.adapter import LatentAdapter, inverse_adapter
from smolwam.nn.wan_vae import WanVAE
from smolwam.preprocessing import decode_jpeg


@torch.no_grad()
def build_pairs(bank: Path, n_rows: int, out: Path, device: str, seed: int = 0) -> None:
    vae = WanVAE.from_file(wan_vae_file(), device)
    meta = json.loads((bank / "meta.json").read_text())
    rows = np.sort(np.random.RandomState(seed).choice(meta["num_rows"], n_rows, replace=False))
    single = np.stack([np.asarray(np.load(bank / f"lat_{c}.npy", mmap_mode="r")[rows]) for c in CAMERAS], 1)
    clip_lat = np.zeros((n_rows, len(CAMERAS), LATENT_CH, LATENT_HW, LATENT_HW), np.float16)
    t0 = time.time()
    with h5py.File(bank / "frames.h5", "r") as frames:
        for i, r in enumerate(rows):
            imgs = [
                torch.from_numpy(decode_jpeg(frames[f"{c}_jpeg"][r])).to(device).permute(2, 0, 1).float() / 127.5 - 1
                for c in CAMERAS
            ]
            blank = torch.full_like(imgs[0], -1.0)
            clip = [blank] * 5 + [im for im in imgs for _ in range(4)]
            z = vae.encode(torch.stack(clip, 1).unsqueeze(0).to(torch.bfloat16))[0].float()  # (16, 2 + n_cams, 28, 28)
            for c in range(len(CAMERAS)):
                clip_lat[i, c] = z[:, 2 + c].cpu().numpy().astype(np.float16)
            if i % 500 == 0:
                print(f"[{i}/{n_rows}] {(time.time() - t0) / 60:.1f} min", flush=True)
    np.savez(out, rows=rows, clip=clip_lat, single=single)
    print("saved", out)


def fit(
    pairs: Path, out: Path, device: str, steps: int = 4000, bs: int = 64, lr: float = 2e-4, width: int = 96
) -> None:
    d = np.load(pairs)
    single, clip = d["single"].astype(np.float32), d["clip"].astype(np.float32)  # (N, n_cams, 16, 28, 28)
    n, n_cams = single.shape[:2]
    n_val = min(max(200, n // 20), n // 2)  # 300 of the 6,000 released pairs
    s_tr, c_tr = torch.from_numpy(single[: n - n_val]).to(device), torch.from_numpy(clip[: n - n_val]).to(device)
    s_va, c_va = torch.from_numpy(single[n - n_val :]).to(device), torch.from_numpy(clip[n - n_val :]).to(device)
    fwd, inv = LatentAdapter(width, n_cams).to(device), inverse_adapter(width, n_cams).to(device)
    opt = torch.optim.AdamW(list(fwd.parameters()) + list(inv.parameters()), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    cams = torch.arange(n_cams, device=device)

    def flat(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, LATENT_CH, LATENT_HW, LATENT_HW)

    for step in range(1, steps + 1):
        idx = torch.randint(0, s_tr.shape[0], (bs,), device=device)
        s, c, cam = flat(s_tr[idx]), flat(c_tr[idx]), cams.repeat(bs)
        loss = F.l1_loss(fwd(s, cam), c) + F.l1_loss(inv(c, cam), s)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == steps:
            with torch.no_grad():
                s, c, cam = flat(s_va), flat(c_va), cams.repeat(s_va.shape[0])
                val = {
                    "fwd_l1": F.l1_loss(fwd(s, cam), c).item(),
                    "inv_l1": F.l1_loss(inv(c, cam), s).item(),
                    "affine_only_l1": F.l1_loss(s * fwd.scale[cam] + fwd.shift[cam], c).item(),
                }
            print(
                f"step {step} | train {loss.item():.4f} | val " + " ".join(f"{k} {v:.4f}" for k, v in val.items()),
                flush=True,
            )
    tensors = {
        **{f"adapter.{k}": v for k, v in fwd.state_dict().items()},
        **{f"adapter_inv.{k}": v for k, v in inv.state_dict().items()},
    }
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in tensors.items()},
        str(out),
        metadata={k: f"{v:.5f}" for k, v in val.items()},
    )
    print("saved", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pairs", help="encode paired single-frame / clip-mode latents")
    p.add_argument("--bank", type=Path, default=Path("data/bank"))
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--out", type=Path, default=Path("data/latent_pairs.npz"))
    f = sub.add_parser("fit", help="train the forward and inverse adapters")
    f.add_argument("--pairs", type=Path, default=Path("data/latent_pairs.npz"))
    f.add_argument("--out", type=Path, default=Path("data/adapter.safetensors"))
    f.add_argument("--steps", type=int, default=4000)
    for q in (p, f):
        q.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.cmd == "pairs":
        build_pairs(args.bank, args.n, args.out, args.device)
    else:
        fit(args.pairs, args.out, args.device, args.steps)


if __name__ == "__main__":
    main()
