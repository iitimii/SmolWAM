"""Cosmos Policy's one-step predictions on every bank row: the distillation (KD) targets.

The teacher (all 28 blocks, its own T5-XXL instruction embeddings and token layout) is run by SmolWAM's single-pass
code on exactly the inputs the student sees: adapted single-frame latents and the student's fixed noise in the
generated slots. Its one-step rectified-flow estimate is then a deterministic function the student can match. Run by
this code, the teacher's action L1 to the demonstrations is 0.0094. Writes are incremental and resumable.

    python -m smolwam.train.kd_targets --bank data/bank --adapter data/adapter.safetensors --out data/kd  # 66 min

Output: rows.npy (bank rows), kd_act.npy (N, 16, 7) normalised actions, kd_fut.npy (N, 2, 16, 28, 28) future latents
in trunk space (fp16), meta.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from smolwam.config import SmolWAMConfig
from smolwam.layout import ACTION_DIM, CAMERAS, CHUNK, LATENT_CH, LATENT_HW
from smolwam.preprocessing import Normalizer, load_norm_stats
from smolwam.teacher import load_t5_embeddings
from smolwam.text import TextCache
from smolwam.train.data import BankDataset
from smolwam.train.train import build_model, prepare_batch


def t5_text_cache(instructions: list[str]) -> TextCache:
    t5 = load_t5_embeddings()
    return TextCache(
        instructions,
        torch.stack([t5[s][0] for s in instructions]),
        torch.stack([t5[s][1] for s in instructions]),
        "t5-xxl (Cosmos Policy)",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bank", default="data/bank")
    ap.add_argument("--adapter", default="data/adapter.safetensors")
    ap.add_argument("--stats", default="data/LIBERO-Cosmos-Policy/success_only/dataset_statistics.json")
    ap.add_argument("--out", default="data/kd")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = build_model(
        SmolWAMConfig.cosmos_teacher(), "teacher", Path(args.adapter), Path(args.bank) / "blank_latent.npy"
    )
    model = model.to(device, torch.bfloat16).eval()
    model.adapter.float()
    model.adapter_inv.float()
    instructions = json.loads((Path(args.bank) / "meta.json").read_text())["instructions"]
    # val_frac=0 still holds out one episode (at least one is always held out), as in the released run
    ds = BankDataset(args.bank, t5_text_cache(instructions), Normalizer(load_norm_stats(args.stats)), val_frac=0.0)
    n = len(ds)
    progress = out / "progress.json"
    done = json.loads(progress.read_text())["done"] if progress.exists() else 0
    mode = "r+" if done > 0 else "w+"
    kd_act = np.lib.format.open_memmap(out / "kd_act.npy", mode=mode, dtype=np.float32, shape=(n, CHUNK, ACTION_DIM))
    kd_fut = np.lib.format.open_memmap(
        out / "kd_fut.npy", mode=mode, dtype=np.float16, shape=(n, len(CAMERAS), LATENT_CH, LATENT_HW, LATENT_HW)
    )
    np.save(out / "rows.npy", ds.rows)
    loader = DataLoader(Subset(ds, list(range(done, n))), batch_size=args.bs, shuffle=False, num_workers=args.workers)
    t0, i0, l1_sum = time.time(), done, 0.0
    for bi, batch in enumerate(loader):
        b = prepare_batch(model, batch, device)
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
            o = model(b["lats"], b["text_feat"], b["text_mask"], b["proprio"])
        bs = b["lats"].shape[0]
        kd_act[i0 : i0 + bs] = o["actions"].float().cpu().numpy()
        kd_fut[i0 : i0 + bs] = o["fut_latents"].float().cpu().numpy().astype(np.float16)
        l1_sum += (o["actions"].float() - b["actions"]).abs().mean().item() * bs
        i0 += bs
        if bi % 50 == 0 or i0 == n:
            kd_act.flush()
            kd_fut.flush()
            progress.write_text(json.dumps({"done": i0, "n": n}))
            print(
                f"[{i0}/{n}] {(time.time() - t0) / 60:.1f} min | teacher-vs-demo action L1 "
                f"{l1_sum / max(i0 - done, 1):.4f}",
                flush=True,
            )
    (out / "meta.json").write_text(json.dumps({"n": n, "teacher_vs_demo_action_l1": l1_sum / max(i0 - done, 1)}) + "\n")
    print("KD targets complete:", out)


if __name__ == "__main__":
    main()
