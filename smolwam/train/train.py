"""Train SmolWAM by distillation from Cosmos Policy with a layout curriculum. Defaults are the released recipe.

    python -m smolwam.train.train --run runs/smolwam --bank data/bank --kd data/kd \
        --text_cache data/text_cache.safetensors --adapter data/adapter.safetensors \
        --stats data/LIBERO-Cosmos-Policy/success_only/dataset_statistics.json

The network starts from the kept teacher blocks (0-2 and 19-27) and new zero-initialised modules. Pruning and token
compaction are separate distribution shifts, so the first `--curriculum_frac` of the steps run in the teacher's
1,764-token layout (where the pruned trunk and the new text path recover the teacher's function) and the rest in
the compact 1,008-token layout. Loss: L1 to the demonstration actions and future latents (+ future proprio and value
while those slots exist) plus L1 to the teacher's cached one-step predictions (KD). AdamW, lr 2e-5 for inherited
trunk weights and 1e-4 for new modules, 200 warm-up steps then cosine decay to 5 %, batch 32, bf16 autocast,
5,000 steps (3.6 h on one A100-40GB). `best.pt` is the compact-layout checkpoint with the lowest validation
action L1; convert it with `python -m smolwam.export`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from smolwam.config import SmolWAMConfig
from smolwam.layout import TEACHER_SLOT_TOKENS
from smolwam.model import SmolWAMModel
from smolwam.nn.adapter import load_adapter_state
from smolwam.preprocessing import Normalizer, load_norm_stats
from smolwam.text import TextCache
from smolwam.train.data import BankDataset

LOSS_WEIGHTS = {"world": 1.0, "fut_proprio": 0.1, "value": 0.1, "kd_act": 1.0, "kd_world": 1.0}


def compute_losses(out: dict, batch: dict, weights: dict = LOSS_WEIGHTS) -> tuple[torch.Tensor, dict]:
    """L1 behaviour cloning on the action chunk, L1 on the world head and the auxiliary slots present in the layout,
    and L1 to the teacher's one-step predictions when the batch carries them."""
    losses = {"action_l1": (out["actions"] - batch["actions"]).abs().mean()}
    total = losses["action_l1"]
    terms = [
        ("world_l1", "world", "fut_latents", "fut_lats"),
        ("fut_proprio_l1", "fut_proprio", "fut_proprio", "fut_proprio"),
        ("value_l1", "value", "value", "value"),
        ("kd_act_l1", "kd_act", "actions", "kd_act"),
        ("kd_world_l1", "kd_world", "fut_latents", "kd_fut"),
    ]
    for name, w, pred, target in terms:
        if pred in out and target in batch and weights[w] > 0:
            losses[name] = (out[pred] - batch[target]).abs().mean()
            total = total + weights[w] * losses[name]
    losses["total"] = total
    return total, losses


def lr_factor(step: int, warmup: int, steps: int) -> float:
    """Linear warm-up, then cosine decay to 5 % of the base learning rate."""
    if step < warmup:
        return step / max(1, warmup)
    p = min(1.0, (step - warmup) / max(1, steps - warmup))
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def prepare_batch(model: SmolWAMModel, batch: dict, device: torch.device) -> dict:
    """Move a batch to the device and map the bank's single-frame latents (input and target) into trunk space."""
    b = to_device(batch, device)
    b["lats"], b["fut_lats"] = model.to_trunk_space(b["lats"]), model.to_trunk_space(b["fut_lats"])
    return b


def build_model(cfg: SmolWAMConfig, init: str, adapter: Path, blank: Path) -> SmolWAMModel:
    """New network with the frozen latent adapters and the blank-frame latent; trunk from the teacher, a checkpoint,
    or random ("none")."""
    model = SmolWAMModel(cfg)
    missing = model.load_state_dict(load_adapter_state(adapter), strict=False).missing_keys
    assert not [k for k in missing if k.startswith("adapter")], "incomplete latent adapter file"
    model.blank.copy_(torch.from_numpy(np.load(blank)))
    if init == "teacher":
        from smolwam.teacher import load_teacher_state_dict

        missing = model.load_cosmos_trunk(load_teacher_state_dict())
        assert not missing, f"teacher checkpoint lacks {missing[:5]}"
    elif init != "none":
        model.load_state_dict(torch.load(init, map_location="cpu", weights_only=True)["model"])
    return model


@torch.no_grad()
def evaluate(model: SmolWAMModel, loader: DataLoader, device: torch.device, amp: bool, max_batches: int = 40) -> dict:
    model.eval()
    sums, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        b = prepare_batch(model, batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            out = model(b["lats"], b["text_feat"], b["text_mask"], b["proprio"])
        for k, v in compute_losses(out, b)[1].items():
            sums[k] = sums.get(k, 0.0) + v.item()
        n += 1
    model.train()
    return {f"val_{k}": v / max(n, 1) for k, v in sums.items()}


def save_checkpoint(path: Path, model: SmolWAMModel, args, step: int, **extra) -> None:
    torch.save(
        {"model": model.state_dict(), "config": model.cfg.to_dict(), "args": vars(args), "step": step, **extra}, path
    )


def train(args, cfg: SmolWAMConfig) -> float:
    """Runs the training loop; returns the best compact-layout validation action L1."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    amp = device.type == "cuda"
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg, args.init, Path(args.adapter), Path(args.bank) / "blank_latent.npy").to(device)
    model.grad_checkpointing = not args.no_grad_ckpt
    switch_step = int(args.curriculum_frac * args.steps)
    if switch_step > 0:
        model.set_slot_tokens(TEACHER_SLOT_TOKENS)  # phase 1 (the state MLP is zero-initialised, so harmless)

    text_cache, normalizer = TextCache.load(args.text_cache), Normalizer(load_norm_stats(args.stats))
    data = dict(bank_dir=args.bank, text_cache=text_cache, normalizer=normalizer, kd_dir=args.kd)
    ds_train = BankDataset(split="train", max_rows=args.max_rows, **data)
    ds_val = BankDataset(split="val", **data)
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    dl_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, drop_last=True, **loader_kw)
    dl_val = DataLoader(ds_val, batch_size=args.bs, shuffle=False, num_workers=min(args.workers, 2))
    print(
        f"[train] {model.num_params() / 1e6:.1f} M parameters, {len(ds_train)} train / {len(ds_val)} val rows, "
        f"{model.num_tokens()} tokens, compact layout {cfg.slot_tokens} from step {switch_step}",
        flush=True,
    )

    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": [p for n, p in params if n.startswith("trunk.")], "lr": args.lr_trunk},
            {"params": [p for n, p in params if not n.startswith("trunk.")], "lr": args.lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=args.wd,
    )
    base_lrs = [g["lr"] for g in opt.param_groups]
    step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step = ck["step"]
        if step >= switch_step:
            model.set_slot_tokens(cfg.slot_tokens)
        print(f"[train] resumed from {args.resume} at step {step}", flush=True)
    (run / "config.json").write_text(json.dumps({"args": vars(args), "config": cfg.to_dict()}, indent=1) + "\n")
    model.train()
    t0, best_val, it = time.time(), float("inf"), iter(dl_train)
    with open(run / "log.jsonl", "a") as log:
        while step < args.steps:
            if step == switch_step and switch_step > 0:
                model.set_slot_tokens(cfg.slot_tokens)
                print(f"[train] step {step}: compact layout ({model.num_tokens()} tokens)", flush=True)
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl_train)
                batch = next(it)
            b = prepare_batch(model, batch, device)
            for g, base in zip(opt.param_groups, base_lrs, strict=True):
                g["lr"] = base * lr_factor(step, args.warmup, args.steps)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(b["lats"], b["text_feat"], b["text_mask"], b["proprio"])
                total, losses = compute_losses(out, b)
            opt.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            phase = 1 if step <= switch_step else 2
            if step % 20 == 0 or step == 1:
                rec = {
                    "step": step,
                    "time_min": (time.time() - t0) / 60,
                    "lr": opt.param_groups[1]["lr"],
                    "grad_norm": float(grad_norm),
                    "phase": phase,
                    **{k: round(v.item(), 5) for k, v in losses.items()},
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % 100 == 0 or step == 1:
                    print(
                        f"step {step} | {rec['time_min']:.1f} min | "
                        + " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()),
                        flush=True,
                    )
            if step % args.val_every == 0 or step == args.steps:
                val = {**evaluate(model, dl_val, device, amp), "step": step, "phase": phase}
                log.write(json.dumps(val) + "\n")
                log.flush()
                print("[val] " + " ".join(f"{k}={v:.4f}" for k, v in val.items() if k.startswith("val_")), flush=True)
                if phase == 2 and val["val_action_l1"] < best_val:  # only the final layout is deployable
                    best_val = val["val_action_l1"]
                    save_checkpoint(run / "best.pt", model, args, step, val=val)
            if step % args.save_every == 0 or step == args.steps:
                save_checkpoint(run / "last.pt", model, args, step, optimizer=opt.state_dict())
    print(f"[train] done; best validation action L1 {best_val:.4f}", flush=True)
    return best_val


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train SmolWAM (released recipe by default).")
    ap.add_argument("--run", required=True, help="output directory")
    ap.add_argument("--bank", default="data/bank")
    ap.add_argument("--text_cache", default="data/text_cache.safetensors")
    ap.add_argument("--adapter", default="data/adapter.safetensors")
    ap.add_argument("--stats", default="data/LIBERO-Cosmos-Policy/success_only/dataset_statistics.json")
    ap.add_argument("--kd", default="data/kd", help="teacher targets (smolwam.train.kd_targets); 'none' disables KD")
    ap.add_argument("--init", default="teacher", help="teacher | none | path to a training checkpoint")
    ap.add_argument("--blocks", default="0,1,2,19,20,21,22,23,24,25,26,27", help="teacher blocks kept")
    ap.add_argument(
        "--slot_tokens", default="196,0,28,0,0", help="final layout: blank,proprio,action,fut_proprio,value"
    )
    ap.add_argument("--curriculum_frac", type=float, default=0.2, help="fraction of steps in the teacher's layout")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="learning rate of the new modules")
    ap.add_argument("--lr_trunk", type=float, default=2e-5, help="learning rate of the inherited trunk")
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--save_every", type=int, default=250)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max_rows", type=int, default=None, help="limit the training rows (debugging)")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="disable activation checkpointing")
    ap.add_argument("--resume", default=None, help="last.pt of an interrupted run")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.kd = None if args.kd == "none" else args.kd
    cfg = SmolWAMConfig(
        trunk_blocks=tuple(int(x) for x in args.blocks.split(",")),
        slot_tokens=tuple(int(x) for x in args.slot_tokens.split(",")),
    )
    train(args, cfg)


if __name__ == "__main__":
    main()
