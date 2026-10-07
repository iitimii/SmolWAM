"""Inference cost of one policy query: parameters, FLOPs, latency (with its parts) and peak memory.

    python -m smolwam.bench --out outputs/bench.json                      # CUDA bf16 by default, else MPS / CPU fp32
    python -m smolwam.bench --device mps --dtype fp32 --n 50 --out bench_mps.json

A query is: preprocess both camera frames, encode them with the VAE, map them through the latent adapter, and run
the network once. Instruction features come from the cache, as in deployment. Latency is the wall time per query
after warm-up, synchronised with the device, from a process that has the device to itself.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

from smolwam.config import SmolWAMConfig
from smolwam.hub import DEFAULT_MODEL
from smolwam.layout import CAMERAS, CHUNK, IMAGE_SIZE, LATENT_CH, LATENT_HW, PROPRIO_DIM
from smolwam.model import SmolWAMModel
from smolwam.nn.wan_vae import WanVAE
from smolwam.policy import SmolWAM


def synthetic_observation(policy: SmolWAM, size: int = 256, seed: int = 0) -> dict:
    """Random camera frames and a mid-range robot state: the cost of a query does not depend on its content."""
    rng = np.random.default_rng(seed)
    stats = policy.normalizer.stats["proprio"]
    return {
        "wrist_image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        "primary_image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
        "proprio": (stats["min"] + stats["max"]) / 2,
    }


@torch.no_grad()
def count_flops(cfg: SmolWAMConfig) -> tuple[int, int]:
    """FLOPs of the network pass and of the VAE encode of both cameras, counted on shape-only (meta) copies with the
    math attention kernel, so the count is exact and the same on every device (fused attention kernels are not
    counted by FlopCounterMode on every backend)."""
    with torch.device("meta"):
        net, vae = SmolWAMModel(cfg).eval(), WanVAE().eval()
    net = net.to("meta")  # the fixed readout noise is drawn on the CPU
    lat = torch.zeros(1, len(CAMERAS), LATENT_CH, LATENT_HW, LATENT_HW, device="meta")
    text = torch.zeros(1, cfg.max_text_tokens, cfg.text_dim, device="meta")
    mask = torch.ones(1, cfg.max_text_tokens, dtype=torch.bool, device="meta")
    with FlopCounterMode(display=False) as network, sdpa_kernel(SDPBackend.MATH):
        net(lat, text, mask, torch.zeros(1, PROPRIO_DIM, device="meta"))
    with FlopCounterMode(display=False) as encoder, sdpa_kernel(SDPBackend.MATH):
        vae.encode(torch.zeros(len(CAMERAS), 3, 1, IMAGE_SIZE, IMAGE_SIZE, device="meta"))
    return network.get_total_flops(), encoder.get_total_flops()


def peak_memory_gib(device: torch.device) -> dict:
    if device.type == "cuda":
        return {"cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}
    if device.type == "mps":
        return {"mps_driver_allocated_gib": torch.mps.driver_allocated_memory() / 2**30}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--n", type=int, default=100, help="timed queries (after 5 warm-up queries)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    policy = SmolWAM.from_pretrained(args.model, device=args.device, dtype=args.dtype)
    obs, instruction = synthetic_observation(policy), policy.text_cache.instructions[0]
    for _ in range(5):
        policy.predict_action(obs, instruction)
    times = {k: [] for k in ("preprocess", "vae", "model", "total")}
    for _ in range(args.n):
        policy.predict_action(obs, instruction)
        for k in times:
            times[k].append(policy.last_timing[k])
    total = np.array(times["total"])

    flops_network, flops_vae = count_flops(policy.config)

    model = policy.model
    encoder_params = sum(p.numel() for n, p in policy.vae.named_parameters() if n.startswith(("encoder.", "conv1.")))
    res = {
        "model": str(args.model),
        "device": str(policy.device),
        "dtype": str(policy.dtype).replace("torch.", ""),
        "params_network": model.num_params(),
        "params_trunk": sum(p.numel() for p in model.trunk.parameters()),
        "params_vae_encoder": encoder_params,
        "params_latent_adapter": sum(p.numel() for p in model.adapter.parameters()),
        "num_tokens": model.num_tokens(),
        "flops_network_T": flops_network / 1e12,
        "flops_vae_encode_T": flops_vae / 1e12,
        "latency_mean_s": float(total.mean()),
        "latency_median_s": float(np.median(total)),
        "latency_p95_s": float(np.percentile(total, 95)),
        "latency_parts_s": {k: float(np.mean(v)) for k, v in times.items() if k != "total"},
        "action_hz": CHUNK / float(total.mean()),
        "policy_hz": 1.0 / float(total.mean()),
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (2**30 if sys.platform == "darwin" else 2**20),
        **peak_memory_gib(policy.device),
    }
    res["flops_query_T"] = res["flops_network_T"] + res["flops_vae_encode_T"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
