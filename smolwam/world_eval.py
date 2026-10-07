"""World-head quality on held-out demonstrations: the predicted view 16 steps ahead against the real one.

    python -m smolwam.world_eval --bank data/bank --out outputs/world_eval.json
    python -m smolwam.world_eval --bank data/bank --baseline copy --out outputs/world_eval_copy.json

For 96 contexts spread evenly over the validation episodes of the bank, the network's future latents (mapped back
to single-frame latents by the inverse adapter) are compared with the bank's latents of the real future frame, and
both are decoded by the VAE for pixel metrics. The copy baseline predicts the current frame. Metrics: latent L1;
per camera, LPIPS (AlexNet), PSNR, and the end-point error between RAFT optical flow from the current frame to the
prediction and to the real future (flow EPE, pixels, capped at 16). Needs the `train` extra (lpips, torchvision).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from smolwam.hub import DEFAULT_MODEL
from smolwam.layout import CAMERAS
from smolwam.nn.wan_vae import WanVAE
from smolwam.policy import SmolWAM
from smolwam.train.data import BankDataset
from smolwam.train.train import to_device


class ImageMetrics:
    """LPIPS, PSNR and RAFT flow end-point error on (B, 3, H, W) images in [0, 255]."""

    def __init__(self, device: torch.device):
        import lpips
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def flow(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.raft(a / 127.5 - 1, b / 127.5 - 1, num_flow_updates=12)[-1]

    @torch.no_grad()
    def __call__(self, cur: torch.Tensor, pred: torch.Tensor, real: torch.Tensor) -> dict[str, torch.Tensor]:
        epe = (self.flow(cur, pred) - self.flow(cur, real)).norm(dim=1).clamp(max=16.0).mean(dim=(1, 2))
        mse = ((pred - real) ** 2).mean(dim=(1, 2, 3)).clamp(min=1e-6)
        return {
            "lpips": self.lpips(pred / 127.5 - 1, real / 127.5 - 1).flatten(),
            "psnr": 10 * torch.log10(255.0**2 / mse),
            "flow_epe": epe,
        }


def decode(vae: WanVAE, latents: torch.Tensor) -> torch.Tensor:
    """(N, 16, 28, 28) single-frame latents -> (N, 3, 224, 224) float images in [0, 255]."""
    x = vae.decode(latents.to(torch.bfloat16).unsqueeze(2))[:, :, 0].float()
    return (x.clamp(-1, 1) + 1) * 127.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bank", default="data/bank")
    ap.add_argument("--baseline", choices=["copy"], default=None, help="evaluate copying the current frame instead")
    ap.add_argument("--n", type=int, default=96, help="held-out contexts")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    policy = SmolWAM.from_pretrained(args.model, device=args.device, dtype=args.dtype)
    model, device, dtype = policy.model, policy.device, policy.dtype
    ds = BankDataset(args.bank, policy.text_cache, policy.normalizer, split="val")
    contexts = np.linspace(0, len(ds) - 1, args.n).astype(int).tolist()
    loader = DataLoader(Subset(ds, contexts), batch_size=args.bs, shuffle=False, num_workers=2)
    metrics = ImageMetrics(device)
    sums, n = {}, 0
    for batch in loader:
        b = to_device(batch, device)
        cur, real = b["lats"], b["fut_lats"]
        if args.baseline == "copy":
            pred = cur
        else:
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                z = model.to_trunk_space(cur)
                out = model(z.to(dtype), b["text_feat"].to(dtype), b["text_mask"], b["proprio"].to(dtype))
                pred = model.to_single_space(out["fut_latents"].float())
        res = {"latent_l1": (pred - real).abs().mean(dim=(1, 2, 3, 4))}
        for c, cam in enumerate(CAMERAS):
            imgs = [decode(policy.vae, x[:, c]) for x in (cur, pred, real)]
            res.update({f"{cam}_{k}": v for k, v in metrics(*imgs).items()})
        for k, v in res.items():
            sums[k] = sums.get(k, 0.0) + v.float().sum().item()
        n += cur.shape[0]
    out = {k: v / n for k, v in sums.items()}
    out.update(model=str(args.model), baseline=args.baseline, n=n)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
