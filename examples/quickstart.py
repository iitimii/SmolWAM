"""Predict one action chunk from a real LIBERO observation and report the query time.

    python examples/quickstart.py                                   # downloads timiiowolabi/SmolWAM
    python examples/quickstart.py --model path/to/release --device cpu

`--model` (or $SMOLWAM_MODEL) also accepts a local release directory.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from smolwam import SmolWAM

SAMPLE = Path(__file__).parent / "sample_obs"


def load_sample(sample_dir: Path = SAMPLE) -> tuple[dict, str]:
    """The bundled observation: LIBERO-Long task 2, initial state 0, first policy query of the episode."""
    meta = json.loads((sample_dir / "obs.json").read_text())
    obs = {
        "primary_image": np.asarray(Image.open(sample_dir / "primary.png").convert("RGB")),
        "wrist_image": np.asarray(Image.open(sample_dir / "wrist.png").convert("RGB")),
        "proprio": np.asarray(meta["proprio"]),
    }
    return obs, meta["instruction"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.environ.get("SMOLWAM_MODEL", "timiiowolabi/SmolWAM"))
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: the best available)")
    ap.add_argument("--dtype", default=None, help="bf16 | fp32 (default: bf16 on CUDA, fp32 elsewhere)")
    args = ap.parse_args()

    policy = SmolWAM.from_pretrained(args.model, device=args.device, dtype=args.dtype)
    obs, instruction = load_sample()
    actions = policy.predict_action(obs, instruction)  # first query includes one-off warm-up costs
    for _ in range(3):
        actions = policy.predict_action(obs, instruction)

    np.set_printoptions(precision=4, suppress=True)
    print(f"instruction: {instruction!r}")
    print(f"action chunk {actions.shape} (xyz delta, rotation delta, gripper):\n{actions}")
    t = policy.last_timing
    print(
        f"query on {policy.device} ({str(policy.dtype).replace('torch.', '')}): {t['total'] * 1000:.0f} ms "
        f"(preprocess {t['preprocess'] * 1000:.0f}, VAE {t['vae'] * 1000:.0f}, network {t['model'] * 1000:.0f})"
    )


if __name__ == "__main__":
    main()
