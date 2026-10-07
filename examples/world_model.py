"""Decode SmolWAM's forecast of both camera views 16 steps ahead, next to the current views. Saves a PNG.

    python examples/world_model.py --out world_model.png
    python examples/world_model.py --model path/to/release --device cpu

The network predicts the future latents in the same forward pass as the actions; at deployment they are not
decoded. Here they are, with the Wan VAE decoder: columns are the current view and the forecast, rows the wrist and
third-person cameras.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from quickstart import load_sample

from smolwam import SmolWAM
from smolwam.layout import CAMERAS
from smolwam.preprocessing import preprocess_env_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.environ.get("SMOLWAM_MODEL", "timiiowolabi/SmolWAM"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--out", default="world_model.png")
    args = ap.parse_args()

    policy = SmolWAM.from_pretrained(args.model, device=args.device, dtype=args.dtype)
    obs, instruction = load_sample()
    actions, future = policy.predict_action(obs, instruction, return_future=True)
    forecast = policy.decode_future(future)

    size, label_h = 224, 22
    grid = Image.new("RGB", (2 * size, len(CAMERAS) * size + label_h), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((8, 5), "current view", fill="black")
    draw.text((size + 8, 5), "forecast, 16 steps ahead", fill="black")
    for row, cam in enumerate(CAMERAS):
        current = preprocess_env_frame(obs[f"{cam}_image"])  # what the model sees
        for col, img in enumerate((current, forecast[f"{cam}_image"])):
            grid.paste(Image.fromarray(np.asarray(img)), (col * size, label_h + row * size))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"{instruction!r}: saved {args.out} (rows: {', '.join(CAMERAS)})")


if __name__ == "__main__":
    main()
