#!/usr/bin/env bash
# Reproduce the released SmolWAM from the LIBERO demonstrations, on one A100-40GB (about 5 h in total):
#   latent bank (8 min) -> latent adapter (12 min of pairs, then a short fit) -> text cache -> Cosmos Policy targets
#   (66 min) -> training (3.6 h, 5,000 steps: 1,000 in the teacher's layout, then the compact layout) -> export.
# Needs the `train` extra and the `hf` CLI (huggingface_hub). Writes data/ and runs/smolwam/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUN="${RUN:-runs/smolwam}"

hf download nvidia/LIBERO-Cosmos-Policy --repo-type dataset --include "success_only/*" --local-dir data/LIBERO-Cosmos-Policy
# 1,700 successful demonstrations, every 4th step, both cameras -> 68,830 rows of single-frame VAE latents
python -m smolwam.train.bank --data data/LIBERO-Cosmos-Policy/success_only --out data/bank
# latent adapter: held-out L1 0.088 (an affine map alone: 0.48)
python -m smolwam.train.adapter pairs --bank data/bank --n 6000 --out data/latent_pairs.npz
python -m smolwam.train.adapter fit --pairs data/latent_pairs.npz --out data/adapter.safetensors
# BERT-base features of the 40 instructions
python -m smolwam.text --bank data/bank --out data/text_cache.safetensors
# Cosmos Policy's one-step targets on every row (downloads the 3.9 GB teacher); teacher-vs-demo action L1 0.0094
python -m smolwam.train.kd_targets --bank data/bank --adapter data/adapter.safetensors --out data/kd
# the released recipe is the default of every flag; expected best validation action L1 0.091 at step 5,000
python -m smolwam.train.train --run "$RUN"
python -m smolwam.export --ckpt "$RUN/best.pt" --out "$RUN/release"
