#!/usr/bin/env bash
# World-head quality on 96 held-out contexts (needs data/bank from experiments/data.sh and the `train` extra).
# Expected (A100, bf16), SmolWAM vs copying the current frame: latent L1 0.101 vs 0.133; primary camera
# LPIPS 0.128 vs 0.115, PSNR 24.7 vs 24.3 dB, flow EPE 1.43 vs 1.63 px.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL="${MODEL:-timiiowolabi/SmolWAM}"
python -m smolwam.world_eval --model "$MODEL" --bank data/bank --out outputs/world_eval.json
python -m smolwam.world_eval --model "$MODEL" --bank data/bank --baseline copy --out outputs/world_eval_copy.json
