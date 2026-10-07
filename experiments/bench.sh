#!/usr/bin/env bash
# Inference cost of one policy query (batch 1). Run on an otherwise idle GPU.
# Expected on one A100-40GB, bf16: 0.878 B network parameters, 1,008 tokens, 2.30 TFLOP per query (1.63 network +
# 0.67 VAE encode), 57 ms per 16-action chunk, 2.36 GiB peak GPU memory.
# On an Apple M2 Pro (MPS, fp32) a query takes about 0.9 s.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL="${MODEL:-timiiowolabi/SmolWAM}"
python -m smolwam.bench --model "$MODEL" --n 100 --out outputs/bench.json "$@"
