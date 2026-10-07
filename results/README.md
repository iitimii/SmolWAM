# Results

The records behind every number in the README and the paper. Values are copied unchanged from the evaluation, benchmark
and training logs of the released model; only internal run names and file paths were removed.

| File | Contents |
|---|---|
| `libero_full_smolwam.json` | SmolWAM on the full LIBERO protocol: 400 episodes (40 tasks × 10 trials), every episode with its success and step count; 382/400 = 95.5% |
| `libero_screen.json` | The 60-episode screening set (tasks Spatial {0, 4, 8}, Object {0, 5, 9}, Goal {3, 5, 9}, Long {3, 4, 8}; 5 trials each) for SmolWAM (93.3%), Cosmos Policy 2B with 5 denoising steps (96.7%) and Cosmos Policy run for one step on single-frame latents (100%) |
| `efficiency_a100.json` | Parameters, FLOPs, latency (mean, median, p95 and its breakdown), throughput and peak memory per 16-action chunk |
| `world_model.json` | World-head metrics on 96 held-out contexts for SmolWAM and the copy-current-frame baseline |
| `variants.json` | Variants reported in the paper: longer training, simulated INT8 weights, 8 blocks, and the latent-action cascade |
| `training/config.json`, `training/log.jsonl` | The training configuration and the loss and validation curves of the released model |

Protocol: seed 195, LIBERO's fixed initial states, 16-action chunks executed open loop, 10 no-op steps before the first
query. Costs were measured on one NVIDIA A100-SXM4-40GB (PyTorch 2.7, CUDA 12.8, bf16, batch 1) by a process that had
the GPU to itself; success rates come from runs with the four suites in parallel. Cosmos Policy's 400-episode result
(98.2%) comes from an earlier evaluation pipeline and has no record here. [`experiments/`](../experiments) has the
command that regenerates each file.
