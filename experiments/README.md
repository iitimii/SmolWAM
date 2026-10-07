# Reproducing the paper's SmolWAM numbers

Every script runs from any directory, writes to `outputs/` (training: `data/` and `runs/`), and takes the model from
`MODEL` (a Hugging Face repo id or a local release directory; default `timiiowolabi/SmolWAM`). Expected values are
the paper's; they were measured on one NVIDIA A100-SXM4-40GB with PyTorch 2.7 (CUDA 12.8) in bf16.

| Experiment | Command | Expected result | Hardware, time |
|---|---|---|---|
| Data pipeline and training | `bash experiments/train_smolwam.sh` | bank of 68,830 rows; latent adapter held-out L1 0.088; teacher-vs-demo action L1 0.0094; best validation action L1 0.091 at step 5,000 | 1 × A100-40GB, about 5 h (training 3.6 h) |
| LIBERO, full protocol | `bash experiments/eval_libero.sh full` | 382/400 = 95.5 % (Spatial 99, Object 100, Goal 94, Long 89) | 1 × A100-40GB, 4 suites in parallel |
| LIBERO, screening set | `bash experiments/eval_libero.sh screen` | 56/60 = 93.3 % (100, 100, 93, 80) | 1 × A100-40GB |
| LIBERO, smoke test | `bash experiments/eval_libero.sh smoke` | 10/12 | 1 × A100-40GB |
| Inference cost | `bash experiments/bench.sh` | 0.878 B network parameters, 1,008 tokens, 2.30 TFLOP per query (1.63 network + 0.67 VAE), 57 ms per 16-action chunk, 2.36 GiB peak GPU memory | 1 × A100-40GB, idle |
| World head | `bash experiments/world_eval.sh` | latent L1 0.101 (copying the current frame: 0.133); primary camera LPIPS 0.128 (0.115), PSNR 24.7 dB (24.3), flow EPE 1.43 px (1.63) | 1 × A100-40GB |
| One episode on a Mac | `python -m smolwam.eval_libero --tasks libero_10:2 --num_trials 1 --device mps --dtype fp32 --out outputs/mac.json` | success after 239 steps (incl. 10 no-op steps), ~0.9 s per query | Apple M2 Pro |

Notes

- Evaluation protocol: seed 195, deterministic, 16-action chunks executed open-loop, 10 no-op steps before the first
  query, initial states 0..n-1 of each task. `screen` and `smoke` use tasks Spatial {0, 4, 8}, Object {0, 5, 9},
  Goal {3, 5, 9} and Long {3, 4, 8}.
- Latency is only meaningful from a process that has the GPU to itself; the four parallel suite processes of
  `eval_libero.sh` report contended latencies.
- Evaluating the teacher (`python -m smolwam.eval_libero --policy cosmos --protocol screen ...`; reported 58/60)
  requires the upstream [cosmos-policy](https://github.com/NVlabs/cosmos-policy) package (Linux, CUDA). Training does
  not: its distillation step only downloads the teacher's weights and cached T5 embeddings.
