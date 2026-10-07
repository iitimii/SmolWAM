"""Run SmolWAM closed-loop for one LIBERO episode and save it as an MP4 (needs the libero extra:
scripts/install_libero.sh).

    python examples/libero_rollout.py --suite libero_10 --task 2 --out rollout.mp4
    python examples/libero_rollout.py --suite libero_goal --task 9 --model path/to/release
"""

import argparse
import os

import imageio

from smolwam import SmolWAM
from smolwam.envs.libero import TASK_MAX_STEPS, benchmark_suite, make_env, seed_everything
from smolwam.eval_libero import SEED, run_episode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=os.environ.get("SMOLWAM_MODEL", "timiiowolabi/SmolWAM"))
    ap.add_argument("--suite", default="libero_10", choices=list(TASK_MAX_STEPS))
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--trial", type=int, default=0, help="index of the initial state")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--out", default="rollout.mp4")
    args = ap.parse_args()

    seed_everything(SEED)
    policy = SmolWAM.from_pretrained(args.model, device=args.device, dtype=args.dtype)
    suite = benchmark_suite(args.suite)
    env, instruction = make_env(suite.get_task(args.task))
    success, steps, latencies, frames = run_episode(
        policy,
        env,
        instruction,
        suite.get_task_init_states(args.task)[args.trial],
        TASK_MAX_STEPS[args.suite],
        record="step",
    )
    env.close()
    imageio.mimsave(args.out, frames, fps=20)
    print(
        f"{instruction!r}: {'success' if success else 'failure'} after {steps} steps, "
        f"{len(latencies)} queries, {sum(latencies) / len(latencies):.2f} s per query -> {args.out}"
    )


if __name__ == "__main__":
    main()
