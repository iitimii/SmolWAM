"""Closed-loop LIBERO evaluation of SmolWAM (or the Cosmos Policy teacher) with the paper's protocol.

    python -m smolwam.eval_libero --protocol screen --out outputs/screen.json                 # 60 episodes
    python -m smolwam.eval_libero --protocol full --suites libero_goal --out outputs/goal.json  # one suite of the 400
    python -m smolwam.eval_libero --tasks libero_10:2,libero_goal:9 --num_trials 1 --device mps --out x.json
    python -m smolwam.eval_libero --policy cosmos --protocol screen --out outputs/cosmos_screen.json   # Linux + CUDA

Protocol (fixed for every number in the paper): seed 195, deterministic, 16-action chunks executed open-loop, 10
no-op steps before the first query, per-suite step limits, JPEG + flip preprocessing as in Cosmos Policy.
  smoke   3 tasks per suite x 1 trial   (12 episodes)
  screen  3 tasks per suite x 5 trials  (60 episodes; includes the tasks an earlier pruned student failed most)
  full    all 40 tasks x 10 trials      (400 episodes)
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from smolwam.envs.libero import (
    NUM_STEPS_WAIT,
    SUITES,
    TASK_MAX_STEPS,
    benchmark_suite,
    dummy_action,
    make_env,
    policy_observation,
    primary_image,
    seed_everything,
    wrist_image,
)
from smolwam.hub import DEFAULT_MODEL
from smolwam.layout import CHUNK

SCREEN_TASKS = {
    "libero_spatial": [0, 4, 8],
    "libero_object": [0, 5, 9],
    "libero_goal": [3, 5, 9],
    "libero_10": [3, 4, 8],
}
PROTOCOLS = {"smoke": (SCREEN_TASKS, 1), "screen": (SCREEN_TASKS, 5), "full": (None, 10)}
SEED = 195


def run_episode(
    policy,
    env,
    instruction: str,
    initial_state,
    max_steps: int,
    record: str | None = None,
    video_res: int | None = None,
) -> tuple[bool, int, list[float], list[np.ndarray]]:
    """One episode. record: None, "chunk" (primary frame per policy query) or "step" (every control step: primary |
    wrist, or the primary camera rendered at `video_res` px; the policy always sees the 256 px observations).
    Returns success, steps taken (incl. the no-op steps), per-query latencies and the recorded frames."""
    env.reset()
    obs = env.set_init_state(initial_state)
    queue: deque = deque(maxlen=CHUNK)
    t, success, latencies, frames = 0, False, [], []
    while t < max_steps + NUM_STEPS_WAIT:
        seed_everything(0)
        if t < NUM_STEPS_WAIT:
            obs, _, _, _ = env.step(dummy_action())
            t += 1
            continue
        if not queue:
            o = policy_observation(obs)
            queue.extend(policy.predict_action(o, instruction))
            latencies.append(policy.last_timing["total"])
            if record == "chunk":
                frames.append(o["primary_image"])
        obs, _, done, _ = env.step(np.asarray(queue.popleft(), dtype=np.float64).tolist())
        t += 1
        if record == "step" and video_res:
            frames.append(
                np.ascontiguousarray(env.sim.render(camera_name="agentview", height=video_res, width=video_res)[::-1])
            )
        elif record == "step":
            frames.append(np.concatenate((primary_image(obs), wrist_image(obs)), axis=1))
        if done:
            success = True
            break
    return success, t, latencies, frames


def select_tasks(args) -> tuple[dict[str, list[int] | None], int]:
    """-> {suite: task ids (None = all)}, trials per task."""
    if args.protocol:
        tasks, trials = PROTOCOLS[args.protocol]
        suites = args.suites.split(",") if args.suites else SUITES
        return {s: (tasks[s] if tasks else None) for s in suites}, trials
    if args.tasks:
        selected: dict[str, list[int]] = {}
        for item in args.tasks.split(","):
            suite, tid = item.split(":")
            selected.setdefault(suite, []).append(int(tid))
        return selected, args.num_trials
    task_ids = [int(x) for x in args.task_ids.split(",")] if args.task_ids else None
    return {s: task_ids for s in (args.suites.split(",") if args.suites else SUITES)}, args.num_trials


def load_policy(args):
    if args.policy == "cosmos":
        from smolwam.teacher import CosmosPolicy

        return CosmosPolicy(num_steps=args.cosmos_steps, seed=args.seed)
    from smolwam.policy import SmolWAM

    return SmolWAM.from_pretrained(args.model, device=args.device, dtype=args.dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="Closed-loop LIBERO evaluation.")
    ap.add_argument("--policy", default="smolwam", choices=["smolwam", "cosmos"])
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local release directory")
    ap.add_argument("--protocol", choices=list(PROTOCOLS), help="smoke | screen | full (sets tasks and trials)")
    ap.add_argument("--suites", default=None, help="comma list of suites (default: all four)")
    ap.add_argument("--task_ids", default=None, help="comma list of task ids within --suites (default: all)")
    ap.add_argument("--tasks", default=None, help="explicit suite:task list, e.g. libero_10:2,libero_goal:9")
    ap.add_argument("--num_trials", type=int, default=10, help="trials per task (overridden by --protocol)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: the best available)")
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"], help="default: bf16 on CUDA, else fp32")
    ap.add_argument("--cosmos_steps", type=int, default=5, help="denoising steps of the Cosmos Policy teacher")
    ap.add_argument("--save_video_dir", default=None)
    ap.add_argument("--video_every_step", action="store_true", help="video at the 20 Hz control rate (primary | wrist)")
    ap.add_argument("--video_res", type=int, default=None, help="with --video_every_step: primary camera at this size")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seed_everything(args.seed)
    selected, trials = select_tasks(args)
    policy = load_policy(args)
    record = ("step" if args.video_every_step else "chunk") if args.save_video_dir else None
    results = {"args": vars(args), "episodes": [], "per_task": {}, "per_suite": {}}
    t_start, all_latencies = time.time(), []
    for suite, task_ids in selected.items():
        task_suite = benchmark_suite(suite)
        suite_success, suite_n = 0, 0
        for tid in task_ids if task_ids is not None else range(task_suite.n_tasks):
            init_states = task_suite.get_task_init_states(tid)
            env, instruction = make_env(task_suite.get_task(tid))
            n_success = 0
            for ep in range(trials):
                ok, steps, latencies, frames = run_episode(
                    policy, env, instruction, init_states[ep], TASK_MAX_STEPS[suite], record, args.video_res
                )
                n_success += int(ok)
                all_latencies += latencies
                results["episodes"].append(
                    {
                        "suite": suite,
                        "task": tid,
                        "instruction": instruction,
                        "trial": ep,
                        "success": ok,
                        "steps": steps,
                        "mean_latency_s": float(np.mean(latencies)),
                    }
                )
                print(
                    f"[{suite} t{tid} ep{ep}] {'OK  ' if ok else 'FAIL'} steps={steps} "
                    f"latency={np.mean(latencies):.3f}s | {instruction}",
                    flush=True,
                )
                if frames:
                    import imageio

                    path = Path(args.save_video_dir) / f"{suite}_t{tid}_ep{ep}_{'ok' if ok else 'fail'}.mp4"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.mimsave(path, frames, fps=20 if record == "step" else 8)
            env.close()
            results["per_task"][f"{suite}/{tid}"] = n_success / trials
            suite_success, suite_n = suite_success + n_success, suite_n + trials
        results["per_suite"][suite] = suite_success / suite_n
        print(f"== {suite}: {suite_success}/{suite_n} = {suite_success / suite_n:.3f}", flush=True)

    lat = np.array(all_latencies[5:] if len(all_latencies) > 5 else all_latencies)  # drop warm-up queries
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == "darwin" else 2**20)
    results.update(
        success_rate=float(np.mean([e["success"] for e in results["episodes"]])),
        num_episodes=len(results["episodes"]),
        latency_mean_s=float(lat.mean()),
        latency_median_s=float(np.median(lat)),
        action_hz=CHUNK / float(lat.mean()),
        wall_min=(time.time() - t_start) / 60,
        cuda_peak_mem_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
        peak_rss_gib=rss,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
    print(
        f"FINAL success {results['success_rate']:.3f} over {results['num_episodes']} episodes | "
        f"latency {results['latency_mean_s']:.3f}s ({results['action_hz']:.1f} actions/s) | {results['per_suite']}"
    )


if __name__ == "__main__":
    main()
