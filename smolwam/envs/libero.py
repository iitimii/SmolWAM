"""LIBERO environments with the Cosmos Policy evaluation conventions.

A faithful port of the helpers in cosmos_policy.experiments.robot.libero.libero_utils that the SmolWAM evaluation
uses: environment construction (256 px cameras, `env.seed(0)`), the no-op action, camera extraction (LIBERO renders
images upside down; they are flipped vertically, `np.flipud`) and the 9-d robot state.

LIBERO reads its paths from `$LIBERO_CONFIG_PATH/config.yaml` (default `~/.libero`) and asks for them interactively
when the file is missing. Unless LIBERO_CONFIG_PATH is set, a config pointing at the installed `libero` package is
written to ~/.cache/smolwam/libero and used, so evaluation never prompts and never picks up another checkout.
MuJoCo renders offscreen with CGL on macOS and EGL elsewhere unless MUJOCO_GL is set.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASK_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
NUM_STEPS_WAIT = 10  # no-op steps at the start of every episode while objects settle
RESOLUTION = 256


def _configure_libero() -> None:
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")  # offscreen rendering backend
    if "LIBERO_CONFIG_PATH" in os.environ:
        return
    import importlib.util

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("LIBERO is not installed: run scripts/install_libero.sh")
    root = Path(next(iter(spec.submodule_search_locations))) / "libero"
    config_dir = Path.home() / ".cache" / "smolwam" / "libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": root,
        "bddl_files": root / "bddl_files",
        "init_states": root / "init_files",
        "datasets": root.parent / "datasets",
        "assets": root / "assets",
    }
    (config_dir / "config.yaml").write_text("".join(f"{k}: {v}\n" for k, v in paths.items()))
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)


def benchmark_suite(name: str):
    """The LIBERO task suite object (tasks, initial states)."""
    _configure_libero()
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()[name]()


def make_env(task, resolution: int = RESOLUTION):
    """-> (OffScreenRenderEnv, task description) for a task of a suite."""
    _configure_libero()
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(0)  # the seed affects object placement even with a fixed initial state
    return env, task.language


def dummy_action() -> list[float]:
    """No-op: zero end-effector motion, gripper open."""
    return [0, 0, 0, 0, 0, 0, -1]


def primary_image(obs: dict) -> np.ndarray:
    return np.flipud(obs["agentview_image"])


def wrist_image(obs: dict) -> np.ndarray:
    return np.flipud(obs["robot0_eye_in_hand_image"])


def proprio(obs: dict) -> np.ndarray:
    """Gripper joint positions (2), end-effector position (3) and quaternion (4)."""
    return np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]))


def policy_observation(obs: dict) -> dict:
    """LIBERO observation -> the input of `SmolWAM.predict_action`."""
    return {"primary_image": primary_image(obs), "wrist_image": wrist_image(obs), "proprio": proprio(obs)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
