#!/usr/bin/env bash
# Install the LIBERO simulation stack into the active Python environment (macOS and Linux).
#
#   bash scripts/install_libero.sh            # uses `uv pip` if available, else `pip`
#
# robosuite 1.4.0 + MuJoCo 3.3.2 come from the `libero` extra. LIBERO itself (the PyPI package `libero==0.1.1`) is
# installed with --no-deps: its declared dependencies pull robomimic -> egl-probe, which does not build on macOS, and
# none of them is needed to run the benchmark. Its object meshes (~0.4 GB) are downloaded from the Hugging Face Hub to
# ~/.cache/libero/assets on first use. SmolWAM writes LIBERO's path config itself (smolwam/envs/libero.py), so the
# interactive first-run prompt never appears. Rendering uses CGL on macOS and EGL on Linux (MUJOCO_GL).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
if command -v uv >/dev/null 2>&1; then PIP=(uv pip install --python "$(command -v "$PY")"); else PIP=("$PY" -m pip install); fi

"${PIP[@]}" -e ".[libero]"
"${PIP[@]}" --no-deps "libero==0.1.1"

"$PY" - <<'EOF'
from smolwam.envs.libero import benchmark_suite
suite = benchmark_suite("libero_10")
print(f"LIBERO ready: libero_10 has {suite.n_tasks} tasks")
EOF
