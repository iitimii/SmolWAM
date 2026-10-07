#!/usr/bin/env bash
# LIBERO closed-loop success with the paper's protocol (seed 195).
#   bash experiments/eval_libero.sh full     # 40 tasks x 10 trials; the four suites run as parallel processes
#   bash experiments/eval_libero.sh screen   # 3 tasks per suite x 5 trials (60 episodes)
#   bash experiments/eval_libero.sh smoke    # 3 tasks per suite x 1 trial (12 episodes)
# Expected (one A100-40GB, bf16): full 382/400 = 95.5 % (Spatial 99, Object 100, Goal 94, Long 89);
# screen 56/60 = 93.3 % (100, 100, 93, 80); smoke 10/12. MODEL may be a repo id or a release directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROTOCOL="${1:-full}"
MODEL="${MODEL:-timiiowolabi/SmolWAM}"
OUT="${OUT:-outputs/libero_$PROTOCOL}"
mkdir -p "$OUT"

pids=()
for suite in libero_spatial libero_object libero_goal libero_10; do
  python -m smolwam.eval_libero --model "$MODEL" --protocol "$PROTOCOL" --suites "$suite" \
      --out "$OUT/$suite.json" > "$OUT/$suite.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
runs = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("libero_*.json"))]
episodes = [e for r in runs for e in r["episodes"]]
n_ok = sum(e["success"] for e in episodes)
print(f"{n_ok}/{len(episodes)} = {100 * n_ok / len(episodes):.1f} %  " +
      "  ".join(f"{s}: {100 * v:.0f}" for r in runs for s, v in r["per_suite"].items()))
PY
