#!/usr/bin/env bash
set -euo pipefail
fr3_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fr3_build="${FR3_ACT_BUILD_DIR:-$fr3_root/artifacts/fr3-act/build-container}"
cmake -S "$fr3_root/tools/fr3_act" -B "$fr3_build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$fr3_build" --parallel 2
ctest --test-dir "$fr3_build" --output-on-failure
python3 "$fr3_root/scripts/fr3/check_native_build.py" --binary "$fr3_build/fr3_act_sim"
if [[ -x "$fr3_build/fr3_act_bridge" ]]; then
  python3 "$fr3_root/scripts/fr3/check_native_build.py" --binary "$fr3_build/fr3_act_bridge"
fi
