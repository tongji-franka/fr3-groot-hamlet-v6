#!/usr/bin/env bash
# Pass dataset paths and --init-from explicitly. Does not restore missing optimizer shards.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root/models/lerobot/src:$root${PYTHONPATH:+:$PYTHONPATH}"
export NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$root"
if [[ "${NUM_GPUS:-1}" -gt 1 ]]; then
  exec torchrun --standalone --nproc_per_node="$NUM_GPUS" scripts/train.py \
    --model hamlet --ddp --zero-optimizer --sharded-checkpoint "$@"
fi
exec python scripts/train.py --model hamlet "$@"
