"""Run inference on local observations; save absolute joint angles without execution."""
import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy import HamletPolicy, ROOT, TASK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='NPZ with state [7] and images [4,3,H,W,3]')
    parser.add_argument('--output', type=Path, default=Path('actions.npy'))
    parser.add_argument('--checkpoint', type=Path, default=ROOT)
    parser.add_argument('--training-root', type=Path, default=ROOT)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task', default=TASK)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new --output')
    with np.load(args.input, allow_pickle=False) as sample:
        state, images = sample['state'], sample['images']
    policy = HamletPolicy(args.checkpoint, args.training_root, args.device)
    started = time.perf_counter()
    actions = policy.predict(state, images, task=args.task, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.save(stream, actions)
    print(f'Saved {actions.shape} absolute joint radians to {args.output}; inference {time.perf_counter()-started:.3f}s')


if __name__ == '__main__':
    main()
