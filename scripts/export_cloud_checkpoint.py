"""Export a vla-v4 trainable-only checkpoint without rewriting its training artifacts."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fr3_vla.cloud import export_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--replan-steps', type=int, default=4)
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output, args.training_root, args.dataset_root, args.replan_steps)
    print(args.output / 'fr3_deployment.json')


if __name__ == '__main__':
    main()
