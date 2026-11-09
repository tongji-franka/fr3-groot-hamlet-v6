"""Download pinned GR00T and the released trainable weights. Never contacts a robot."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = 'tongji-franka/fr3-groot-hamlet-v6'
BASE_REPO = 'nvidia/GR00T-N1.6-3B'
BASE_REVISION = 'd0814e7ecb19202e7c8468b46098b0b7ef3a6d61'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--repo', default=REPO)
    parser.add_argument('--revision', default='main', help='Pin a release commit for repeatable downloads')
    parser.add_argument('--base-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / 'weights/hamlet'
    snapshot_download(BASE_REPO, revision=BASE_REVISION, local_dir=base,
                      allow_patterns=['config.json', 'model-*.safetensors', 'model.safetensors.index.json',
                                      'processor_config.json', 'statistics.json', 'embodiment_id.json',
                                      'LICENSE', 'README.md'])
    if not args.base_only:
        snapshot_download(args.repo, revision=args.revision, local_dir=root,
                          allow_patterns=['trainable.safetensors', 'trainable.safetensors.sha256',
                                          'config.json', 'data_audit.json', 'training_contract.json',
                                          'COMPLETE', 'runtime/*.py', 'split.json'])
    (root / 'reports').mkdir(exist_ok=True)
    (root / 'reports/download_hamlet.json').write_text(json.dumps({
        'repo': BASE_REPO, 'revision': BASE_REVISION, 'path': str(base), 'status': 'ok'
    }, indent=2) + '\n')
    weights = root / 'trainable.safetensors'
    if weights.is_file():
        expected = (root / 'trainable.safetensors.sha256').read_text().split()[0]
        with weights.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('Trainable weights SHA256 mismatch')
    print('Assets ready:', root)


if __name__ == '__main__':
    main()
