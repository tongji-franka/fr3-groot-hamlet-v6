import hashlib
import json
from pathlib import Path
import sys
import unittest
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy import prepare_observation, HamletPolicy
from fr3_vla.cloud import load_sources
from scripts.make_manifest import release_files


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = load_sources(ROOT, ROOT)
        stats = json.loads((ROOT / 'data_audit.json').read_text())['normalization']
        cls.norm = cls.data.Normalizer(stats, 'bounds')

    def test_training_sources_match_checkpoint(self):
        hashes = json.loads((ROOT / 'config.json').read_text())['source_sha256']
        for name in ['train.py', 'training_support.py', 'checkpoint_shards.py', 'adapters.py', 'v4_data.py']:
            self.assertEqual(hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest(), hashes['scripts/' + name])

    def test_preprocessing_order_and_normalization(self):
        images = np.zeros((4, 3, 40, 80, 3), np.uint8)
        images[2, 1] = 255
        state = np.array(self.norm.stats['observation.state']['mean'], np.float32)
        batch = prepare_observation(self.data, self.norm, state, images, 'test', 'cpu')
        self.assertEqual(tuple(batch['images'].shape), (1, 4, 3, 3, 224, 224))
        self.assertEqual(float(batch['images'][0, 2, 1, :, 60:164].min()), 1.)
        self.assertEqual(float(batch['images'][0, 0].max()), 0.)
        torch.testing.assert_close(self.norm.undo(batch['state'][0], 'observation.state'), torch.from_numpy(state))
        self.assertEqual(tuple(batch['action'].shape), (1, 50, 7))

    def test_input_validation(self):
        for state, images in [(np.zeros(6), np.zeros((4, 3, 4, 4, 3), np.uint8)),
                              (np.zeros(7), np.zeros((4, 3, 4, 4, 3), np.float32)),
                              (np.full(7, np.nan), np.zeros((4, 3, 4, 4, 3), np.uint8))]:
            with self.assertRaises(ValueError):
                prepare_observation(self.data, self.norm, state, images, 'test', 'cpu')

    def test_cpu_mixed_precision_boundary(self):
        class MixedHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(7, 7)

            def predict(self, batch):
                return self.linear(torch.zeros(1, 50, 7, dtype=torch.bfloat16))

        policy = HamletPolicy.__new__(HamletPolicy)
        policy.device, policy.data, policy.normalizer = torch.device('cpu'), self.data, self.norm
        policy.model = MixedHead()
        actions = policy.predict(np.zeros(7), np.zeros((4, 3, 8, 8, 3), np.uint8), seed=42)
        self.assertEqual(actions.shape, (50, 7))
        self.assertTrue(np.isfinite(actions).all())

    def test_manifest_excludes_local_assets_and_keeps_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['README.md', 'trainable.safetensors', 'weights/base.bin',
                         '.venv-upload/a.py', 'runtime/__pycache__/a.pyc',
                         'provenance/files.sha256', 'robot/artifacts/build.bin']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / 'local_link').symlink_to(root / 'README.md')
            self.assertEqual({p.relative_to(root).as_posix() for p in release_files(root)},
                             {'README.md', 'trainable.safetensors'})


if __name__ == '__main__':
    unittest.main()
