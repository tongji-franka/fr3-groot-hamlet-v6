"""Verify the native binary against local control sources without contacting hardware."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def check(binary, root):
    source = root / 'tools/fr3_act'
    names = ('../fr3_replay/trajectory.hpp', 'CMakeLists.txt', 'bridge.cpp', 'build_info.hpp.in',
             'collision.hpp', 'limits.hpp.in', 'motion.hpp')
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    identity = ''.join(f'{name}:{digest(source/name)}\n' for name in names)
    package = root / 'ros2/src/franka_lerobot_recorder'
    tree = ast.parse((package / 'franka_lerobot_recorder/act_deploy.py').read_text())
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('PROTOCOL_VERSION', 'COMMAND_WATCHDOG_SECONDS')}
    expected = dict(protocol=constants['PROTOCOL_VERSION'], watchdog_seconds=constants['COMMAND_WATCHDOG_SECONDS'],
                    source_sha256=hashlib.sha256(identity.encode()).hexdigest(),
                    profile_sha256=digest(package / 'config/act_limits.json'))
    actual = json.loads(subprocess.check_output([str(binary), '--build-info'], text=True, timeout=5))
    if actual != expected:
        raise RuntimeError('Native binary differs from current sources/profile/protocol; rebuild with scripts/fr3/build_act.sh')
    return actual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.binary, Path(__file__).resolve().parents[2]), sort_keys=True))
