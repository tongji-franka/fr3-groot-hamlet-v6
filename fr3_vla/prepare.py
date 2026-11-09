"""Generate v4 training contracts and device-free layout previews."""
import argparse
import json

import numpy as np

from .contract import CAMERAS, DATASET, JOINTS, MODELS, ROOT, digest, validate_actions, write_json
from .data import V4Dataset, training_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=str, default=str(DATASET))
    parser.add_argument('--output', type=str, default=str(ROOT / 'artifacts'))
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    from pathlib import Path
    output = Path(args.output)
    dataset = V4Dataset(args.dataset_root)
    stats = dataset.statistics()
    for key in ('observation.state', 'action'):
        validate_actions(np.array([stats[key]['min'], stats[key]['max']]))
    write_json(output / 'v4_stats.json', stats)
    settings = {'xvla': (30, 1), 'flower': (10, 1), 'hamlet': (16, 4), 'rdt': (64, 1), 'pi05': (50, 1), 'wsa': (50, 1)}
    report = dict(contacts_robot=False, loads_model=False, dataset_root=str(dataset.root),
                  episodes=len(dataset.episodes), frames=int(dataset.episodes.length.sum()),
                  task=dataset.task, models={})
    state, images, actions = dataset.sample(horizon=64)
    validate_actions(actions)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'v4_sample.npz', state=state, actions=actions,
                        **dict(zip(CAMERAS, images, strict=True)))
    for model in ([args.model] if args.model else MODELS):
        horizon, replan = settings[model]
        inference = {
            'xvla': dict(action_mode='auto', real_action_dim=7, max_action_dim=20, domain_id=29, steps=10),
            'flower': dict(normalization='minmax', action_type='joint_single', model_action_dim=8,
                           dummy_gripper=0, use_proprio=True, image_size=224,
                           image_transform='letterbox_clip', camera_order=list(CAMERAS)),
            'hamlet': dict(embodiment='NEW_EMBODIMENT', hamlet_mode='finetune', memory_window=4,
                           memory_stride=4, n_moment_tokens=4, action_key='joint_position'),
            'rdt': dict(state_dim=128, joint_indices=list(range(7)), control_frequency=30,
                        camera_order=['overhead', 'wrist', 'third'], image_history='duplicate_current'),
            'pi05': dict(backend='lerobot', policy_type='pi05', checkpoint_format='lora',
                         project='/home/user1/projects/pi05-fr3-hockey'),
            'wsa': dict(backend='wsa_base', action_mode='abs', normalization='mean_std',
                        model_state_dim=32, model_action_dim=32, valid_action_dim=7,
                        image_size=224, camera_order=['overhead', 'wrist', 'third'],
                        image_delta_indices=[-15, 0, 15], steps=10),
        }[model]
        manifest = dict(schema_version=1, model=model, fps=30, joint_names=JOINTS,
            action_units='rad', action_definition=dataset.contract['action_definition'],
            camera_order=list(CAMERAS), cameras=dataset.contract['cameras'],
            dataset_repo='tongji-franka/hockey_three_cameras_aligned_v4',
            contract_sha256=digest(dataset.root / 'recording/contract.json'),
            info_sha256=digest(dataset.root / 'meta/info.json'), task=dataset.task,
            horizon=horizon, replan_steps=replan, statistics=stats, inference=inference,
            weights_sha256={}, status='training_template_not_a_trained_checkpoint')
        write_json(ROOT / f'models/{model}/configs/fr3_deployment.template.json', manifest)
        sample = training_sample(dataset, model, 0, 0, horizon, stats)
        report['models'][model] = dict(horizon=horizon, replan_steps=replan,
                keys=list(sample), contract_prepared=True, trained_inference_tested=False)
    write_json(output / 'preparation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
