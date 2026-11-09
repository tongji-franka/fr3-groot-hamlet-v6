"""Compare deployment against the training dataset/adapter on real, historical RGB frames."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import default_collate
from fr3_vla.adapters import Agent
from fr3_vla.contract import write_json, digest, LOWER, UPPER
from fr3_vla.data import V4Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=int, default=38088)
    args = parser.parse_args()
    torch.set_num_threads(4)
    agent = Agent(args.model, args.checkpoint, args.dataset_root, 'cuda', args.training_root)
    image_size = agent.backend.recipe['image_size']
    image_options = {'image_size': image_size} if isinstance(image_size, (list, tuple)) else {}
    reference = agent.backend.data.V4Dataset(args.model, root=args.dataset_root, **image_options)
    saved_audit = json.loads((args.checkpoint / 'data_audit.json').read_text())
    subset_options = {'allow_source_subset': True} if len(saved_audit.get('datasets', [])) > 1 else {}
    reference.restore_stats(args.checkpoint, **subset_options)
    item = reference[args.index]
    batch = default_collate([item])
    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    dataset = V4Dataset(args.dataset_root)
    episode = item['episode_index']
    first_index = int(dataset.episodes.loc[episode]['dataset_from_index'])
    frame_index = args.index - first_index
    for offset in agent.backend.recipe['image_delta_indices']:
        frame = max(0, frame_index + offset)
        state, images, _ = dataset.sample(episode, frame, agent.manifest['horizon'])
        agent.backend.history.observe(images, frame / 30)
    prepared = agent.backend.prepare(state, images)
    torch.testing.assert_close(prepared['state'], batch['state'], rtol=0, atol=1e-6)
    torch.testing.assert_close(prepared['images'], batch['images'], rtol=0, atol=1e-6)
    if prepared['task'] != batch['task']:
        raise AssertionError('Task text differs')
    torch.manual_seed(123)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = agent.backend.norm.undo(agent.backend.model.predict(batch)[0].float()).cpu().numpy()
    torch.cuda.synchronize()
    torch.manual_seed(123)
    begin = time.perf_counter()
    motion_admission = 'passed'
    rejection = None
    try:
        actual = agent.predict_chunk(state, *images)
    except ValueError as error:
        if str(error) != 'Predicted absolute radians exceed finite FR3 joint limits':
            raise
        # Preserve the rejection; inspect the raw result only for offline numerical parity.
        motion_admission, rejection = 'rejected_joint_limits', str(error)
        torch.manual_seed(123)
        begin = time.perf_counter()
        actual = agent.backend.predict(state, images, None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - begin
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    invalid = (actual < LOWER) | (actual > UPPER) | ~np.isfinite(actual)
    report = dict(status='passed' if motion_admission == 'passed' else 'parity_passed_motion_rejected',
                  parity_status='passed', motion_admission=motion_admission, rejection=rejection,
                  model=args.model, contacts_robot=False,
                  checkpoint=str(args.checkpoint), manifest_sha256=digest(args.checkpoint / 'fr3_deployment.json'),
                  episode=episode, frame_index=frame_index, input_index=args.index,
                  image_delta_indices=agent.backend.recipe['image_delta_indices'],
                  image_shape=list(prepared['images'].shape), action_shape=list(actual.shape),
                  state_max_abs_error=float((prepared['state']-batch['state']).abs().max()),
                  image_max_abs_error=float((prepared['images']-batch['images']).abs().max()),
                  action_max_abs_error=float(np.max(np.abs(actual-expected))),
                  inference_seconds=seconds, joint_limits_passed=not bool(invalid.any()),
                  invalid_action_coordinates=int(invalid.sum()),
                  action_min_rad=actual.min(axis=0).tolist(), action_max_rad=actual.max(axis=0).tolist(),
                  target_actions_supplied_to_deployment=False,
                  legacy_runtime_migration=agent.manifest['legacy_runtime_migration'])
    write_json(args.output, report)
    write_json(args.checkpoint / 'offline_parity.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
