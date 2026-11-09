"""Pass this file as --modality-config-path for cloud GR00T N1.6 + HAMLET training."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

fr3_v4_joint = {
    'video': ModalityConfig(delta_indices=[0], modality_keys=['wrist', 'overhead', 'third']),
    'state': ModalityConfig(delta_indices=[0], modality_keys=['joint_position']),
    'action': ModalityConfig(delta_indices=list(range(16)), modality_keys=['joint_position'],
        action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                         type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)]),
    'language': ModalityConfig(delta_indices=[0], modality_keys=['annotation.human.action.task_description']),
}
register_modality_config(fr3_v4_joint, EmbodimentTag.NEW_EMBODIMENT)
