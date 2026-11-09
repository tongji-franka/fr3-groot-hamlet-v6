"""Official model backbones with FR3 actions and explicit padding-aware losses."""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from v4_data import ROOT, CAMERAS, masked_mean, pad_last

WEIGHTS = ROOT / 'weights'


def load_checked(module, state, *, allowed_missing=(), allowed_unexpected=()):
    result = module.load_state_dict(state, strict=False)
    bad_missing = [k for k in result.missing_keys if not any(k.startswith(p) for p in allowed_missing)]
    bad_unexpected = [k for k in result.unexpected_keys if not any(k.startswith(p) for p in allowed_unexpected)]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f'Checkpoint mismatch: missing={bad_missing[:25]}, unexpected={bad_unexpected[:25]}')
    return {'loaded_keys': len(state), 'missing': result.missing_keys, 'unexpected': result.unexpected_keys}


def imagenet(images):
    return (images - images.new_tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)) / images.new_tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)


def native_features():
    from lerobot.configs import FeatureType, PolicyFeature
    return ({'observation.state': PolicyFeature(FeatureType.STATE, (7,)),
             **{k: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for k in CAMERAS}},
            {'action': PolicyFeature(FeatureType.ACTION, (7,))})


class XVLAAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        from lerobot.policies.xvla.configuration_xvla import XVLAConfig
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy, _remap_vendored_florence_state_dict
        from safetensors.torch import load_file
        raw = json.loads((WEIGHTS / 'xvla/config.json').read_text())
        inputs, outputs = native_features()
        cfg = XVLAConfig(input_features=inputs, output_features=outputs, device='cpu',
                         florence_config=raw['florence_config'], chunk_size=30, n_action_steps=16,
                         action_mode='auto', max_state_dim=20, max_action_dim=20, dtype=torch.bfloat16,
                         num_image_views=3, freeze_vision_encoder=True, freeze_language_encoder=True)
        self.policy = XVLAPolicy(cfg)
        state = load_file(str(WEIGHTS / 'xvla/model.safetensors'))
        state = {'model.' + k if not k.startswith('model.') else k: v for k, v in state.items()}
        state = _remap_vendored_florence_state_dict(state)
        shared = 'model.vlm.language_model.shared.weight'
        embed = 'model.vlm.language_model.encoder.embed_tokens.weight'
        if shared in state:
            state.setdefault(embed, state[shared])
        elif embed in state:
            state[shared] = state[embed]
        self.load_report = load_checked(self.policy, state)
        self.policy.model.vlm.requires_grad_(False)
        self.policy.model.transformer.float()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(WEIGHTS / 'xvla/tokenizer.json'))
        for field in ('pad_token', 'bos_token', 'eos_token'):
            token_id = raw['florence_config'][field + '_id']
            setattr(self.tokenizer, field, self.tokenizer.convert_ids_to_tokens(token_id))
            assert getattr(self.tokenizer, field + '_id') == token_id

    def forward(self, batch):
        m = self.policy.model
        images = imagenet(batch['images'][:, -1]).to(torch.bfloat16)
        ids = self.tokenizer(batch['task'], padding=True, truncation=True, max_length=64, return_tensors='pt')['input_ids'].to(images.device)
        with torch.no_grad():
            m.vlm.eval()
            enc = m.forward_vlm(ids, images, torch.ones(images.shape[:2], device=images.device, dtype=torch.bool))
        action = pad_last(batch['action'], 20)
        state = pad_last(batch['state'], 20)
        t = torch.rand(action.shape[0], device=action.device) * (1 - 1e-5)
        noisy = torch.randn_like(action) * t[:, None, None] + action * (1 - t[:, None, None])
        pred = m.transformer(domain_id=torch.full((action.shape[0],), 29, device=action.device, dtype=torch.long),
                             action_with_noise=noisy, t=t, proprio=state, **enc)
        return masked_mean((pred[..., :7].float() - batch['action']) ** 2, ~batch['action_is_pad'][..., None])

    def predict(self, batch):
        images = imagenet(batch['images'][:, -1]).to(torch.bfloat16)
        ids = self.tokenizer(batch['task'], padding=True, truncation=True, max_length=64, return_tensors='pt')['input_ids'].to(images.device)
        return self.policy.model.generate_actions(
            input_ids=ids, image_input=images,
            image_mask=torch.ones(images.shape[:2], device=images.device, dtype=torch.bool),
            domain_id=torch.full((images.shape[0],), 29, device=images.device, dtype=torch.long),
            proprio=pad_last(batch['state'], 20), steps=10)[..., :7]


class FlowerAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(ROOT / 'models/flower/upstream'))
        from flower.models.flower import FLOWERVLA
        self.model = FLOWERVLA(vlm_path=str(WEIGHTS / 'florence'), freeze_florence=True,
                               multistep=20, num_sampling_steps=4, lowdim_obs_dim=16, action_dim=8,
                               act_window_size=20, use_second_view=True, use_proprio=True,
                               dit_dim=1024, n_layers=12, n_heads=16, use_rope=True,
                               rope_theta=1000., query_seq_len=100, sampling_type='uniform',
                               vlm_prompt_style='default')
        checkpoint = torch.load(WEIGHTS / 'flower/360000_model_weights.pt', map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        state = {k.removeprefix('agent.').replace('vlm.language_encoder.', 'vlm.language_model.model.encoder.')
                 .replace('.mlp.c_fc1.', '.mlp.fc1.').replace('.mlp.c_fc2.', '.mlp.fc2.')
                 .replace('.mlp.c_proj.', '.mlp.proj.'): v for k, v in state.items()}
        self.load_report = load_checked(self.model, state)
        # The released joint_single proprio encoder is a zero encoder: introduce the actual 7-joint input.
        from timm.layers.mlp import Mlp
        self.model.proprio_encoders['joint_single'] = Mlp(8, 1024, out_features=1024)
        self.load_report['new_fr3_parameters'] = 'proprio_encoders.joint_single (pretrained branch was ZeroEncoder)'
        self.model.vlm.requires_grad_(False)
        self.model.vlm.to(torch.bfloat16)
        self.model.prompt_embeds.requires_grad_(False)
        self.model.prompt_embeds.data = self.model.prompt_embeds.data.to(torch.bfloat16)

    def to(self, *args, **kwargs):
        # Lightning caches its device; moving only the containing nn.Module leaves that cache stale.
        self.model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def conditions(self, batch):
        m = self.model
        images = imagenet(batch['images'][:, -1])
        b, v, c, h, w = images.shape
        with torch.no_grad():
            m.vlm.eval()
            features = m.vlm._encode_image(images.flatten(0, 1).to(torch.bfloat16)).reshape(b, -1, 1024)
            prompts = [f'Agent Type: 1-arm FR3, Action Space: joint position. Task Instruction: {t}' for t in batch['task']]
            tokens = m.tokenizer(prompts, padding=True, truncation=True, max_length=64, return_tensors='pt').to(images.device)
            text = m.vlm.get_input_embeddings()(tokens.input_ids)
            merged = torch.cat([features, m.prompt_embeds.expand(b, -1, -1).to(features), text], dim=1)
            mask = torch.cat([torch.ones(b, features.shape[1] + 1, device=images.device, dtype=torch.long), tokens.attention_mask], dim=1)
            cond = m.vlm.get_encoder()(inputs_embeds=merged, attention_mask=mask).last_hidden_state
        return {'features': cond, 'frequency_embeds': m.frequency_embedder(torch.full((b,), 30., device=images.device)),
                'action_type': torch.zeros(b, device=images.device, dtype=torch.long),
                'proprio': pad_last(batch['state'], 8).unsqueeze(1)}

    def forward(self, batch):
        actions = pad_last(batch['action'], 8)
        noise = torch.randn_like(actions)
        t = torch.rand(actions.shape[0], device=actions.device)
        noisy = (1 - t[:, None, None]) * actions + t[:, None, None] * noise
        pred = self.model.dit_forward(noisy, t, self.conditions(batch))
        return masked_mean((pred[..., :7].float() - (noise - actions)[..., :7]) ** 2, ~batch['action_is_pad'][..., None])

    def predict(self, batch):
        noise = torch.randn(batch['state'].shape[0], 20, 8, device=batch['state'].device)
        return self.model.sample_actions(noise, self.conditions(batch), inference=True)[..., :7]


class RDTAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(ROOT / 'models/rdt/upstream'))
        from models.rdt_runner import RDTRunner
        from transformers import SiglipVisionModel, SiglipImageProcessor
        cfg = json.loads((WEIGHTS / 'rdt/config.json').read_text())
        self.model = RDTRunner(action_dim=128, pred_horizon=64, config=cfg, lang_token_dim=4096,
                               img_token_dim=1152, state_token_dim=128, max_lang_cond_len=1024,
                               img_cond_len=4374, img_pos_embed_config=cfg['img_pos_embed_config'],
                               lang_pos_embed_config=cfg['lang_pos_embed_config'], dtype=torch.float32)
        state = torch.load(WEIGHTS / 'rdt/pytorch_model.bin', map_location='cpu', weights_only=True)
        self.load_report = load_checked(self.model, state)
        self.vision, vision_load = SiglipVisionModel.from_pretrained(
            WEIGHTS / 'siglip', torch_dtype=torch.bfloat16, local_files_only=True, output_loading_info=True)
        assert not vision_load['missing_keys'] and not vision_load.get('mismatched_keys')
        self.load_report['vision_missing_keys'] = vision_load['missing_keys']
        self.vision.requires_grad_(False)
        self.image_processor = SiglipImageProcessor.from_pretrained(WEIGHTS / 'siglip', local_files_only=True)
        cached = torch.load(ROOT / 'reports/rdt_language.pt', map_location='cpu', weights_only=True)
        self.task = cached['task']
        self.register_buffer('language', cached['embedding'])

    def conditions(self, batch):
        assert all(task == self.task for task in batch['task']), 'Rebuild the RDT language cache for changed task text'
        b, hist, views, c, h, w = batch['images'].shape
        flat = batch['images'].flatten(0, 2)
        # SigLIP uses [-1,1], 384px, 27x27 patch tokens per image.
        imgs = F.interpolate(flat, (384, 384), mode='bilinear', align_corners=False, antialias=True)
        with torch.no_grad():
            self.vision.eval()
            img = self.vision(pixel_values=(imgs * 2 - 1).to(torch.bfloat16)).last_hidden_state.reshape(b, 4374, 1152)
        language = self.language.expand(b, -1, -1)
        language_mask = torch.ones(language.shape[:2], dtype=torch.bool, device=language.device)
        state = pad_last(batch['state'], 128).unsqueeze(1)
        dim_mask = torch.zeros(b, 1, 128, device=state.device)
        dim_mask[..., :7] = 1
        frequency = torch.full((b,), 30., device=state.device)
        return language, language_mask, img, state, dim_mask, frequency

    def forward(self, batch):
        m = self.model
        language, language_mask, img, state, dim_mask, frequency = self.conditions(batch)
        action = pad_last(batch['action'], 128)
        b = action.shape[0]
        noise = torch.randn_like(action)
        steps = torch.randint(0, m.num_train_timesteps, (b,), device=action.device)
        noisy = m.noise_scheduler.add_noise(action, noise, steps)
        sequence = torch.cat([state, noisy], dim=1)
        sequence = torch.cat([sequence, dim_mask.expand(-1, sequence.shape[1], -1)], dim=-1)
        lang_cond, img_cond, sequence = m.adapt_conditions(language, img, sequence)
        pred = m.model(sequence, frequency, steps, lang_cond, img_cond, lang_mask=language_mask)
        return masked_mean((pred[..., :7].float() - action[..., :7]) ** 2, ~batch['action_is_pad'][..., None])

    def predict(self, batch):
        return self.model.predict_action(*self.conditions(batch))[..., :7]


def _rtc_prefix_mask(batch_size, horizon, max_delay, valid_steps, device):
    """Per-example RTC clean-prefix mask for training-time Real-Time Chunking.

    Mirrors lerobot.policies.pi05.modeling_pi05._sample_training_rtc_prefix_mask
    (delay ~ U[0, max_delay]; True = clean prefix step that is not noised and is
    excluded from the flow loss), with one addition: the delay is clamped to
    valid_steps - 1 so the prefix never covers episode-end padding and at least
    one valid postfix step remains to train on.
    """
    delays = torch.randint(0, max_delay + 1, (batch_size,), device=device)
    delays = torch.minimum(delays, (valid_steps - 1).clamp(min=0))
    positions = torch.arange(horizon, device=device)
    return positions.unsqueeze(0) < delays.unsqueeze(1)


class PI05Adapter(nn.Module):
    def __init__(self, lora=False, lora_r=32, lora_alpha=64, rtc_max_delay=10):
        super().__init__()
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from safetensors.torch import load_file
        inputs, outputs = native_features()
        cfg = PI05Config(input_features=inputs, output_features=outputs, device='cpu', dtype=torch.bfloat16,
                         freeze_vision_encoder=True, train_expert_only=True, gradient_checkpointing=True,
                         text_tokenizer_name=str(WEIGHTS / 'paligemma_tokenizer'), n_action_steps=16)
        assert 0 <= int(rtc_max_delay) < cfg.chunk_size, 'rtc_max_delay must be in [0, chunk_size)'
        self.rtc_max_delay = int(rtc_max_delay)
        # Recorded on the policy config so this checkpoint can run the fork's trained-mode RTC.
        cfg.rtc_training_max_delay = self.rtc_max_delay
        self.policy = PI05Policy(cfg)
        state = load_file(str(WEIGHTS / 'pi05/model.safetensors'))
        state = self.policy._convert_openpi_state_dict(state)
        self.load_report = load_checked(self.policy, state)
        # The expert lm_head (semantic-output head) takes no part in the flow-matching loss and
        # never received a gradient in the verified single-GPU baseline either; freeze it so the
        # trainable set is honest and DDP sees a fully-participating graph (-263.3M params).
        self.policy.model.paligemma_with_expert.gemma_expert.lm_head.weight.requires_grad_(False)
        for p in self.policy.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.lora_config = None
        if lora:
            from peft import LoraConfig, get_peft_model
            # Full-target set computed from the real expert shapes: attention q/k/v/o + all
            # expert MLPs + the action/time projections (~14.1M @ r=32). Anchored on
            # gemma_expert so the frozen PaliGemma VLM is never touched.
            target = (r'.*\.gemma_expert\..*(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)'
                      r'|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)')
            for p in self.policy.parameters():
                p.requires_grad_(False)
            self.policy = get_peft_model(self.policy, LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=0., bias='none', target_modules=target))
            for p in self.policy.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
            self.lora_config = {'r': lora_r, 'alpha': lora_alpha}
        self.tokenizer = AutoTokenizer.from_pretrained(WEIGHTS / 'paligemma_tokenizer', local_files_only=True)

    @property
    def _policy(self):
        """The PI05Policy, unwrapping PEFT when active (same module tree either way)."""
        return self.policy.get_base_model() if self.lora_config else self.policy

    @property
    def core(self):
        """PI05Pytorch, i.e. the object whose forward takes prefix_mask."""
        return self._policy.model

    def prepare_batch(self, batch):
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        states = batch['state'].detach().cpu().numpy()
        bins = np.digitize(states, np.linspace(-1, 1, 257)[:-1]) - 1
        prompts = [f'Task: {t.strip()}, State: {" ".join(map(str, row))};\nAction: ' for t, row in zip(batch['task'], bins)]
        tokens = self.tokenizer(prompts, max_length=200, padding='max_length', truncation=True, return_tensors='pt').to(batch['state'].device)
        data = {'action': batch['action'], 'observation.state': batch['state'],
                OBS_LANGUAGE_TOKENS: tokens.input_ids, OBS_LANGUAGE_ATTENTION_MASK: tokens.attention_mask.bool()}
        data.update({k: batch['images'][:, -1, i] for i, k in enumerate(CAMERAS)})
        return data

    def forward(self, batch):
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        data = self.prepare_batch(batch)
        images, masks = self._policy._preprocess_images(data)
        actions = self._policy.prepare_action(data)
        noise = self.core.sample_noise(actions.shape, actions.device)
        time = self.core.sample_time(actions.shape[0], actions.device)
        prefix = None
        if self.rtc_max_delay > 0:
            valid = ~batch['action_is_pad']
            prefix = _rtc_prefix_mask(valid.shape[0], actions.shape[1], self.rtc_max_delay,
                                      valid.sum(1), actions.device)
        losses = self.core(images, masks, data[OBS_LANGUAGE_TOKENS], data[OBS_LANGUAGE_ATTENTION_MASK],
                           actions, noise, time, prefix_mask=prefix)
        keep = ~batch['action_is_pad'][..., None]
        if prefix is not None:
            keep = keep & ~prefix[..., None]
        return masked_mean(losses[..., :7].float(), keep)

    def predict(self, batch):
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        data = self.prepare_batch(batch)
        images, masks = self._policy._preprocess_images(data)
        return self.core.sample_actions(images, masks, data[OBS_LANGUAGE_TOKENS], data[OBS_LANGUAGE_ATTENTION_MASK])[..., :7]


class HamletAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        sys.path.insert(0, str(ROOT / 'models/hamlet/upstream'))
        from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
        from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
        from safetensors.torch import load_file
        cfg = Gr00tN1d6Config.from_pretrained(WEIGHTS / 'hamlet')
        cfg.hamlet_mode = 'finetune'
        cfg.memory_window, cfg.memory_stride = 4, 16
        cfg.n_moment_tokens, cfg.memory_num_layers = 4, 2
        cfg.freeze_moment_tokens = False
        cfg.tune_top_llm_layers = 0
        cfg.tune_llm = cfg.tune_visual = cfg.use_relative_action = cfg.apply_sincos_state_encoding = False
        cfg.use_flash_attention = False
        self.model = Gr00tN1d6(cfg, transformers_loading_kwargs={'trust_remote_code': True, 'local_files_only': True})
        state = {}
        for p in sorted((WEIGHTS / 'hamlet').glob('model-*.safetensors')):
            state.update(load_file(str(p)))
        self.load_report = load_checked(self.model, state, allowed_missing=('backbone.moment_tokens', 'action_head.memory_transformer.'))
        self.model.backbone.model.to(torch.bfloat16)
        for p in self.model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        # Checkpoint each frozen LM layer because trainable moment tokens backpropagate through it.
        self.model.backbone.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.cfg = cfg

    def prepare_inputs(self, batch):
        from PIL import Image
        m, b = self.model, batch['state'].shape[0]
        features = []
        for i in range(b):
            history = []
            for t in range(4):
                imgs = [Image.fromarray((im.permute(1, 2, 0).detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)) for im in batch['images'][i, t]]
                conversation = [{'role': 'user', 'content': [{'type': 'text', 'text': batch['task'][i]}] + [{'type': 'image', 'image': im} for im in imgs]}]
                text = m.collator.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
                history.append({'text': text, 'images': imgs, 'conversation': conversation})
            state = pad_last(batch['state'][i], 128).unsqueeze(0)
            action = pad_last(batch['action'][i], 128)
            mask = torch.zeros_like(action)
            mask[..., :7] = (~batch['action_is_pad'][i, :, None]).float()
            features.append({'vlm_content_list': history, 'state': state.detach().cpu().numpy(),
                             'action': action.detach().cpu().numpy(), 'action_mask': mask.detach().cpu().numpy(),
                             'embodiment_id': np.array(10, dtype=np.int64)})
        inputs = m.collator(features)['inputs']
        return inputs

    def forward(self, batch):
        return self.model(self.prepare_inputs(batch))['loss']

    def predict(self, batch):
        self.model.action_head.reset_memory()
        inputs = self.prepare_inputs(batch)
        inputs.pop('action')
        return self.model.get_action(inputs)['action_pred'][..., :7]


ADAPTERS = {'xvla': XVLAAdapter, 'flower': FlowerAdapter, 'hamlet': HamletAdapter, 'rdt': RDTAdapter, 'pi05': PI05Adapter}
