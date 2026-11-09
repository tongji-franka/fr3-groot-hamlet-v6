"""Inference-only HAMLET optimizations; checkpoint parameters on disk stay unchanged."""
import warnings
from collections.abc import Mapping

import torch
from transformers.feature_extraction_utils import BatchFeature


def configure_action_head(head, *, bf16_linear, cuda_graph):
    if bf16_linear:
        for module in head.modules():
            if isinstance(module, torch.nn.Linear):
                for parameter in module.parameters(recurse=False):
                    if parameter.dtype == torch.float32:
                        parameter.data = parameter.data.to(torch.bfloat16)
    if cuda_graph:
        sampler = ActionSamplerGraph(head)
        head.get_action_with_features = sampler
        return sampler
    return None


class ChatTemplateCache:
    """Cache the fixed HAMLET text/image layout, retaining no image objects."""
    def __init__(self, processor):
        self.processor = processor
        self.original = processor.apply_chat_template
        self.key = self.text = None

    def __call__(self, conversation, *args, **kwargs):
        if (args or kwargs != dict(tokenize=False, add_generation_prompt=False)
                or len(conversation) != 1 or conversation[0].get('role') != 'user'):
            return self.original(conversation, *args, **kwargs)
        content = conversation[0]['content']
        if (not content or content[0].get('type') != 'text'
                or any(item.get('type') != 'image' for item in content[1:])):
            return self.original(conversation, **kwargs)
        key = (content[0]['text'], tuple((item['image'].size, item['image'].mode) for item in content[1:]),
               self.processor.chat_template)
        if key != self.key:
            self.text = self.original(conversation, **kwargs)
            self.key = key
        return self.text


class ActionSamplerGraph:
    """One graph per model owner; new shapes fall back without recapturing live."""
    def __init__(self, head):
        self.head = head
        self.original = head.get_action_with_features
        self.graph = self.buffers = self.output = self.signature = None
        self.disabled = False
        self.captures = self.replays = self.fallbacks = 0
        self.last_fallback = None

    def stats(self):
        return dict(captures=self.captures, replays=self.replays, fallbacks=self.fallbacks,
                    disabled=self.disabled, last_fallback=self.last_fallback)

    def _key(self, arguments):
        def key(value):
            if isinstance(value, torch.Tensor):
                return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)
            if isinstance(value, Mapping):
                return tuple((name, key(item)) for name, item in sorted(value.items()))
            return value
        return (key(arguments), self.head.use_encoder_kv_cache, self.head.num_inference_timesteps,
                torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'))

    def _capture(self, arguments, noise, signature):
        self.buffers = {name: (BatchFeature(data={k: v.clone() if isinstance(v, torch.Tensor) else v
                                               for k, v in value.items()})
                              if isinstance(value, Mapping) else value.clone())
                        for name, value in arguments.items()}
        self.buffers['initial_noise'] = noise.clone()
        stream = torch.cuda.Stream(device=noise.device)
        stream.wait_stream(torch.cuda.current_stream(noise.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.original(**self.buffers)
        torch.cuda.current_stream(noise.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.output = self.original(**self.buffers)
        self.graph, self.signature = graph, signature
        self.captures += 1

    def __call__(self, backbone_features, state_features, embodiment_id, backbone_output):
        arguments = dict(backbone_features=backbone_features, state_features=state_features,
                         embodiment_id=embodiment_id, backbone_output=backbone_output)
        if (self.disabled or self.head.training or torch.is_grad_enabled()
                or backbone_features.device.type != 'cuda'):
            return self.original(**arguments)
        signature = self._key(arguments)
        if self.graph is not None and signature != self.signature:
            self.fallbacks += 1
            self.last_fallback = 'input shape, layout, dtype or sampling options changed'
            return self.original(**arguments)
        # Draw once outside capture: warmup/replay cannot consume or freeze the noise stream.
        noise = self.head.sample_action_noise(backbone_features)
        if self.graph is None:
            try:
                self._capture(arguments, noise, signature)
            except Exception as error:
                self.graph = self.buffers = self.output = None
                self.disabled = True
                self.fallbacks += 1
                self.last_fallback = repr(error)
                warnings.warn(f'HAMLET CUDA Graph unavailable; using eager inference: {error}', RuntimeWarning)
                return self.original(**arguments, initial_noise=noise)
        for name, value in arguments.items():
            if isinstance(value, Mapping):
                for key, tensor in value.items():
                    if isinstance(tensor, torch.Tensor):
                        self.buffers[name][key].copy_(tensor)
            else:
                self.buffers[name].copy_(value)
        self.buffers['initial_noise'].copy_(noise)
        self.graph.replay()
        self.replays += 1
        # Callers may retain predictions across the next replay.
        return BatchFeature(data=dict(action_pred=self.output.action_pred.clone(),
                                      backbone_features=backbone_features, state_features=state_features))
