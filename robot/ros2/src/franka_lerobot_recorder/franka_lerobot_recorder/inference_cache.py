"""Bounded deployment caches for fixed text and immutable camera frames."""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps


def _key(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((name, _key(item)) for name, item in value.items()))
    raise TypeError('Dynamic condition is not cacheable')


class CachedTokenizer:
    def __init__(self, tokenizer, capacity=16):
        self.tokenizer, self.capacity = tokenizer, capacity
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, *args, **kwargs):
        try:
            key = (_key(args), _key(kwargs))
        except TypeError:
            return self.tokenizer(*args, **kwargs)
        if key not in self.cache:
            self.misses += 1
            self.cache[key] = deepcopy(self.tokenizer(*args, **kwargs))
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        else:
            self.hits += 1
        self.cache.move_to_end(key)
        # BatchEncoding.to() mutates its tensors; callers must own their result.
        return deepcopy(self.cache[key])


class FramePreprocessor:
    def __init__(self, transform, capacity=48):
        self.transform, self.capacity = transform, capacity
        self.cache = OrderedDict()
        self.hits = self.misses = 0

    def clear(self):
        self.cache.clear()

    def __call__(self, frame):
        if frame.flags.writeable or not frame.flags.owndata:
            self.misses += 1
            return self.transform(frame)
        key = id(frame)
        if key not in self.cache:
            self.misses += 1
            self.cache[key] = (frame, self.transform(frame))
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        else:
            self.hits += 1
        self.cache.move_to_end(key)
        # Retain the array itself, preventing identity reuse while its value is cached.
        return self.cache[key][1]


class VisualCache:
    """Reuse a frozen image tower only for the exact same owned, immutable frames.

    Language-conditioned outputs and VLM K/V are deliberately recomputed: they
    depend on the current state/task even when the camera image is unchanged.
    """
    def __init__(self, owner, capacity=2):
        if owner.training or any(p.requires_grad for p in owner.parameters()):
            raise ValueError('Visual caching requires frozen eval weights')
        self.owner, self.capacity = owner, capacity
        self.frames = None
        self.cache = OrderedDict()
        self.hits = self.misses = self.bypassed = 0

    @contextmanager
    def observation(self, frames):
        previous = self.frames
        self.frames = tuple(frames) if all(not frame.flags.writeable and frame.flags.owndata for frame in frames) else None
        try:
            yield
        finally:
            self.frames = previous

    def clear(self):
        self.cache.clear()

    def wrap(self, module, method='forward'):
        original = getattr(module, method)
        @wraps(original)
        def forward(*args, **kwargs):
            import torch
            pixels = kwargs.get('pixel_values', args[0] if args else None)
            if (self.frames is None or self.owner.training or torch.is_grad_enabled()
                    or not isinstance(pixels, torch.Tensor) or pixels.ndim != 4
                    or pixels.shape[0] != len(self.frames)):
                self.bypassed += 1
                return original(*args, **kwargs)
            try:
                options = _key({key: value for key, value in kwargs.items() if key != 'pixel_values'})
                extra = _key(args[1:])
            except TypeError:
                self.bypassed += 1
                return original(*args, **kwargs)
            key = (tuple(map(id, self.frames)), tuple(pixels.shape), pixels.dtype, pixels.device,
                   torch.is_autocast_enabled(pixels.device.type), torch.get_autocast_dtype(pixels.device.type), options, extra)
            if key not in self.cache:
                self.misses += 1
                self.cache[key] = (self.frames, original(*args, **kwargs))
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
            else:
                self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key][1]
        setattr(module, method, forward)


class HistoryFeatureCache(VisualCache):
    """Cache only the entire processed history; Eagle packed attention couples images.

    Per-image reuse changes this checkpoint's outputs. Hashing processed pixels
    also covers preprocessing changes and mutable inputs. No language/state output
    is cached. The opt-in path must be benchmarked for the actual observation rate.
    """
    def wrap(self, module, method='extract_feature'):
        import hashlib
        original = getattr(module, method)
        @wraps(original)
        def forward(pixel_values, *args, **kwargs):
            import torch
            listed = isinstance(pixel_values, (list, tuple))
            images = pixel_values if listed else [pixel_values]
            if (self.frames is None or self.owner.training or torch.is_grad_enabled()
                    or args or any(value is not None for value in kwargs.values())
                    or not images or not all(isinstance(x, torch.Tensor) for x in images)):
                self.bypassed += 1
                return original(pixel_values, *args, **kwargs)
            identities = []
            for pixels in images:
                cpu = pixels.detach().contiguous().cpu()
                identities.append((tuple(cpu.shape), str(cpu.dtype), str(pixels.device),
                    torch.is_autocast_enabled(pixels.device.type),
                    torch.get_autocast_dtype(pixels.device.type),
                    hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).digest()))
            key = (listed, tuple(identities))
            if key not in self.cache:
                self.misses += 1
                output = original(pixel_values, *args, **kwargs)
                if not isinstance(output, torch.Tensor):
                    self.bypassed += 1
                    return output
                self.cache[key] = output.detach().clone()
                if len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
            else:
                self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key].clone()
        setattr(module, method, forward)


def install_frozen_pi05_cache(agent):
    """Install outside the training package, retaining its checkpoint source hashes."""
    model = agent.model
    model.tokenizer = CachedTokenizer(model.tokenizer)
    cache = VisualCache(model)
    cache.wrap(model.policy.model.paligemma_with_expert, 'embed_image')
    predict, reset = agent.predict_chunk, agent.reset
    @wraps(predict)
    def cached(state, *images, **kwargs):
        # Native adapters pass wrist/overhead/third; vision batches use overhead first.
        with cache.observation([images[i] for i in (1, 0, 2)]):
            return predict(state, *images, **kwargs)
    def clear():
        cache.clear()
    def reset_cached():
        clear()
        reset()
    agent.predict_chunk, agent.reset = cached, reset_cached
    agent.clear_visual_cache = clear
    agent.inference_cache = cache
    agent.metadata['inference_cache'] = 'immutable-vision-and-exact-text-v1'
