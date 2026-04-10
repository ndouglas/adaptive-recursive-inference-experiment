import copy
from contextlib import contextmanager
from torch import nn


class LayerDuplicator:
    """Duplicates a block of layers (i..j-1) in a model's forward pass.

    Uses shallow copies so weights are shared — zero extra VRAM for parameters.
    Each copy gets a unique layer_idx for correct KV cache indexing.

    With iterations=1 (default), this is standard RYS: [0..i-1, i..j-1, i..j-1, j..N-1].
    With iterations=K, the block runs K+1 times total (1 original + K extra).
    """

    def __init__(self, i, j, iterations=1):
        self.i = i
        self.j = j
        self.iterations = iterations

    def _make_layer_sequence(self, model):
        original_layers = model.model.layers
        num_original = len(original_layers)
        block_size = self.j - self.i
        sequence = []
        cache_idx = 0

        # Layers 0 through j-1 (first pass through block)
        for layer_num in range(self.j):
            layer = copy.copy(original_layers[layer_num])
            layer.self_attn = copy.copy(layer.self_attn)
            layer.self_attn.layer_idx = cache_idx
            sequence.append(layer)
            cache_idx += 1

        # Extra iterations of the block (layers i through j-1)
        for _ in range(self.iterations):
            for layer_num in range(self.i, self.j):
                layer = copy.copy(original_layers[layer_num])
                layer.self_attn = copy.copy(layer.self_attn)
                layer.self_attn.layer_idx = cache_idx
                sequence.append(layer)
                cache_idx += 1

        # Remaining layers j through N-1
        for layer_num in range(self.j, num_original):
            layer = copy.copy(original_layers[layer_num])
            layer.self_attn = copy.copy(layer.self_attn)
            layer.self_attn.layer_idx = cache_idx
            sequence.append(layer)
            cache_idx += 1

        return sequence, cache_idx

    def _make_layer_types(self, model, num_layers):
        """Expand layer_types to match the new layer count."""
        original_types = model.config.layer_types
        types = []
        for layer_num in range(self.j):
            types.append(original_types[layer_num])
        for _ in range(self.iterations):
            for layer_num in range(self.i, self.j):
                types.append(original_types[layer_num])
        for layer_num in range(self.j, len(original_types)):
            types.append(original_types[layer_num])
        return types

    @contextmanager
    def apply(self, model):
        """Context manager that temporarily swaps in the duplicated layer sequence."""
        original_layers = model.model.layers
        original_num_layers = model.config.num_hidden_layers
        original_layer_types = model.config.layer_types

        sequence, num_layers = self._make_layer_sequence(model)
        model.model.layers = nn.ModuleList(sequence)
        model.config.num_hidden_layers = num_layers
        model.config.layer_types = self._make_layer_types(model, num_layers)

        try:
            yield model
        finally:
            model.model.layers = original_layers
            model.config.num_hidden_layers = original_num_layers
            model.config.layer_types = original_layer_types
