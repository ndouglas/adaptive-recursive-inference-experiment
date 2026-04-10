"""Shared test fixtures.

The mock model simulates a 4-layer transformer with a 2-layer circuit block.
It produces deterministic outputs suitable for testing convergence instrumentation
without loading a real model (which requires GPU and ~3GB).
"""
import pytest
import torch
import torch.nn as nn


class MockLayer(nn.Module):
    """A transformer layer that applies a small linear perturbation.

    Each forward call slightly transforms the hidden state, producing
    a cosine similarity trajectory that converges over repeated passes.
    """
    def __init__(self, hidden_size, perturbation_scale=0.01):
        super().__init__()
        self.weight = nn.Parameter(
            torch.eye(hidden_size) + perturbation_scale * torch.randn(hidden_size, hidden_size)
        )

    def forward(self, hidden_states, **kwargs):
        return hidden_states @ self.weight


class MockModel(nn.Module):
    """Minimal model matching the interface AdaptiveLoop expects.

    Structure: embed_tokens -> 4 layers -> norm -> lm_head
    Circuit block is layers[1:3] (block_i=1, block_j=3).
    """
    def __init__(self, vocab_size=100, hidden_size=32, num_layers=4):
        super().__init__()
        self.config = type("Config", (), {"num_hidden_layers": num_layers})()

        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList(
            [MockLayer(hidden_size) for _ in range(num_layers)]
        )
        self.model.norm = nn.LayerNorm(hidden_size)
        self.model.rotary_emb = MockRotaryEmb(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def device(self):
        return next(self.parameters()).device


class MockRotaryEmb(nn.Module):
    """Returns dummy position embeddings matching the expected interface."""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden_states, position_ids):
        seq_len = hidden_states.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype
        cos = torch.ones(1, seq_len, self.hidden_size, device=device, dtype=dtype)
        sin = torch.zeros(1, seq_len, self.hidden_size, device=device, dtype=dtype)
        return cos, sin


@pytest.fixture
def mock_model():
    """A tiny deterministic model for testing convergence instrumentation."""
    torch.manual_seed(42)
    return MockModel(vocab_size=100, hidden_size=32, num_layers=4)


@pytest.fixture
def block_config():
    """Circuit block configuration for the mock model."""
    return {"block_i": 1, "block_j": 3}
