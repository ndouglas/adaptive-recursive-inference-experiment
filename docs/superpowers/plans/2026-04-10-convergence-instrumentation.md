# Stage 1: Convergence Trajectory Instrumentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture rich, structured, per-token convergence data from the adaptive inference loop, suitable for statistical analysis in later stages.

**Architecture:** A `ConvergenceTrace` dataclass captures per-token iteration counts, similarity trajectories, L2 norm trajectories, and timing data. A `ConvergenceTracer` class wraps `AdaptiveLoop` to collect traces during generation. The tracer delegates all inference to the existing loop — it only adds measurement. Traces serialize to JSON for offline analysis.

**Tech Stack:** Python dataclasses, pytest, torch, existing `AdaptiveLoop` from `src/inference/adaptive_loop.py`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/inference/convergence_trace.py` | `ConvergenceTrace` and `TokenTrace` dataclasses |
| Create | `src/inference/convergence_tracer.py` | `ConvergenceTracer` wrapping `AdaptiveLoop` |
| Modify | `src/inference/adaptive_loop.py:75-144` | Add optional L2 norm capture to `forward()` |
| Create | `tests/conftest.py` | Pytest fixtures (tiny mock model) |
| Create | `tests/inference/test_convergence_trace.py` | Tests for trace dataclasses |
| Create | `tests/inference/test_convergence_tracer.py` | Tests for tracer wrapping adaptive loop |
| Create | `tests/inference/__init__.py` | Package marker |
| Create | `tests/__init__.py` | Package marker |
| Modify | `pyproject.toml` | Add pytest dependency and config |

---

### Task 1: Set Up Test Infrastructure

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`
- Create: `tests/inference/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest dependency**

```bash
uv add --dev pytest
```

- [ ] **Step 2: Add pytest configuration to pyproject.toml**

Add this section to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Create test package structure**

Create `tests/__init__.py` — empty file.

Create `tests/inference/__init__.py` — empty file.

- [ ] **Step 4: Create conftest.py with a tiny mock model fixture**

Create `tests/conftest.py`:

```python
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
```

- [ ] **Step 5: Verify pytest runs with no tests collected**

Run: `uv run pytest --co -q`

Expected: `no tests ran` (or `0 items collected`) with exit code 5 (no tests). No import errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/
git commit -m "Add pytest infrastructure with mock model fixture"
```

---

### Task 2: ConvergenceTrace and TokenTrace Dataclasses

**Files:**
- Create: `src/inference/convergence_trace.py`
- Create: `tests/inference/test_convergence_trace.py`

- [ ] **Step 1: Write tests for the trace dataclasses**

Create `tests/inference/test_convergence_trace.py`:

```python
"""Tests for ConvergenceTrace and TokenTrace dataclasses."""
import json
import math

from src.inference.convergence_trace import TokenTrace, ConvergenceTrace


class TestTokenTrace:
    def test_create_with_required_fields(self):
        trace = TokenTrace(
            token_id=42,
            iterations=2,
            similarities=[0.85, 0.97],
            l2_norms=[1.5, 1.4],
            halted_early=True,
            elapsed_s=0.05,
        )
        assert trace.token_id == 42
        assert trace.iterations == 2
        assert trace.similarities == [0.85, 0.97]
        assert trace.l2_norms == [1.5, 1.4]
        assert trace.halted_early is True
        assert trace.elapsed_s == 0.05

    def test_final_similarity(self):
        trace = TokenTrace(
            token_id=1, iterations=3, similarities=[0.8, 0.9, 0.95],
            l2_norms=[1.0, 1.0, 1.0], halted_early=True, elapsed_s=0.01,
        )
        assert trace.final_similarity == 0.95

    def test_final_similarity_empty(self):
        trace = TokenTrace(
            token_id=1, iterations=0, similarities=[],
            l2_norms=[], halted_early=False, elapsed_s=0.0,
        )
        assert trace.final_similarity is None

    def test_convergence_speed(self):
        trace = TokenTrace(
            token_id=1, iterations=3, similarities=[0.80, 0.90, 0.95],
            l2_norms=[1.0, 1.0, 1.0], halted_early=True, elapsed_s=0.01,
        )
        # (0.95 - 0.80) / (3 - 1) = 0.075
        assert abs(trace.convergence_speed - 0.075) < 1e-9

    def test_convergence_speed_single_iteration(self):
        trace = TokenTrace(
            token_id=1, iterations=1, similarities=[0.99],
            l2_norms=[1.0], halted_early=True, elapsed_s=0.01,
        )
        assert trace.convergence_speed is None

    def test_to_dict(self):
        trace = TokenTrace(
            token_id=42, iterations=2, similarities=[0.85, 0.97],
            l2_norms=[1.5, 1.4], halted_early=True, elapsed_s=0.05,
        )
        d = trace.to_dict()
        assert d["token_id"] == 42
        assert d["iterations"] == 2
        assert d["final_similarity"] == 0.97
        assert d["convergence_speed"] is not None
        # Must be JSON-serializable
        json.dumps(d)


class TestConvergenceTrace:
    def _make_token_traces(self, n=3):
        return [
            TokenTrace(
                token_id=i, iterations=i + 1,
                similarities=[0.8 + 0.05 * j for j in range(i + 1)],
                l2_norms=[1.0] * (i + 1),
                halted_early=(i < 2), elapsed_s=0.01 * (i + 1),
            )
            for i in range(n)
        ]

    def test_create(self):
        tokens = self._make_token_traces()
        trace = ConvergenceTrace(
            prompt="What is 2+2?",
            generated="4",
            score=1.0,
            threshold=0.95,
            max_iterations=4,
            token_traces=tokens,
        )
        assert trace.prompt == "What is 2+2?"
        assert len(trace.token_traces) == 3

    def test_summary_statistics(self):
        tokens = self._make_token_traces()
        trace = ConvergenceTrace(
            prompt="test", generated="answer", score=0.5,
            threshold=0.95, max_iterations=4, token_traces=tokens,
        )
        summary = trace.summary()
        assert summary["total_tokens"] == 3
        assert summary["avg_iterations"] == 2.0  # (1+2+3)/3
        assert summary["pct_early_halt"] == 2 / 3
        assert "avg_final_similarity" in summary
        assert "avg_convergence_speed" in summary

    def test_to_dict_roundtrip(self):
        tokens = self._make_token_traces()
        trace = ConvergenceTrace(
            prompt="test", generated="answer", score=0.5,
            threshold=0.95, max_iterations=4, token_traces=tokens,
        )
        d = trace.to_dict()
        # Must be fully JSON-serializable
        serialized = json.dumps(d)
        restored = json.loads(serialized)
        assert restored["prompt"] == "test"
        assert len(restored["token_traces"]) == 3
        assert restored["summary"]["total_tokens"] == 3

    def test_from_dict(self):
        tokens = self._make_token_traces()
        original = ConvergenceTrace(
            prompt="test", generated="answer", score=0.5,
            threshold=0.95, max_iterations=4, token_traces=tokens,
        )
        d = original.to_dict()
        restored = ConvergenceTrace.from_dict(d)
        assert restored.prompt == original.prompt
        assert restored.score == original.score
        assert len(restored.token_traces) == len(original.token_traces)
        assert restored.token_traces[0].token_id == original.token_traces[0].token_id
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/inference/test_convergence_trace.py -v`

Expected: `ModuleNotFoundError: No module named 'src.inference.convergence_trace'`

- [ ] **Step 3: Implement the trace dataclasses**

Create `src/inference/convergence_trace.py`:

```python
"""Structured convergence trace data for statistical analysis.

TokenTrace captures per-token convergence dynamics (iterations, similarity
trajectory, L2 norms). ConvergenceTrace wraps a full generation with all
token traces plus metadata.

Both classes serialize to/from plain dicts for JSON storage.
"""
from dataclasses import dataclass, field


@dataclass
class TokenTrace:
    """Convergence data for a single generated token.

    Attributes:
        token_id: The generated token's vocabulary ID.
        iterations: Number of block passes used for this token.
        similarities: Cosine similarity at each iteration.
        l2_norms: L2 norm of hidden state after each iteration.
        halted_early: Whether convergence threshold was reached before max_iterations.
        elapsed_s: Wall-clock time for this token's generation.
    """
    token_id: int
    iterations: int
    similarities: list[float]
    l2_norms: list[float]
    halted_early: bool
    elapsed_s: float

    @property
    def final_similarity(self) -> float | None:
        return self.similarities[-1] if self.similarities else None

    @property
    def convergence_speed(self) -> float | None:
        """Similarity gain per iteration: (last - first) / (n - 1)."""
        if len(self.similarities) < 2:
            return None
        return (self.similarities[-1] - self.similarities[0]) / (len(self.similarities) - 1)

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "iterations": self.iterations,
            "similarities": self.similarities,
            "l2_norms": self.l2_norms,
            "halted_early": self.halted_early,
            "elapsed_s": self.elapsed_s,
            "final_similarity": self.final_similarity,
            "convergence_speed": self.convergence_speed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenTrace":
        return cls(
            token_id=d["token_id"],
            iterations=d["iterations"],
            similarities=d["similarities"],
            l2_norms=d["l2_norms"],
            halted_early=d["halted_early"],
            elapsed_s=d["elapsed_s"],
        )


@dataclass
class ConvergenceTrace:
    """Full convergence trace for one generation (prompt -> output).

    Attributes:
        prompt: The input prompt text.
        generated: The generated output text.
        score: Evaluation score (0-1) for the generated answer.
        threshold: Cosine similarity halting threshold used.
        max_iterations: Max iteration cap used.
        token_traces: Per-token convergence data.
    """
    prompt: str
    generated: str
    score: float
    threshold: float
    max_iterations: int
    token_traces: list[TokenTrace] = field(default_factory=list)

    def summary(self) -> dict:
        """Aggregate statistics across all tokens."""
        if not self.token_traces:
            return {}
        iters = [t.iterations for t in self.token_traces]
        finals = [t.final_similarity for t in self.token_traces if t.final_similarity is not None]
        speeds = [t.convergence_speed for t in self.token_traces if t.convergence_speed is not None]
        return {
            "total_tokens": len(self.token_traces),
            "avg_iterations": sum(iters) / len(iters),
            "min_iterations": min(iters),
            "max_iterations_used": max(iters),
            "pct_early_halt": sum(1 for t in self.token_traces if t.halted_early) / len(self.token_traces),
            "avg_final_similarity": sum(finals) / len(finals) if finals else None,
            "avg_convergence_speed": sum(speeds) / len(speeds) if speeds else None,
            "total_elapsed_s": sum(t.elapsed_s for t in self.token_traces),
        }

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "generated": self.generated,
            "score": self.score,
            "threshold": self.threshold,
            "max_iterations": self.max_iterations,
            "token_traces": [t.to_dict() for t in self.token_traces],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConvergenceTrace":
        return cls(
            prompt=d["prompt"],
            generated=d["generated"],
            score=d["score"],
            threshold=d["threshold"],
            max_iterations=d["max_iterations"],
            token_traces=[TokenTrace.from_dict(t) for t in d["token_traces"]],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_convergence_trace.py -v`

Expected: All 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/inference/convergence_trace.py tests/inference/test_convergence_trace.py
git commit -m "Add ConvergenceTrace and TokenTrace dataclasses with tests"
```

---

### Task 3: Add L2 Norm Capture to AdaptiveLoop.forward()

**Files:**
- Modify: `src/inference/adaptive_loop.py:75-144`
- Create: `tests/inference/test_adaptive_loop.py`

- [ ] **Step 1: Write test for L2 norm capture**

Create `tests/inference/test_adaptive_loop.py`:

```python
"""Tests for AdaptiveLoop modifications."""
import torch

from src.inference.adaptive_loop import AdaptiveLoop


class TestAdaptiveLoopL2Norms:
    def test_forward_returns_l2_norms(self, mock_model, block_config):
        loop = AdaptiveLoop(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=3,
        )
        input_ids = torch.tensor([[1, 2, 3]])
        logits, info = loop.forward(input_ids)

        assert "l2_norms" in info
        assert len(info["l2_norms"]) == info["iterations"]
        for norm in info["l2_norms"]:
            assert isinstance(norm, float)
            assert norm > 0

    def test_forward_l2_norms_length_matches_trajectory(self, mock_model, block_config):
        loop = AdaptiveLoop(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=4,
        )
        input_ids = torch.tensor([[1, 2, 3, 4]])
        _, info = loop.forward(input_ids)

        assert len(info["l2_norms"]) == len(info["trajectory"])

    def test_forward_preserves_existing_behavior(self, mock_model, block_config):
        loop = AdaptiveLoop(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=3,
        )
        input_ids = torch.tensor([[1, 2, 3]])
        logits, info = loop.forward(input_ids)

        assert logits.shape == (1, 3, 100)  # (batch, seq, vocab)
        assert "iterations" in info
        assert "final_similarity" in info
        assert "trajectory" in info
        assert "halted_early" in info
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/inference/test_adaptive_loop.py -v`

Expected: `test_forward_returns_l2_norms` fails with `KeyError: 'l2_norms'`

- [ ] **Step 3: Add L2 norm capture to AdaptiveLoop.forward()**

In `src/inference/adaptive_loop.py`, modify the `forward()` method. Replace the adaptive block loop section (approximately lines 105-125):

Old code:
```python
        # Phase 2: Adaptive block loop
        block_layers = list(self.layers[self.block_i:self.block_j])
        prev_output = hidden_states
        trajectory = []
        iterations = 0

        for _ in range(self.max_iterations):
            current_output = self._forward_layers(
                prev_output, block_layers, causal_mask, position_embeddings, position_ids
            )
            iterations += 1

            sim = self._cosine_sim(prev_output, current_output)
            trajectory.append(sim)

            prev_output = current_output

            if sim > self.threshold:
                break
```

New code:
```python
        # Phase 2: Adaptive block loop
        block_layers = list(self.layers[self.block_i:self.block_j])
        prev_output = hidden_states
        trajectory = []
        l2_norms = []
        iterations = 0

        for _ in range(self.max_iterations):
            current_output = self._forward_layers(
                prev_output, block_layers, causal_mask, position_embeddings, position_ids
            )
            iterations += 1

            sim = self._cosine_sim(prev_output, current_output)
            trajectory.append(sim)
            l2_norms.append(current_output.float().norm(dim=-1).mean().item())

            prev_output = current_output

            if sim > self.threshold:
                break
```

Also update the return `info` dict (approximately line 137). Replace:

```python
        info = {
            "iterations": iterations,
            "final_similarity": trajectory[-1] if trajectory else None,
            "trajectory": trajectory,
            "halted_early": iterations < self.max_iterations,
        }
```

With:

```python
        info = {
            "iterations": iterations,
            "final_similarity": trajectory[-1] if trajectory else None,
            "trajectory": trajectory,
            "l2_norms": l2_norms,
            "halted_early": iterations < self.max_iterations,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_adaptive_loop.py -v`

Expected: All 3 tests pass.

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest -v`

Expected: All tests pass (both convergence_trace and adaptive_loop tests).

- [ ] **Step 6: Commit**

```bash
git add src/inference/adaptive_loop.py tests/inference/test_adaptive_loop.py
git commit -m "Add L2 norm capture to AdaptiveLoop.forward()"
```

---

### Task 4: ConvergenceTracer

**Files:**
- Create: `src/inference/convergence_tracer.py`
- Create: `tests/inference/test_convergence_tracer.py`

- [ ] **Step 1: Write tests for the tracer**

Create `tests/inference/test_convergence_tracer.py`:

```python
"""Tests for ConvergenceTracer."""
import torch

from src.inference.convergence_tracer import ConvergenceTracer
from src.inference.convergence_trace import ConvergenceTrace, TokenTrace


class MockTokenizer:
    """Minimal tokenizer for testing."""
    def __init__(self, vocab_size=100):
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors=None):
        # Return 5 fixed tokens for any input
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        return type("Encoding", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=False):
        return "mock output"


class TestConvergenceTracer:
    def test_trace_returns_convergence_trace(self, mock_model, block_config):
        tracer = ConvergenceTracer(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=3,
        )
        tokenizer = MockTokenizer()
        input_ids = torch.tensor([[1, 2, 3]])

        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text="What is 2+2?",
            score=1.0,
            max_new_tokens=3,
        )

        assert isinstance(trace, ConvergenceTrace)
        assert trace.prompt == "What is 2+2?"
        assert trace.score == 1.0
        assert trace.threshold == 0.5
        assert trace.max_iterations == 3

    def test_trace_captures_token_traces(self, mock_model, block_config):
        tracer = ConvergenceTracer(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=3,
        )
        tokenizer = MockTokenizer()
        input_ids = torch.tensor([[1, 2, 3]])

        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text="test",
            score=0.0,
            max_new_tokens=3,
        )

        assert len(trace.token_traces) > 0
        for tt in trace.token_traces:
            assert isinstance(tt, TokenTrace)
            assert tt.iterations > 0
            assert len(tt.similarities) == tt.iterations
            assert len(tt.l2_norms) == tt.iterations
            assert tt.elapsed_s >= 0

    def test_trace_generated_text_populated(self, mock_model, block_config):
        tracer = ConvergenceTracer(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=3,
        )
        tokenizer = MockTokenizer()
        input_ids = torch.tensor([[1, 2, 3]])

        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text="test",
            score=0.0,
            max_new_tokens=3,
        )

        assert isinstance(trace.generated, str)

    def test_trace_summary_consistent(self, mock_model, block_config):
        tracer = ConvergenceTracer(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=3,
        )
        tokenizer = MockTokenizer()
        input_ids = torch.tensor([[1, 2, 3]])

        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text="test",
            score=0.0,
            max_new_tokens=3,
        )

        summary = trace.summary()
        assert summary["total_tokens"] == len(trace.token_traces)
        assert summary["avg_iterations"] > 0

    def test_trace_with_logits_processor(self, mock_model, block_config):
        tracer = ConvergenceTracer(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=3,
        )
        tokenizer = MockTokenizer()
        input_ids = torch.tensor([[1, 2, 3]])

        calls = []

        class MockProcessor:
            def reset(self):
                pass

            def __call__(self, input_ids, logits):
                calls.append(1)
                return logits

        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text="test",
            score=0.0,
            max_new_tokens=3,
            logits_processor=MockProcessor(),
        )

        assert len(calls) > 0
        assert isinstance(trace, ConvergenceTrace)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/inference/test_convergence_tracer.py -v`

Expected: `ModuleNotFoundError: No module named 'src.inference.convergence_tracer'`

- [ ] **Step 3: Implement ConvergenceTracer**

Create `src/inference/convergence_tracer.py`:

```python
"""Convergence tracer for capturing rich per-token convergence data.

Wraps AdaptiveLoop to collect structured ConvergenceTrace objects during
generation. The tracer delegates all inference to the loop — it only adds
measurement (timing, L2 norms, structured output).
"""
import time

import torch

from src.inference.adaptive_loop import AdaptiveLoop
from src.inference.convergence_trace import ConvergenceTrace, TokenTrace


class ConvergenceTracer:
    """Wraps AdaptiveLoop to produce ConvergenceTrace objects.

    Args:
        model: HuggingFace causal LM.
        block_i: Start layer of circuit block (inclusive).
        block_j: End layer of circuit block (exclusive).
        threshold: Cosine similarity halting threshold.
        max_iterations: Safety cap on block passes.
    """

    def __init__(self, model, block_i, block_j, threshold=0.995, max_iterations=4):
        self.loop = AdaptiveLoop(model, block_i, block_j, threshold, max_iterations)
        self.threshold = threshold
        self.max_iterations = max_iterations

    def trace_generation(self, input_ids, tokenizer, prompt_text, score,
                         max_new_tokens=256, logits_processor=None):
        """Generate with full convergence tracing.

        Args:
            input_ids: (batch=1, seq_len) token IDs.
            tokenizer: HuggingFace tokenizer.
            prompt_text: Original prompt string (for the trace record).
            score: Evaluation score for this generation (set by caller).
            max_new_tokens: Generation length cap.
            logits_processor: Optional constrained decoding processor.

        Returns:
            ConvergenceTrace with per-token convergence data.
        """
        token_traces = []

        if logits_processor is not None:
            logits_processor.reset()

        current_ids = input_ids.clone()
        eos_id = tokenizer.eos_token_id

        with torch.no_grad():
            for _ in range(max_new_tokens):
                t0 = time.perf_counter()
                logits, info = self.loop.forward(current_ids)
                elapsed = time.perf_counter() - t0

                last_logits = logits[:, -1, :]
                if logits_processor is not None:
                    last_logits = logits_processor(current_ids, last_logits)
                next_token = last_logits.argmax(dim=-1, keepdim=True)

                token_traces.append(TokenTrace(
                    token_id=next_token.item(),
                    iterations=info["iterations"],
                    similarities=info["trajectory"],
                    l2_norms=info["l2_norms"],
                    halted_early=info["halted_early"],
                    elapsed_s=elapsed,
                ))

                current_ids = torch.cat([current_ids, next_token], dim=1)

                if next_token.item() == eos_id:
                    break

        generated_ids = current_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return ConvergenceTrace(
            prompt=prompt_text,
            generated=generated_text,
            score=score,
            threshold=self.threshold,
            max_iterations=self.max_iterations,
            token_traces=token_traces,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_convergence_tracer.py -v`

Expected: All 5 tests pass.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`

Expected: All 18 tests pass (10 trace + 3 loop + 5 tracer).

- [ ] **Step 6: Commit**

```bash
git add src/inference/convergence_tracer.py tests/inference/test_convergence_tracer.py
git commit -m "Add ConvergenceTracer for structured convergence data capture"
```

---

### Task 5: Integration Smoke Test with a Real Probe

**Files:**
- Create: `tests/inference/test_tracer_integration.py`

This test runs the tracer on a real math probe (loaded from `data/math_probe.json`) using the mock model. It verifies the full pipeline: trace generation -> serialization -> deserialization -> summary statistics.

- [ ] **Step 1: Write the integration test**

Create `tests/inference/test_tracer_integration.py`:

```python
"""Integration test: trace a math probe end-to-end and verify serialization."""
import json

import torch

from src.inference.convergence_tracer import ConvergenceTracer
from src.inference.convergence_trace import ConvergenceTrace
from tests.inference.test_convergence_tracer import MockTokenizer


class TestTracerIntegration:
    def test_trace_serialize_deserialize_roundtrip(self, mock_model, block_config):
        tracer = ConvergenceTracer(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=3,
        )
        tokenizer = MockTokenizer()
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text="What is 7 + 3 * 5?",
            score=0.75,
            max_new_tokens=5,
        )

        # Serialize to JSON string
        trace_dict = trace.to_dict()
        json_str = json.dumps(trace_dict, indent=2)

        # Deserialize back
        restored_dict = json.loads(json_str)
        restored = ConvergenceTrace.from_dict(restored_dict)

        # Verify roundtrip
        assert restored.prompt == trace.prompt
        assert restored.generated == trace.generated
        assert restored.score == trace.score
        assert restored.threshold == trace.threshold
        assert restored.max_iterations == trace.max_iterations
        assert len(restored.token_traces) == len(trace.token_traces)

        for orig, rest in zip(trace.token_traces, restored.token_traces):
            assert orig.token_id == rest.token_id
            assert orig.iterations == rest.iterations
            assert orig.similarities == rest.similarities
            assert orig.l2_norms == rest.l2_norms

    def test_trace_batch_multiple_probes(self, mock_model, block_config):
        tracer = ConvergenceTracer(
            mock_model,
            block_config["block_i"],
            block_config["block_j"],
            threshold=0.5,
            max_iterations=3,
        )
        tokenizer = MockTokenizer()

        probes = [
            {"prompt": "What is 2+2?", "input_ids": [1, 2, 3]},
            {"prompt": "What is 10*5?", "input_ids": [4, 5, 6, 7]},
            {"prompt": "What is sqrt(144)?", "input_ids": [8, 9, 10]},
        ]

        traces = []
        for p in probes:
            ids = torch.tensor([p["input_ids"]])
            trace = tracer.trace_generation(
                input_ids=ids,
                tokenizer=tokenizer,
                prompt_text=p["prompt"],
                score=0.0,
                max_new_tokens=3,
            )
            traces.append(trace)

        assert len(traces) == 3
        for trace in traces:
            assert len(trace.token_traces) > 0
            summary = trace.summary()
            assert summary["total_tokens"] > 0
            assert summary["avg_iterations"] > 0

        # All traces should be independently serializable
        batch_json = json.dumps([t.to_dict() for t in traces])
        restored = json.loads(batch_json)
        assert len(restored) == 3
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/inference/test_tracer_integration.py -v`

Expected: Both tests pass.

- [ ] **Step 3: Run the full test suite one final time**

Run: `uv run pytest -v`

Expected: All 20 tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/inference/test_tracer_integration.py
git commit -m "Add integration tests for convergence trace serialization roundtrip"
```
