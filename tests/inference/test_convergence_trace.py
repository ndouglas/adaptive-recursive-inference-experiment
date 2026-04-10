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
