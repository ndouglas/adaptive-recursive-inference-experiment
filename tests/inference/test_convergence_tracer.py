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
