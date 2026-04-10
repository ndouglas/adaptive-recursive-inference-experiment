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
