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
