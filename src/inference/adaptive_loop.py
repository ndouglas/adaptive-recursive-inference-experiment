"""Adaptive recursive inference with cosine-similarity halting.

Replaces fixed-iteration layer duplication with a convergence-based loop:
run the circuit block until cosine similarity between consecutive outputs
exceeds a threshold, or a max iteration cap is reached.

Design choices:
- Compare output of pass N vs output of pass N-1 (option b from the plan)
- Sequence-averaged cosine similarity (simpler than per-token)
- No KV caching (reprocesses full sequence each forward pass — fine for short probes)
"""
import copy
from contextlib import contextmanager

import torch
import torch.nn.functional as F


class AdaptiveLoop:
    """Run a circuit block adaptively with cosine-similarity halting.

    Args:
        model: HuggingFace causal LM (e.g., Qwen2ForCausalLM)
        block_i: Start layer of the circuit block (inclusive)
        block_j: End layer of the circuit block (exclusive)
        threshold: Cosine similarity threshold for halting (default 0.995)
        max_iterations: Safety cap on total passes through the block (default 4)
    """

    def __init__(self, model, block_i, block_j, threshold=0.995, max_iterations=4):
        self.model = model
        self.block_i = block_i
        self.block_j = block_j
        self.threshold = threshold
        self.max_iterations = max_iterations

        # Cache references to model components
        self.embed_tokens = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary_emb = model.model.rotary_emb
        self.config = model.config
        self.layers = model.model.layers

        # Per-generation diagnostics
        self.token_iterations = []   # iterations used per token
        self.token_similarities = [] # final similarity per token
        self.token_trajectories = [] # full trajectory per token

    def _get_causal_mask(self, seq_len, device, dtype):
        """Create a simple causal attention mask."""
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        # Expand for batch and head dimensions: (1, 1, seq_len, seq_len)
        return mask.unsqueeze(0).unsqueeze(0)

    def _forward_layers(self, hidden_states, layers, attention_mask,
                        position_embeddings, position_ids):
        """Run a sequence of layers on hidden_states."""
        for layer in layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
        return hidden_states

    def _cosine_sim(self, a, b):
        """Sequence-averaged cosine similarity between two hidden states."""
        return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()

    def forward(self, input_ids):
        """Single forward pass with adaptive looping. Returns logits and diagnostics.

        Args:
            input_ids: (batch=1, seq_len) token IDs

        Returns:
            logits: (batch=1, seq_len, vocab_size)
            info: dict with iterations, similarity, trajectory
        """
        device = input_ids.device
        seq_len = input_ids.shape[1]

        # Embed
        hidden_states = self.embed_tokens(input_ids)
        dtype = hidden_states.dtype

        # Position info
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Causal mask — must match model dtype for SDPA attention
        causal_mask = self._get_causal_mask(seq_len, device, dtype)

        # Phase 1: Encoder layers (0 to block_i - 1)
        encoder_layers = list(self.layers[:self.block_i])
        hidden_states = self._forward_layers(
            hidden_states, encoder_layers, causal_mask, position_embeddings, position_ids
        )

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

        hidden_states = prev_output

        # Phase 3: Decoder layers (block_j to end)
        decoder_layers = list(self.layers[self.block_j:])
        hidden_states = self._forward_layers(
            hidden_states, decoder_layers, causal_mask, position_embeddings, position_ids
        )

        # Final norm + LM head
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        info = {
            "iterations": iterations,
            "final_similarity": trajectory[-1] if trajectory else None,
            "trajectory": trajectory,
            "l2_norms": l2_norms,
            "halted_early": iterations < self.max_iterations,
        }

        return logits, info

    def generate(self, input_ids, tokenizer, max_new_tokens=32,
                 logits_processor=None):
        """Greedy decoding with adaptive loop per token.

        No KV caching — reprocesses full sequence each step.
        Fine for short probes, not for long generation.

        Args:
            input_ids: (batch=1, seq_len) token IDs
            tokenizer: HuggingFace tokenizer
            max_new_tokens: generation length cap
            logits_processor: optional callable(input_ids, logits) -> logits
                for constrained decoding (e.g., outlines JSON schema processor)

        Returns:
            generated_ids: full sequence (prompt + generated)
        """
        self.token_iterations = []
        self.token_similarities = []
        self.token_trajectories = []

        if logits_processor is not None:
            logits_processor.reset()

        current_ids = input_ids.clone()
        eos_id = tokenizer.eos_token_id

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, info = self.forward(current_ids)

                self.token_iterations.append(info["iterations"])
                self.token_similarities.append(info["final_similarity"])
                self.token_trajectories.append(info["trajectory"])

                last_logits = logits[:, -1, :]
                if logits_processor is not None:
                    last_logits = logits_processor(current_ids, last_logits)
                next_token = last_logits.argmax(dim=-1, keepdim=True)
                current_ids = torch.cat([current_ids, next_token], dim=1)

                if next_token.item() == eos_id:
                    break

        return current_ids

    def diagnostics_summary(self):
        """Summary of per-token adaptive behavior from the last generation."""
        if not self.token_iterations:
            return {}
        iters = self.token_iterations
        sims = [s for s in self.token_similarities if s is not None]
        return {
            "total_tokens": len(iters),
            "avg_iterations": sum(iters) / len(iters),
            "min_iterations": min(iters),
            "max_iterations": max(iters),
            "iterations_distribution": {k: iters.count(k) for k in sorted(set(iters))},
            "avg_final_similarity": sum(sims) / len(sims) if sims else None,
            "pct_early_halt": sum(1 for i in iters if i < self.max_iterations) / len(iters),
        }


@contextmanager
def adaptive_model(model, block_i, block_j, threshold=0.995, max_iterations=4):
    """Context manager that provides an AdaptiveLoop for the given model."""
    loop = AdaptiveLoop(model, block_i, block_j, threshold, max_iterations)
    yield loop
