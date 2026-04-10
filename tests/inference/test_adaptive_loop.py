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
