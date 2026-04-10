"""Tests for TokenProfileAnalyzer."""
import numpy as np
import pytest

from src.analysis.token_profiles import TokenProfileAnalyzer


def _make_trace(generated, token_traces, correct, category="math"):
    """Build a minimal trace dict matching collect_traces.py output format."""
    return {
        "generated": generated,
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "category": category,
        "token_traces": token_traces,
    }


def _make_token_trace(token_id, iterations, final_sim, speed):
    return {
        "token_id": token_id,
        "iterations": iterations,
        "similarities": [0.5, final_sim] if iterations >= 2 else [final_sim],
        "l2_norms": [1.0] * iterations,
        "halted_early": iterations < 4,
        "elapsed_s": 0.01 * iterations,
        "final_similarity": final_sim,
        "convergence_speed": speed,
    }


class TestTokenProfileAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return TokenProfileAnalyzer("Qwen/Qwen2.5-1.5B")

    def test_aggregate_by_role(self, analyzer):
        gen = '{"reasoning": "2 + 3 = 5", "answer": 5}'
        n_tokens = len(analyzer.classifier.tokenizer.encode(gen, add_special_tokens=False))
        tts = [_make_token_trace(0, 2, 0.95, 0.05) for _ in range(n_tokens)]
        trace = _make_trace(gen, tts, correct=True)

        result = analyzer.aggregate_by_role([trace])
        assert "structural" in result
        assert "reasoning" in result
        assert "answer" in result
        for role_data in result.values():
            assert "mean_iterations" in role_data
            assert "mean_final_similarity" in role_data
            assert "mean_convergence_speed" in role_data
            assert "count" in role_data

    def test_correct_vs_incorrect(self, analyzer):
        gen_correct = '{"reasoning": "easy math", "answer": 5}'
        gen_wrong = '{"reasoning": "wrong math", "answer": 3}'
        n_c = len(analyzer.classifier.tokenizer.encode(gen_correct, add_special_tokens=False))
        n_w = len(analyzer.classifier.tokenizer.encode(gen_wrong, add_special_tokens=False))

        tts_correct = [_make_token_trace(0, 1, 0.98, 0.08) for _ in range(n_c)]
        tts_wrong = [_make_token_trace(0, 3, 0.85, 0.02) for _ in range(n_w)]

        trace_c = _make_trace(gen_correct, tts_correct, correct=True)
        trace_w = _make_trace(gen_wrong, tts_wrong, correct=False)

        result = analyzer.compare_correct_vs_incorrect([trace_c, trace_w])
        assert "correct" in result
        assert "incorrect" in result
        for role in ["reasoning", "answer"]:
            if role in result["correct"] and role in result["incorrect"]:
                assert result["correct"][role]["mean_final_similarity"] > \
                       result["incorrect"][role]["mean_final_similarity"]

    def test_positional_profile(self, analyzer):
        gen = '{"reasoning": "x = 5", "answer": 5}'
        n_tokens = len(analyzer.classifier.tokenizer.encode(gen, add_special_tokens=False))
        tts = [_make_token_trace(0, i % 4 + 1, 0.9 + i * 0.001, 0.05)
               for i in range(n_tokens)]
        trace = _make_trace(gen, tts, correct=True)

        profile = analyzer.positional_profile([trace])
        assert "positions" in profile
        assert "mean_iterations" in profile
        assert "mean_final_similarity" in profile
        assert len(profile["positions"]) == n_tokens
        assert len(profile["mean_iterations"]) == n_tokens
