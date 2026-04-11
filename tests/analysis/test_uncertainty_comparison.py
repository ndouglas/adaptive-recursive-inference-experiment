"""Tests for UncertaintyComparison."""
import numpy as np
import pytest

from src.analysis.uncertainty_comparison import UncertaintyComparison


def _make_matched_data(n=100, seed=42):
    """Create matched data where convergence and entropy correlate with correctness."""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n):
        correct = rng.random() > 0.3  # 70% correct
        if correct:
            similarity = rng.uniform(0.85, 0.99)
            agreement = rng.uniform(0.7, 1.0)
            entropy = rng.uniform(0.5, 2.0)
            iterations = rng.uniform(1.0, 2.0)
        else:
            similarity = rng.uniform(0.70, 0.90)
            agreement = rng.uniform(0.2, 0.6)
            entropy = rng.uniform(2.0, 5.0)
            iterations = rng.uniform(2.0, 4.0)
        data.append({
            "question": f"q{_}",
            "correct": correct,
            "convergence_similarity": similarity,
            "convergence_iterations": iterations,
            "convergence_speed": rng.uniform(0.01, 0.10),
            "sampling_agreement": agreement,
            "mean_entropy": entropy,
        })
    return data


def _make_random_data(n=100, seed=99):
    """Create data where signals are random (no predictive power)."""
    rng = np.random.RandomState(seed)
    data = []
    for _ in range(n):
        data.append({
            "question": f"q{_}",
            "correct": rng.random() > 0.5,
            "convergence_similarity": rng.uniform(0.70, 0.99),
            "convergence_iterations": rng.uniform(1.0, 4.0),
            "convergence_speed": rng.uniform(0.01, 0.10),
            "sampling_agreement": rng.uniform(0.2, 1.0),
            "mean_entropy": rng.uniform(0.5, 5.0),
        })
    return data


class TestUncertaintyComparison:
    def test_roc_auc_all_with_signal(self):
        data = _make_matched_data()
        comp = UncertaintyComparison(data)
        aucs = comp.roc_auc_all()
        assert aucs["convergence_similarity"]["auc"] > 0.6
        assert aucs["sampling_agreement"]["auc"] > 0.6
        assert aucs["mean_entropy"]["auc"] > 0.6

    def test_roc_auc_all_without_signal(self):
        data = _make_random_data()
        comp = UncertaintyComparison(data)
        aucs = comp.roc_auc_all()
        for method, result in aucs.items():
            assert 0.3 < result["auc"] < 0.7

    def test_roc_auc_returns_expected_keys(self):
        data = _make_matched_data(n=50)
        comp = UncertaintyComparison(data)
        aucs = comp.roc_auc_all()
        expected_methods = {
            "convergence_similarity", "convergence_iterations",
            "convergence_speed", "sampling_agreement", "mean_entropy",
        }
        assert set(aucs.keys()) == expected_methods
        for method, result in aucs.items():
            assert "auc" in result
            assert 0 <= result["auc"] <= 1

    def test_pareto_data(self):
        data = _make_matched_data()
        comp = UncertaintyComparison(data, block_layers=6, total_layers=32,
                                     num_samples=8)
        pareto = comp.pareto_data()
        assert len(pareto) > 0
        for point in pareto:
            assert "method" in point
            assert "auc" in point
            assert "cost" in point
            assert point["cost"] > 0

    def test_pareto_cost_ordering(self):
        data = _make_matched_data()
        comp = UncertaintyComparison(data, block_layers=6, total_layers=32,
                                     num_samples=8)
        pareto = comp.pareto_data()
        costs = {p["method"]: p["cost"] for p in pareto}
        assert costs["mean_entropy"] < costs["convergence_similarity"]
        assert costs["convergence_similarity"] < costs["sampling_agreement"]

    def test_disagreement_analysis(self):
        data = _make_matched_data()
        comp = UncertaintyComparison(data)
        disagree = comp.disagreement_analysis(
            method_a="convergence_similarity",
            method_b="sampling_agreement",
            threshold_a=0.5,
            threshold_b=0.5,
        )
        assert "a_confident_b_not" in disagree
        assert "b_confident_a_not" in disagree
        assert "both_confident" in disagree
        assert "neither_confident" in disagree
        total = sum(disagree[k]["count"] for k in disagree)
        assert total == len(data)

    def test_summary_table(self):
        data = _make_matched_data()
        comp = UncertaintyComparison(data, block_layers=6, total_layers=32,
                                     num_samples=8)
        table = comp.summary_table()
        assert "methods" in table
        assert "n" in table
        assert table["n"] == len(data)
        assert len(table["methods"]) == 5
        for entry in table["methods"]:
            assert "method" in entry
            assert "auc" in entry
            assert "cost" in entry
