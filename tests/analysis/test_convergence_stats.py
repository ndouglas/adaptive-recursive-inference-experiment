"""Tests for ConvergenceAnalyzer."""
import numpy as np
import pytest

from src.analysis.convergence_stats import ConvergenceAnalyzer


def _make_traces(n=20, seed=42):
    """Generate synthetic trace dicts for testing.

    Correct answers get fast convergence (low iterations, high similarity).
    Wrong answers get slow convergence (high iterations, low similarity).
    This simulates the ideal case where convergence perfectly predicts correctness.
    """
    rng = np.random.RandomState(seed)
    traces = []
    for i in range(n):
        correct = i < n // 2  # first half correct, second half wrong
        avg_iters = rng.uniform(1.0, 2.0) if correct else rng.uniform(2.5, 4.0)
        avg_sim = rng.uniform(0.95, 0.99) if correct else rng.uniform(0.80, 0.90)
        speed = rng.uniform(0.05, 0.10) if correct else rng.uniform(0.01, 0.03)
        traces.append({
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "summary": {
                "avg_iterations": avg_iters,
                "avg_final_similarity": avg_sim,
                "avg_convergence_speed": speed,
                "pct_early_halt": 0.8 if correct else 0.2,
            },
            "category": "easy" if i % 3 == 0 else "hard",
            "difficulty": "easy" if correct else "hard",
        })
    rng.shuffle(traces)
    return traces


def _make_uncorrelated_traces(n=20, seed=99):
    """Traces where convergence does NOT predict correctness."""
    rng = np.random.RandomState(seed)
    traces = []
    for i in range(n):
        correct = rng.random() > 0.5
        avg_iters = rng.uniform(1.0, 4.0)
        avg_sim = rng.uniform(0.80, 0.99)
        speed = rng.uniform(0.01, 0.10)
        traces.append({
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "summary": {
                "avg_iterations": avg_iters,
                "avg_final_similarity": avg_sim,
                "avg_convergence_speed": speed,
                "pct_early_halt": rng.random(),
            },
            "category": "mixed",
        })
    return traces


class TestConvergenceAnalyzer:
    def test_correlations_with_correlated_data(self):
        traces = _make_traces(n=40)
        analyzer = ConvergenceAnalyzer(traces)
        corr = analyzer.compute_correlations()

        # With perfectly correlated data, iterations should negatively correlate with score
        assert corr["avg_iterations"]["r"] < -0.5
        # Similarity should positively correlate with score
        assert corr["avg_final_similarity"]["r"] > 0.5
        # Convergence speed should positively correlate with score
        assert corr["avg_convergence_speed"]["r"] > 0.5

    def test_correlations_with_uncorrelated_data(self):
        traces = _make_uncorrelated_traces(n=40)
        analyzer = ConvergenceAnalyzer(traces)
        corr = analyzer.compute_correlations()

        # Correlations should be weak
        for metric in ["avg_iterations", "avg_final_similarity", "avg_convergence_speed"]:
            assert abs(corr[metric]["r"]) < 0.5

    def test_roc_auc_with_correlated_data(self):
        traces = _make_traces(n=40)
        analyzer = ConvergenceAnalyzer(traces)
        roc = analyzer.compute_roc_auc()

        # Should be able to separate correct from incorrect
        assert roc["avg_iterations"]["auc"] > 0.8
        assert roc["avg_final_similarity"]["auc"] > 0.8
        assert roc["avg_convergence_speed"]["auc"] > 0.8

    def test_roc_auc_with_uncorrelated_data(self):
        traces = _make_uncorrelated_traces(n=40)
        analyzer = ConvergenceAnalyzer(traces)
        roc = analyzer.compute_roc_auc()

        # AUC should be near 0.5 (random)
        for metric in ["avg_iterations", "avg_final_similarity", "avg_convergence_speed"]:
            assert 0.2 < roc[metric]["auc"] < 0.8

    def test_bootstrap_ci(self):
        traces = _make_traces(n=40)
        analyzer = ConvergenceAnalyzer(traces)
        corr = analyzer.compute_correlations(bootstrap_n=100)

        for metric in ["avg_iterations", "avg_final_similarity"]:
            ci = corr[metric]["ci_95"]
            assert len(ci) == 2
            assert ci[0] < corr[metric]["r"] < ci[1]

    def test_stratified_by_category(self):
        traces = _make_traces(n=40)
        analyzer = ConvergenceAnalyzer(traces)
        strat = analyzer.stratified_correlations(group_key="category")

        assert "easy" in strat
        assert "hard" in strat
        for group in strat.values():
            assert "avg_iterations" in group
            assert "n" in group

    def test_summary_table(self):
        traces = _make_traces(n=40)
        analyzer = ConvergenceAnalyzer(traces)
        table = analyzer.summary_table()

        assert "overall" in table
        assert "correlations" in table["overall"]
        assert "roc_auc" in table["overall"]
        assert "n" in table["overall"]

    def test_constant_metric_no_warning(self):
        """When a metric is constant, return NaN without scipy warning."""
        traces = _make_traces(n=20)
        # Make pct_early_halt constant
        for t in traces:
            t["summary"]["pct_early_halt"] = 1.0
        analyzer = ConvergenceAnalyzer(traces)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            corr = analyzer.compute_correlations(bootstrap_n=100)
        # pct_early_halt should be NaN, others should be normal
        assert np.isnan(corr["pct_early_halt"]["r"])
        assert np.isnan(corr["pct_early_halt"]["p"])
        assert np.isnan(corr["pct_early_halt"]["ci_95"][0])
        assert not np.isnan(corr["avg_iterations"]["r"])
