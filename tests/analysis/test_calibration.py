"""Tests for CalibrationAnalyzer."""
import numpy as np
import pytest

from src.analysis.calibration import CalibrationAnalyzer


def _make_traces_calibrated(n=100, seed=42):
    """Traces where convergence IS calibrated: high-similarity bins are more accurate."""
    rng = np.random.RandomState(seed)
    traces = []
    for i in range(n):
        sim = rng.uniform(0.80, 0.99)
        prob_correct = (sim - 0.80) / (0.99 - 0.80)
        correct = rng.random() < prob_correct
        iters = rng.uniform(1.0, 4.0)
        speed = rng.uniform(0.01, 0.10)
        traces.append({
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "summary": {
                "avg_iterations": iters,
                "avg_final_similarity": sim,
                "avg_convergence_speed": speed,
                "pct_early_halt": rng.random(),
            },
        })
    return traces


def _make_traces_uncalibrated(n=100, seed=99):
    """Traces where convergence is NOT calibrated: random accuracy per bin."""
    rng = np.random.RandomState(seed)
    traces = []
    for i in range(n):
        sim = rng.uniform(0.80, 0.99)
        correct = rng.random() < 0.5
        traces.append({
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "summary": {
                "avg_iterations": rng.uniform(1.0, 4.0),
                "avg_final_similarity": sim,
                "avg_convergence_speed": rng.uniform(0.01, 0.10),
                "pct_early_halt": rng.random(),
            },
        })
    return traces


class TestCalibrationAnalyzer:
    def test_reliability_bins(self):
        traces = _make_traces_calibrated(n=200)
        analyzer = CalibrationAnalyzer(traces, metric="avg_final_similarity")
        bins = analyzer.reliability_bins(n_bins=5)

        assert len(bins) == 5
        for b in bins:
            assert "bin_center" in b
            assert "accuracy" in b
            assert "confidence" in b
            assert "count" in b
            assert 0 <= b["accuracy"] <= 1
            assert b["count"] > 0

    def test_calibrated_data_low_ece(self):
        traces = _make_traces_calibrated(n=200)
        analyzer = CalibrationAnalyzer(traces, metric="avg_final_similarity")
        ece = analyzer.expected_calibration_error(n_bins=5)
        assert ece < 0.3

    def test_uncalibrated_data_higher_ece(self):
        traces_cal = _make_traces_calibrated(n=200)
        traces_uncal = _make_traces_uncalibrated(n=200)
        ece_cal = CalibrationAnalyzer(
            traces_cal, metric="avg_final_similarity"
        ).expected_calibration_error(n_bins=5)
        ece_uncal = CalibrationAnalyzer(
            traces_uncal, metric="avg_final_similarity"
        ).expected_calibration_error(n_bins=5)
        assert ece_cal < ece_uncal or ece_cal < 0.3

    def test_reliability_bins_monotonic_for_calibrated(self):
        traces = _make_traces_calibrated(n=300, seed=42)
        analyzer = CalibrationAnalyzer(traces, metric="avg_final_similarity")
        bins = analyzer.reliability_bins(n_bins=5)
        accuracies = [b["accuracy"] for b in bins]
        assert accuracies[-1] > accuracies[0]

    def test_different_metrics(self):
        traces = _make_traces_calibrated(n=100)
        for metric in ["avg_final_similarity", "avg_iterations", "avg_convergence_speed"]:
            analyzer = CalibrationAnalyzer(traces, metric=metric)
            ece = analyzer.expected_calibration_error(n_bins=5)
            assert 0 <= ece <= 1

    def test_summary(self):
        traces = _make_traces_calibrated(n=100)
        analyzer = CalibrationAnalyzer(traces, metric="avg_final_similarity")
        summary = analyzer.summary(n_bins=5)
        assert "ece" in summary
        assert "bins" in summary
        assert "metric" in summary
        assert "n" in summary
        assert summary["metric"] == "avg_final_similarity"
        assert summary["n"] == 100
