"""Tests for StatisticalSummary."""
import json

import numpy as np
import pytest

from src.analysis.statistical_summary import StatisticalSummary


def _make_fake_traces(n=20, seed=42):
    """Create fake traces with convergence-correctness signal."""
    rng = np.random.RandomState(seed)
    traces = []
    for i in range(n):
        correct = bool(rng.random() > 0.3)
        if correct:
            sim = 0.90 + rng.random() * 0.08
            iters = 1 + rng.random() * 1.5
        else:
            sim = 0.82 + rng.random() * 0.10
            iters = 2 + rng.random() * 2.0
        speed = sim / max(iters, 1)
        traces.append({
            "prompt": f"Question {i}",
            "generated": f'{{"reasoning": "step {i}", "answer": {i}}}',
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "summary": {
                "avg_final_similarity": float(sim),
                "avg_iterations": float(iters),
                "avg_convergence_speed": float(speed),
                "pct_early_halt": float(rng.random()),
            },
            "token_traces": [],
        })
    return traces


def _write_traces(tmp_path, filename, traces, threshold=0.80):
    path = tmp_path / filename
    with open(path, "w") as f:
        json.dump({"traces": traces, "threshold": threshold}, f)
    return path


def _write_sweep(tmp_path, filename):
    data = {
        "results": [
            {"threshold": 0.50, "accuracy": 0.97, "mean_score": 0.97,
             "mean_avg_iterations": 1.0},
            {"threshold": 0.80, "accuracy": 0.93, "mean_score": 0.93,
             "mean_avg_iterations": 1.8},
            {"threshold": 0.95, "accuracy": 0.82, "mean_score": 0.82,
             "mean_avg_iterations": 2.5},
            {"threshold": 0.99, "accuracy": 0.35, "mean_score": 0.35,
             "mean_avg_iterations": 4.0},
        ],
    }
    path = tmp_path / filename
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestCohensD:
    def test_equal_groups_zero(self):
        ss = StatisticalSummary.__new__(StatisticalSummary)
        d = ss._cohens_d([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(d) < 1e-10

    def test_known_effect(self):
        ss = StatisticalSummary.__new__(StatisticalSummary)
        g1 = [0.0, 0.0, 0.0, 0.0]
        g2 = [1.0, 1.0, 1.0, 1.0]
        d = ss._cohens_d(g1, g2)
        assert np.isnan(d)

    def test_moderate_effect(self):
        ss = StatisticalSummary.__new__(StatisticalSummary)
        g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        g2 = [3.0, 4.0, 5.0, 6.0, 7.0]
        d = ss._cohens_d(g1, g2)
        assert d < 0
        assert abs(d) > 1.0

    def test_tiny_group_returns_nan(self):
        ss = StatisticalSummary.__new__(StatisticalSummary)
        d = ss._cohens_d([1.0], [2.0, 3.0, 4.0])
        assert np.isnan(d)


class TestAnalyzeTask:
    def test_returns_expected_keys(self, tmp_path):
        traces = _make_fake_traces(20)
        _write_traces(tmp_path, "traces.json", traces)
        ss = StatisticalSummary(results_dir=tmp_path)
        result = ss._analyze_task("traces.json", bootstrap_n=0)
        assert result["n"] == 20
        assert "accuracy" in result
        assert "correlations" in result
        assert "roc_auc" in result
        assert "effect_sizes" in result
        assert "calibration" in result

    def test_accuracy_matches(self, tmp_path):
        traces = _make_fake_traces(20)
        _write_traces(tmp_path, "traces.json", traces)
        ss = StatisticalSummary(results_dir=tmp_path)
        result = ss._analyze_task("traces.json", bootstrap_n=0)
        expected_acc = sum(1 for t in traces if t["correct"]) / len(traces)
        assert abs(result["accuracy"] - expected_acc) < 1e-6

    def test_effect_sizes_all_metrics(self, tmp_path):
        traces = _make_fake_traces(20)
        _write_traces(tmp_path, "traces.json", traces)
        ss = StatisticalSummary(results_dir=tmp_path)
        result = ss._analyze_task("traces.json", bootstrap_n=0)
        for metric in ["avg_final_similarity", "avg_iterations",
                       "avg_convergence_speed", "pct_early_halt"]:
            assert metric in result["effect_sizes"]
            assert "cohens_d" in result["effect_sizes"][metric]


class TestAnalyzePhaseTransition:
    def test_returns_expected_keys(self, tmp_path):
        _write_sweep(tmp_path, "sweep.json")
        ss = StatisticalSummary(results_dir=tmp_path)
        result = ss._analyze_phase_transition("sweep.json")
        assert "thresholds" in result
        assert "accuracies" in result
        assert "iterations" in result
        assert "critical_theta" in result
        assert len(result["thresholds"]) == 4

    def test_critical_theta_found(self, tmp_path):
        _write_sweep(tmp_path, "sweep.json")
        ss = StatisticalSummary(results_dir=tmp_path)
        result = ss._analyze_phase_transition("sweep.json")
        assert result["critical_theta"] == 0.80


class TestToJson:
    def test_writes_valid_json(self, tmp_path):
        traces = _make_fake_traces(20)
        _write_traces(tmp_path, "traces.json", traces)
        _write_sweep(tmp_path, "sweep.json")
        ss = StatisticalSummary(results_dir=tmp_path)
        out_path = tmp_path / "summary.json"
        ss.to_json(
            out_path,
            trace_configs={"test": "traces.json"},
            sweep_configs={"test": "sweep.json"},
            bootstrap_n=0,
        )
        with open(out_path) as f:
            data = json.load(f)
        assert "test" in data["tasks"]
        assert "test" in data["phase_transitions"]
