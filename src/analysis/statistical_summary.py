"""Consolidate all key statistics for the writeup.

Produces a structured dict with correlations, ROC AUC, effect sizes,
calibration ECE, and phase transition parameters — every number
referenced in WRITEUP.md.
"""
import json

import numpy as np
from pathlib import Path

from src.analysis.convergence_stats import ConvergenceAnalyzer, CONVERGENCE_METRICS
from src.analysis.calibration import CalibrationAnalyzer


class StatisticalSummary:
    """Collect all experimental statistics into one structured summary.

    Args:
        results_dir: Path to directory containing result JSON files.
    """

    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)

    def _load_json(self, filename):
        with open(self.results_dir / filename) as f:
            return json.load(f)

    def _load_traces(self, filename):
        return self._load_json(filename)["traces"]

    def _cohens_d(self, group1, group2):
        """Compute Cohen's d effect size between two groups."""
        g1, g2 = np.asarray(group1, dtype=float), np.asarray(group2, dtype=float)
        n1, n2 = len(g1), len(g2)
        if n1 < 2 or n2 < 2:
            return float("nan")
        var1 = np.var(g1, ddof=1)
        var2 = np.var(g2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0:
            return float("nan")
        return float((np.mean(g1) - np.mean(g2)) / pooled_std)

    def _compute_effect_sizes(self, traces):
        """Compute Cohen's d for each metric: correct vs incorrect."""
        correct = [t for t in traces if t["correct"]]
        incorrect = [t for t in traces if not t["correct"]]
        result = {}
        for metric in CONVERGENCE_METRICS:
            c_vals = [t["summary"].get(metric, 0) for t in correct]
            i_vals = [t["summary"].get(metric, 0) for t in incorrect]
            result[metric] = {"cohens_d": self._cohens_d(c_vals, i_vals)}
        return result

    def _analyze_task(self, trace_file, bootstrap_n=1000):
        """Analyze one task type (math or reasoning)."""
        traces = self._load_traces(trace_file)
        analyzer = ConvergenceAnalyzer(traces)

        n = len(traces)
        accuracy = sum(1 for t in traces if t["correct"]) / n

        correlations = analyzer.compute_correlations(bootstrap_n=bootstrap_n)
        roc_auc = analyzer.compute_roc_auc()
        effect_sizes = self._compute_effect_sizes(traces)

        calibration = {}
        for metric in CONVERGENCE_METRICS:
            cal = CalibrationAnalyzer(traces, metric=metric)
            calibration[metric] = {"ece": cal.expected_calibration_error()}

        return {
            "n": n,
            "accuracy": round(accuracy, 4),
            "correlations": correlations,
            "roc_auc": roc_auc,
            "effect_sizes": effect_sizes,
            "calibration": calibration,
        }

    def _analyze_phase_transition(self, sweep_file):
        """Extract phase transition parameters from a sweep file."""
        data = self._load_json(sweep_file)
        results = sorted(data["results"], key=lambda r: r["threshold"])

        thresholds = [r["threshold"] for r in results]
        accuracies = [r.get("accuracy", r.get("mean_score", 0)) for r in results]
        iterations = [r.get("mean_avg_iterations", float("nan")) for r in results]

        critical_theta = None
        for i, acc in enumerate(accuracies):
            if acc < 0.85 and i > 0:
                critical_theta = thresholds[i - 1]
                break

        return {
            "thresholds": thresholds,
            "accuracies": accuracies,
            "iterations": iterations,
            "critical_theta": critical_theta,
        }

    def to_json(self, path, trace_configs=None, sweep_configs=None,
                comparison_configs=None, bootstrap_n=1000):
        """Compute all statistics and save to JSON.

        Args:
            path: Output JSON file path.
            trace_configs: Dict of {name: trace_filename} for task analysis.
            sweep_configs: Dict of {name: sweep_filename} for phase transitions.
            comparison_configs: Dict of {name: {conv, samples, entropy}} (optional).
            bootstrap_n: Number of bootstrap resamples for CIs.

        Returns:
            The summary dict.
        """
        trace_configs = trace_configs or {
            "math": "traces_7b_math_expanded.json",
            "reasoning": "traces_7b_reasoning_expanded_t0.80.json",
        }
        sweep_configs = sweep_configs or {
            "math": "phase_sweep_math.json",
            "reasoning": "phase_sweep_reasoning.json",
        }

        summary = {"tasks": {}, "phase_transitions": {}}

        for name, filename in trace_configs.items():
            summary["tasks"][name] = self._analyze_task(filename, bootstrap_n)

        for name, filename in sweep_configs.items():
            summary["phase_transitions"][name] = self._analyze_phase_transition(filename)

        if comparison_configs:
            summary["comparisons"] = {}
            for name, cfg in comparison_configs.items():
                summary["comparisons"][name] = self._analyze_comparison(**cfg)

        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary
