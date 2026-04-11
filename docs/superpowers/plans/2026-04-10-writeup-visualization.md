# Writeup and Visualization Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a hybrid technical post (`WRITEUP.md`) with 9 publication-quality figures and a statistical summary module, publishable on GitHub.

**Architecture:** Single `scripts/generate_figures.py` reads from `results/` JSON files and uses existing analysis modules to produce 9 figures into `figures/`. A `StatisticalSummary` class consolidates all key numbers. `WRITEUP.md` at repo root references the figures.

**Tech Stack:** Python, matplotlib, numpy, sklearn (already in use), existing `src/analysis/` modules (`ConvergenceAnalyzer`, `CalibrationAnalyzer`, `UncertaintyComparison`, `TokenProfileAnalyzer`, `TokenRoleClassifier`).

**Design spec:** `docs/superpowers/specs/2026-04-10-writeup-visualization-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/analysis/statistical_summary.py` | Consolidate all key statistics into one structured dict |
| Create | `tests/analysis/test_statistical_summary.py` | Unit tests for statistical summary |
| Create | `scripts/generate_figures.py` | Master figure script producing 9 PNGs |
| Create | `WRITEUP.md` | The hybrid technical post |
| Create | `figures/` (directory) | Output directory for publication figures |

**Existing files used (read-only):**
- `src/analysis/convergence_stats.py` — `ConvergenceAnalyzer` (correlations, ROC AUC)
- `src/analysis/calibration.py` — `CalibrationAnalyzer` (ECE, reliability bins)
- `src/analysis/uncertainty_comparison.py` — `UncertaintyComparison`, `METHOD_CONFIG`
- `src/analysis/token_profiles.py` — `TokenProfileAnalyzer`
- `src/analysis/token_roles.py` — `TokenRoleClassifier`, `TokenRole`
- `scripts/plot_uncertainty_comparison.py` — `merge_data_sources()` function
- All `results/*.json` files

---

### Task 1: Statistical Summary Module (TDD)

**Files:**
- Create: `tests/analysis/test_statistical_summary.py`
- Create: `src/analysis/statistical_summary.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/analysis/test_statistical_summary.py
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
        # Groups differ by exactly 1 std
        g1 = [0.0, 0.0, 0.0, 0.0]
        g2 = [1.0, 1.0, 1.0, 1.0]
        d = ss._cohens_d(g1, g2)
        # Perfect separation: |d| should be very large (inf for zero variance within)
        # With zero within-group variance, pooled_std = 0 -> nan
        assert np.isnan(d)

    def test_moderate_effect(self):
        ss = StatisticalSummary.__new__(StatisticalSummary)
        g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        g2 = [3.0, 4.0, 5.0, 6.0, 7.0]
        d = ss._cohens_d(g1, g2)
        # Mean diff = 2.0, pooled std = sqrt(2.5) ≈ 1.58 -> d ≈ -1.26
        assert d < 0  # g1 < g2
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
        # Accuracy drops below 0.85 at θ=0.95, so critical_theta = 0.80
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/analysis/test_statistical_summary.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.statistical_summary'`

- [ ] **Step 3: Write the implementation**

```python
# src/analysis/statistical_summary.py
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

        # Find critical theta: last threshold with accuracy >= 0.85
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/analysis/test_statistical_summary.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/analysis/statistical_summary.py tests/analysis/test_statistical_summary.py
git commit -m "feat: add StatisticalSummary module for writeup statistics (TDD)"
```

---

### Task 2: Figure Script Scaffold and Method Diagram (Fig 1)

**Files:**
- Create: `scripts/generate_figures.py`

- [ ] **Step 1: Create the figure script with scaffold, style config, and Fig 1**

```python
# scripts/generate_figures.py
"""Generate all publication-quality figures for WRITEUP.md.

Produces 9 figures into figures/ directory with consistent styling.

Usage:
    python scripts/generate_figures.py
    python scripts/generate_figures.py --output-dir figures --results-dir results
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
STYLE = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLORS = {
    "convergence_similarity": "#2196F3",
    "convergence_speed": "#4CAF50",
    "convergence_iterations": "#FF9800",
    "sampling_agreement": "#E91E63",
    "mean_entropy": "#9C27B0",
    "correct": "#2196F3",
    "incorrect": "#E53935",
    "math": "#2196F3",
    "reasoning": "#FF9800",
    "structural": "#888888",
    "reasoning_role": "#4488cc",
    "answer": "#cc4444",
}

METHOD_LABELS = {
    "convergence_similarity": "Conv. Similarity",
    "convergence_speed": "Conv. Speed",
    "convergence_iterations": "Conv. Iterations",
    "sampling_agreement": "Sampling (N=8)",
    "mean_entropy": "Softmax Entropy",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def setup_style():
    plt.rcParams.update(STYLE)


def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 1: Method Diagram
# ---------------------------------------------------------------------------
def fig_method_diagram(out_dir):
    """Schematic of adaptive recursive inference within the transformer."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(-0.5, 4.5)
    ax.axis("off")

    boxes = [
        (0.0, 1.2, 2.0, 1.6, "Input\nTokens", "#E3F2FD", "#1565C0"),
        (2.8, 1.2, 2.2, 1.6, "Layers\n1–14", "#BBDEFB", "#1565C0"),
        (5.8, 0.8, 3.0, 2.4, "Adaptive Loop\nLayers 15–20", "#FFF3E0", "#E65100"),
        (9.6, 1.2, 2.2, 1.6, "Layers\n21–32", "#BBDEFB", "#1565C0"),
        (12.5, 1.2, 1.5, 1.6, "Output", "#E8F5E9", "#2E7D32"),
    ]

    for x, y, w, h, text, face, edge in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                               facecolor=face, edgecolor=edge, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#333")

    # Arrows between boxes
    arrow_kw = dict(arrowstyle="-|>", lw=2, color="#555")
    for x1, x2, y in [(2.0, 2.8, 2.0), (5.0, 5.8, 2.0),
                       (8.8, 9.6, 2.0), (11.8, 12.5, 2.0)]:
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=arrow_kw)

    # Loop arrow
    loop_x = 7.3
    ax.annotate("", xy=(loop_x, 3.3), xytext=(loop_x + 0.8, 3.3),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#E65100",
                                connectionstyle="arc3,rad=-1.2"))
    ax.text(7.3, 3.8, "cos(h_i, h_{i-1}) < θ ?  → repeat", ha="center",
            fontsize=9, color="#E65100", style="italic")

    # Cost annotation
    ax.text(7.3, 0.3, "~1.09 forward passes (6/32 layers × ~2 iterations)",
            ha="center", fontsize=9, color="#666", style="italic")

    savefig(fig, os.path.join(out_dir, "fig1_method_diagram.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate all writeup figures")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name for tokenizer (needed for token role figs)")
    args = parser.parse_args()

    out_dir = args.output_dir
    results_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)
    setup_style()

    print("Generating figures...")
    fig_method_diagram(out_dir)
    # Remaining figures added in subsequent tasks
    print(f"\nDone. {len([f for f in os.listdir(out_dir) if f.endswith('.png')])} figures in {out_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script to verify Fig 1 is produced**

Run: `python scripts/generate_figures.py`
Expected: Output shows `Saved: figures/fig1_method_diagram.png` and `1 figures in figures/`

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_figures.py
git commit -m "feat: add figure generation script scaffold with method diagram (Fig 1)"
```

---

### Task 3: ROC Curves (Fig 2) and Phase Transition (Fig 3)

**Files:**
- Modify: `scripts/generate_figures.py`

**Context:** These are the "surprise" and "three regimes" figures. Fig 2 shows ROC curves for math vs reasoning side-by-side. Fig 3 shows accuracy vs threshold with regime annotations.

**Data files needed:**
- `results/traces_7b_math_expanded.json` — traces with `correct`, `summary.avg_final_similarity`, etc.
- `results/traces_7b_reasoning_expanded_t0.80.json` — same structure for reasoning
- `results/phase_sweep_math.json` — `results` list with `threshold`, `accuracy`, `mean_avg_iterations`
- `results/phase_sweep_reasoning.json` — same structure

- [ ] **Step 1: Add fig_roc_curves() function**

Add this function to `scripts/generate_figures.py` before `main()`:

```python
# ---------------------------------------------------------------------------
# Fig 2: ROC Curves — Math vs Reasoning
# ---------------------------------------------------------------------------
def fig_roc_curves(out_dir, results_dir):
    """Two-panel ROC curves showing convergence predicts math but not reasoning."""
    math_traces = load_json(os.path.join(results_dir, "traces_7b_math_expanded.json"))["traces"]
    reas_traces = load_json(os.path.join(results_dir, "traces_7b_reasoning_expanded_t0.80.json"))["traces"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    metric_map = {
        "avg_final_similarity": ("convergence_similarity", False),
        "avg_convergence_speed": ("convergence_speed", False),
        "avg_iterations": ("convergence_iterations", True),
    }

    for ax, traces, title in [(ax1, math_traces, "Math (n={})"),
                               (ax2, reas_traces, "Reasoning (n={})")]:
        labels = np.array([t["correct"] for t in traces], dtype=int)
        title = title.format(len(traces))

        for metric_key, (color_key, negate) in metric_map.items():
            values = np.array([t["summary"].get(metric_key, 0) for t in traces])
            if negate:
                values = -values
            if len(np.unique(labels)) < 2:
                continue
            try:
                fpr, tpr, _ = roc_curve(labels, values)
                auc = roc_auc_score(labels, values)
                ax.plot(fpr, tpr, color=COLORS[color_key], lw=2,
                        label=f"{METHOD_LABELS[color_key]} (AUC={auc:.3f})")
            except ValueError:
                continue

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random (AUC=0.5)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Convergence Metrics as Correctness Predictors", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig2_roc_curves.png"))
```

- [ ] **Step 2: Add fig_phase_transition() function**

Add this function after `fig_roc_curves()`:

```python
# ---------------------------------------------------------------------------
# Fig 3: Phase Transition — Three Regimes
# ---------------------------------------------------------------------------
def fig_phase_transition(out_dir, results_dir):
    """Accuracy and iterations vs threshold with regime annotations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for sweep_file, label, color in [
        ("phase_sweep_math.json", "Math", COLORS["math"]),
        ("phase_sweep_reasoning.json", "Reasoning", COLORS["reasoning"]),
    ]:
        data = load_json(os.path.join(results_dir, sweep_file))
        results = sorted(data["results"], key=lambda r: r["threshold"])
        thresholds = [r["threshold"] for r in results]
        accuracies = [r.get("accuracy", r.get("mean_score", 0)) for r in results]
        iterations = [r.get("mean_avg_iterations", float("nan")) for r in results]

        ax1.plot(thresholds, accuracies, "o-", color=color, label=label,
                 markersize=5, linewidth=2)
        ax2.plot(thresholds, iterations, "o-", color=color, label=label,
                 markersize=5, linewidth=2)

    # Regime shading on accuracy plot
    ax1.axvspan(0.0, 0.70, alpha=0.06, color="green", label="Safe")
    ax1.axvspan(0.70, 0.95, alpha=0.06, color="gold", label="Plateau")
    ax1.axvspan(0.95, 1.0, alpha=0.06, color="red", label="Cliff")

    # Critical theta markers
    ax1.axvline(x=0.96, color=COLORS["math"], linestyle=":", alpha=0.6, linewidth=1.5)
    ax1.axvline(x=0.95, color=COLORS["reasoning"], linestyle=":", alpha=0.6, linewidth=1.5)
    ax1.text(0.96, 0.25, "θ*≈0.96\n(math)", fontsize=8, color=COLORS["math"],
             ha="center")
    ax1.text(0.935, 0.18, "θ*≈0.95\n(reas.)", fontsize=8, color=COLORS["reasoning"],
             ha="center")

    ax1.set_xlabel("Similarity Threshold (θ)")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs Threshold")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.set_ylim(0, 1.05)

    ax2.set_xlabel("Similarity Threshold (θ)")
    ax2.set_ylabel("Mean Iterations")
    ax2.set_title("Mean Iterations vs Threshold")
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle("Phase Transition: Three-Regime Structure", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig3_phase_transition.png"))
```

- [ ] **Step 3: Update main() to call the new functions**

In `main()`, replace `# Remaining figures added in subsequent tasks` with:

```python
    fig_roc_curves(out_dir, results_dir)
    fig_phase_transition(out_dir, results_dir)
```

- [ ] **Step 4: Run the script to verify 3 figures are produced**

Run: `python scripts/generate_figures.py`
Expected: Output shows 3 `Saved:` lines and `3 figures in figures/`

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_figures.py
git commit -m "feat: add ROC curves (Fig 2) and phase transition (Fig 3)"
```

---

### Task 4: Token Role Comparison (Fig 4) and Convergence Heatmaps (Fig 5)

**Files:**
- Modify: `scripts/generate_figures.py`

**Context:** These figures require the `TokenProfileAnalyzer` and `TokenRoleClassifier` from `src/analysis/`, which need a HuggingFace tokenizer. The model name is passed via `--model-name` CLI arg (default `Qwen/Qwen2.5-7B-Instruct`). The tokenizer will download automatically if not cached (~2MB).

**Data files needed:**
- `results/traces_7b_math_expanded.json` — traces with `generated` text and `token_traces`
- `results/traces_7b_reasoning_expanded_t0.80.json` — same

**Key data structures in traces:**
- `trace["generated"]` — the generated JSON string like `{"reasoning": "...", "answer": 42}`
- `trace["token_traces"]` — list of dicts, one per token, each with:
  - `"iterations"`: int
  - `"similarities"`: list of floats (one per iteration)
  - `"final_similarity"`: float or None
  - `"convergence_speed"`: float or None

- [ ] **Step 1: Add fig_token_roles() function**

Add these imports near the top of `scripts/generate_figures.py` (after the existing imports):

```python
from src.analysis.token_profiles import TokenProfileAnalyzer
from src.analysis.token_roles import TokenRoleClassifier, TokenRole
```

Add this function before `main()`:

```python
# ---------------------------------------------------------------------------
# Fig 4: Convergence by Token Role
# ---------------------------------------------------------------------------
def fig_token_roles(out_dir, results_dir, model_name):
    """Bar chart: mean iterations by token role, correct vs incorrect."""
    math_traces = load_json(os.path.join(results_dir, "traces_7b_math_expanded.json"))["traces"]

    analyzer = TokenProfileAnalyzer(model_name)
    comparison = analyzer.compare_correct_vs_incorrect(math_traces)

    roles = ["structural", "reasoning", "answer"]
    role_labels = ["Structural", "Reasoning", "Answer"]
    metrics = [
        ("mean_iterations", "Mean Iterations"),
        ("mean_final_similarity", "Mean Final Similarity"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 5))
    x = np.arange(len(roles))
    width = 0.35

    for ax, (metric, title) in zip(axes, metrics):
        correct_vals = [
            comparison.get("correct", {}).get(r, {}).get(metric, 0) for r in roles
        ]
        incorrect_vals = [
            comparison.get("incorrect", {}).get(r, {}).get(metric, 0) for r in roles
        ]

        bars1 = ax.bar(x - width / 2, correct_vals, width,
                        label="Correct", color=COLORS["correct"], alpha=0.85)
        bars2 = ax.bar(x + width / 2, incorrect_vals, width,
                        label="Incorrect", color=COLORS["incorrect"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(role_labels)
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle("Convergence by Token Role (Math, θ=0.80)", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig4_token_roles.png"))
```

- [ ] **Step 2: Add fig_heatmaps() function**

Add this function after `fig_token_roles()`:

```python
# ---------------------------------------------------------------------------
# Fig 5: Convergence Heatmaps (Representative Examples)
# ---------------------------------------------------------------------------
def _build_heatmap_array(token_traces, max_iters=4):
    """Build (iterations × positions) similarity matrix from token traces."""
    n_tokens = len(token_traces)
    matrix = np.full((max_iters, n_tokens), np.nan)
    for j, tt in enumerate(token_traces):
        sims = tt.get("similarities", [])
        for i, s in enumerate(sims):
            if i < max_iters:
                matrix[i, j] = s
        # Pad with final value (token halted)
        if sims:
            for i in range(len(sims), max_iters):
                matrix[i, j] = sims[-1]
    return matrix


def _select_representative(traces, correct):
    """Pick a median-convergence trace from the correct/incorrect group."""
    group = [t for t in traces if t["correct"] == correct]
    if not group:
        return None
    # Sort by avg_iterations, pick the middle one
    group.sort(key=lambda t: t["summary"].get("avg_iterations", 0))
    return group[len(group) // 2]


def fig_heatmaps(out_dir, results_dir, model_name):
    """2×2 grid of convergence heatmaps: correct/incorrect × math/reasoning."""
    math_traces = load_json(os.path.join(results_dir, "traces_7b_math_expanded.json"))["traces"]
    reas_traces = load_json(os.path.join(results_dir, "traces_7b_reasoning_expanded_t0.80.json"))["traces"]

    classifier = TokenRoleClassifier(model_name)

    selections = [
        (_select_representative(math_traces, True), "Math — Correct"),
        (_select_representative(math_traces, False), "Math — Incorrect"),
        (_select_representative(reas_traces, True), "Reasoning — Correct"),
        (_select_representative(reas_traces, False), "Reasoning — Incorrect"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    role_colors = {
        TokenRole.STRUCTURAL: COLORS["structural"],
        TokenRole.REASONING: COLORS["reasoning_role"],
        TokenRole.ANSWER: COLORS["answer"],
    }

    for idx, (trace, title) in enumerate(selections):
        ax = axes[idx // 2][idx % 2]
        if trace is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        matrix = _build_heatmap_array(trace["token_traces"])
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.7, vmax=1.0,
                        interpolation="nearest", origin="lower")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("Iteration")
        ax.set_title(title, fontsize=11)

        # Role color bar along the top
        try:
            roles = classifier.classify(trace["generated"])
            n = min(len(roles), matrix.shape[1])
            for i in range(n):
                ax.axvspan(i - 0.5, i + 0.5, ymax=1.05, ymin=1.0,
                           color=role_colors.get(roles[i], "#888"),
                           clip_on=False)
        except Exception:
            pass  # Skip role annotations if classification fails

    # Colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Cosine Similarity")

    # Legend for role colors
    legend_patches = [
        mpatches.Patch(color=COLORS["structural"], label="Structural"),
        mpatches.Patch(color=COLORS["reasoning_role"], label="Reasoning"),
        mpatches.Patch(color=COLORS["answer"], label="Answer"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=9,
               bbox_to_anchor=(0.98, 0.98))

    fig.suptitle("Per-Token Convergence Heatmaps", fontsize=14, y=1.02)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    savefig(fig, os.path.join(out_dir, "fig5_heatmaps.png"))
```

- [ ] **Step 3: Update main() to call the new functions**

Add after the existing `fig_phase_transition()` call in `main()`:

```python
    fig_token_roles(out_dir, results_dir, args.model_name)
    fig_heatmaps(out_dir, results_dir, args.model_name)
```

- [ ] **Step 4: Run the script to verify 5 figures are produced**

Run: `python scripts/generate_figures.py`
Expected: Output shows 5 `Saved:` lines including `fig4_token_roles.png` and `fig5_heatmaps.png`

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_figures.py
git commit -m "feat: add token role comparison (Fig 4) and heatmaps (Fig 5)"
```

---

### Task 5: Calibration, Pareto, Scatter, and Summary Table (Figs 6–9)

**Files:**
- Modify: `scripts/generate_figures.py`

**Context:** These four figures complete the set. Fig 6 uses `CalibrationAnalyzer`. Figs 7-9 need merged data from convergence traces + sampling + entropy results.

**Data merge logic:** Convergence traces key questions by `trace["prompt"]`, while sampling and entropy results key by `result["question"]`. Match by finding common question texts (see `scripts/plot_uncertainty_comparison.py:merge_data_sources()` for the existing pattern — replicate that logic here).

**Data files needed:**
- `results/traces_7b_math_expanded.json`, `results/traces_7b_reasoning_expanded_t0.80.json`
- `results/samples_7b_math.json`, `results/samples_7b_reasoning.json`
- `results/entropy_7b_math.json`, `results/entropy_7b_reasoning.json`

**JSON field names (verified):**
- Convergence traces: `trace["prompt"]`, `trace["correct"]`, `trace["summary"]["avg_final_similarity"]`, `trace["summary"]["avg_iterations"]`, `trace["summary"]["avg_convergence_speed"]`
- Samples: `result["question"]`, `result["agreement"]`
- Entropy: `result["question"]`, `result["mean_entropy"]`

- [ ] **Step 1: Add merge helper and fig_reliability() function**

Add this helper function to `scripts/generate_figures.py` (after the style constants, before the figure functions):

```python
# ---------------------------------------------------------------------------
# Data Helpers
# ---------------------------------------------------------------------------
def merge_data_sources(results_dir, conv_file, samp_file, ent_file):
    """Match problems across convergence, sampling, and entropy by question text."""
    conv_data = load_json(os.path.join(results_dir, conv_file))
    samp_data = load_json(os.path.join(results_dir, samp_file))
    ent_data = load_json(os.path.join(results_dir, ent_file))

    conv_by_q = {t["prompt"]: t for t in conv_data["traces"]}
    samp_by_q = {r["question"]: r for r in samp_data["results"]}
    ent_by_q = {r["question"]: r for r in ent_data["results"]}

    common = set(conv_by_q) & set(samp_by_q) & set(ent_by_q)
    matched = []
    for q in sorted(common):
        c, s, e = conv_by_q[q], samp_by_q[q], ent_by_q[q]
        matched.append({
            "question": q,
            "correct": c["correct"],
            "convergence_similarity": c["summary"]["avg_final_similarity"],
            "convergence_iterations": c["summary"]["avg_iterations"],
            "convergence_speed": c["summary"]["avg_convergence_speed"],
            "sampling_agreement": s["agreement"],
            "mean_entropy": e["mean_entropy"],
        })
    return matched
```

Add this figure function:

```python
# ---------------------------------------------------------------------------
# Fig 6: Reliability Diagrams
# ---------------------------------------------------------------------------
def fig_reliability(out_dir, results_dir):
    """Calibration reliability diagrams for best and worst metrics per task."""
    from src.analysis.calibration import CalibrationAnalyzer

    math_traces = load_json(os.path.join(results_dir, "traces_7b_math_expanded.json"))["traces"]
    reas_traces = load_json(os.path.join(results_dir, "traces_7b_reasoning_expanded_t0.80.json"))["traces"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Math panel: show all three metrics
    metrics_to_plot = [
        ("avg_iterations", "Iterations"),
        ("avg_final_similarity", "Similarity"),
        ("avg_convergence_speed", "Speed"),
    ]
    metric_colors = ["#FF9800", "#2196F3", "#4CAF50"]

    for ax, traces, title in [(ax1, math_traces, "Math"), (ax2, reas_traces, "Reasoning")]:
        for (metric, label), color in zip(metrics_to_plot, metric_colors):
            cal = CalibrationAnalyzer(traces, metric=metric)
            bins = cal.reliability_bins(10)
            ece = cal.expected_calibration_error(10)
            if bins:
                confs = [b["confidence"] for b in bins]
                accs = [b["accuracy"] for b in bins]
                ax.plot(confs, accs, "o-", color=color, markersize=4, linewidth=1.5,
                        label=f"{label} (ECE={ece:.2f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
        ax.set_xlabel("Binned Confidence")
        ax.set_ylabel("Observed Accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Calibration Reliability Diagrams", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig6_reliability.png"))
```

- [ ] **Step 2: Add fig_pareto() and fig_scatter() functions**

```python
# ---------------------------------------------------------------------------
# Fig 7: Pareto Frontier
# ---------------------------------------------------------------------------
def fig_pareto(out_dir, results_dir):
    """AUC vs compute cost for all methods, both task types."""
    from src.analysis.uncertainty_comparison import UncertaintyComparison

    fig, ax = plt.subplots(figsize=(9, 6))

    task_configs = [
        ("math", "traces_7b_math_expanded.json",
         "samples_7b_math.json", "entropy_7b_math.json"),
        ("reasoning", "traces_7b_reasoning_expanded_t0.80.json",
         "samples_7b_reasoning.json", "entropy_7b_reasoning.json"),
    ]

    markers = {"convergence_similarity": "o", "convergence_iterations": "s",
               "convergence_speed": "^", "sampling_agreement": "D",
               "mean_entropy": "v"}

    for task_name, conv, samp, ent in task_configs:
        matched = merge_data_sources(results_dir, conv, samp, ent)
        if not matched:
            continue
        comp = UncertaintyComparison(matched)
        points = comp.pareto_data()
        color = COLORS[task_name]

        for p in points:
            marker = markers.get(p["method"], "o")
            ax.scatter(p["cost"], p["auc"], marker=marker, color=color,
                       s=100, zorder=5, edgecolors="white", linewidth=0.5)
            offset_y = 0.015 if p["method"] != "sampling_agreement" else -0.025
            ax.annotate(METHOD_LABELS.get(p["method"], p["method"]),
                        (p["cost"], p["auc"]),
                        textcoords="offset points", xytext=(5, 8 if offset_y > 0 else -12),
                        fontsize=7, color=color)

        # Connect points for this task type
        sorted_pts = sorted(points, key=lambda x: x["cost"])
        ax.plot([p["cost"] for p in sorted_pts],
                [p["auc"] for p in sorted_pts],
                color=color, alpha=0.25, linestyle="--", label=task_name.capitalize())

    ax.axhline(y=0.5, color="gray", alpha=0.2, linestyle=":")
    ax.set_xlabel("Compute Cost (forward pass equivalents)")
    ax.set_ylabel("ROC AUC (correct/incorrect classification)")
    ax.set_title("Uncertainty Quality vs Compute Cost")
    ax.legend(fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.8, 12)
    ax.set_ylim(0.25, 1.0)

    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig7_pareto.png"))


# ---------------------------------------------------------------------------
# Fig 8: Scatter — Convergence vs Sampling
# ---------------------------------------------------------------------------
def fig_scatter(out_dir, results_dir):
    """Scatter: convergence similarity vs sampling agreement, colored by correctness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    task_configs = [
        (ax1, "Math", "traces_7b_math_expanded.json",
         "samples_7b_math.json", "entropy_7b_math.json"),
        (ax2, "Reasoning", "traces_7b_reasoning_expanded_t0.80.json",
         "samples_7b_reasoning.json", "entropy_7b_reasoning.json"),
    ]

    for ax, title, conv, samp, ent in task_configs:
        matched = merge_data_sources(results_dir, conv, samp, ent)
        if not matched:
            continue

        sim = [d["convergence_similarity"] for d in matched]
        agr = [d["sampling_agreement"] for d in matched]
        correct = [d["correct"] for d in matched]
        colors = [COLORS["correct"] if c else COLORS["incorrect"] for c in correct]

        ax.scatter(sim, agr, c=colors, alpha=0.6, s=30, edgecolors="white", linewidth=0.3)
        ax.set_xlabel("Convergence Similarity")
        ax.set_ylabel("Sampling Agreement")
        ax.set_title(f"{title} (n={len(matched)})")

    # Legend
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["correct"],
               markersize=8, label="Correct"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["incorrect"],
               markersize=8, label="Incorrect"),
    ]
    ax2.legend(handles=legend_els, fontsize=9)

    fig.suptitle("Convergence Similarity vs Sampling Agreement", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig8_scatter.png"))
```

- [ ] **Step 3: Add fig_summary_table() function**

```python
# ---------------------------------------------------------------------------
# Fig 9: Summary Table
# ---------------------------------------------------------------------------
def fig_summary_table(out_dir, results_dir):
    """Render the key results as a matplotlib table figure."""
    from src.analysis.uncertainty_comparison import UncertaintyComparison
    from src.analysis.calibration import CalibrationAnalyzer

    # Collect data for both task types
    rows = []
    task_configs = [
        ("Math", "traces_7b_math_expanded.json",
         "samples_7b_math.json", "entropy_7b_math.json"),
        ("Reasoning", "traces_7b_reasoning_expanded_t0.80.json",
         "samples_7b_reasoning.json", "entropy_7b_reasoning.json"),
    ]

    for task_name, conv, samp, ent in task_configs:
        matched = merge_data_sources(results_dir, conv, samp, ent)
        if not matched:
            continue
        comp = UncertaintyComparison(matched)
        aucs = comp.roc_auc_all()
        pareto = {p["method"]: p["cost"] for p in comp.pareto_data()}

        for method in ["sampling_agreement", "convergence_similarity",
                       "convergence_speed", "convergence_iterations", "mean_entropy"]:
            auc = aucs[method]["auc"]
            cost = pareto.get(method, 0)
            rows.append([
                task_name,
                METHOD_LABELS.get(method, method),
                f"{auc:.3f}",
                f"{cost:.2f}",
            ])

    # Render as table
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    col_labels = ["Task", "Method", "AUC", "Cost (FP)"]
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E3F2FD")
        table[0, j].set_text_props(fontweight="bold")

    # Alternate row shading
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#F5F5F5")

    fig.suptitle("Summary: Uncertainty Method Comparison", fontsize=14)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig9_summary_table.png"))
```

- [ ] **Step 4: Update main() to call all remaining figure functions**

Add after the existing calls in `main()`:

```python
    fig_reliability(out_dir, results_dir)
    fig_pareto(out_dir, results_dir)
    fig_scatter(out_dir, results_dir)
    fig_summary_table(out_dir, results_dir)
```

- [ ] **Step 5: Run the script to verify all 9 figures are produced**

Run: `python scripts/generate_figures.py`
Expected: Output shows 9 `Saved:` lines and `9 figures in figures/`

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_figures.py
git commit -m "feat: add calibration, Pareto, scatter, and summary table (Figs 6-9)"
```

---

### Task 6: Write WRITEUP.md

**Files:**
- Create: `WRITEUP.md`

**Context:** This is the main deliverable — a hybrid technical post following the "Problem → Surprise → Deep Dive" narrative structure. All figure references point to `figures/figN_name.png`. Tone: first person plural, concise, accessible to ML practitioners with deeper content for researchers.

- [ ] **Step 1: Write WRITEUP.md**

```markdown
# Convergence as Uncertainty: What Adaptive Inference Reveals About How Transformers Think

## TL;DR

We ran the mid-layers of a transformer repeatedly until its hidden states stabilized, and used convergence speed as a cheap uncertainty signal. For math problems, this predicts correctness (AUC=0.69) at 14% the cost of sampling-based uncertainty. For reasoning problems, the signal **inverts** — the model converges *harder* on wrong answers. This task-dependent behavior reveals something interesting about how transformers process different types of information.

---

## The Problem

Large language models are confident liars. They produce fluent, well-structured text with no built-in indication of whether the content is correct. If you're deploying an LLM where correctness matters — math tutoring, code generation, medical Q&A — you need some way to estimate when the model is likely wrong.

The standard approach is **sampling-based uncertainty**: generate N outputs (typically 8) at elevated temperature and measure agreement. If all samples produce the same answer, the model is probably right. If they diverge, it's uncertain. This works well — we measure AUC=0.87 for math — but costs 8× the compute of a single generation. For latency-sensitive or budget-constrained applications, that's often prohibitive.

We found a cheaper signal hiding in the inference dynamics themselves.

## The Idea

**Adaptive recursive inference** runs a subset of transformer layers — the "circuit block" — repeatedly until the hidden-state representations stabilize. At each iteration, we measure the cosine similarity between the current and previous hidden states. When similarity exceeds a threshold θ, we stop iterating and continue through the remaining layers to produce output.

![Fig 1: Method diagram](figures/fig1_method_diagram.png)
*Fig 1: Adaptive recursive inference. Layers 15–20 of Qwen2.5-7B run repeatedly until hidden-state cosine similarity exceeds threshold θ, or a maximum of 4 iterations is reached.*

The hypothesis is simple: **convergence behavior is informative**. A model that stabilizes quickly might be more confident — and more likely correct — than one that keeps iterating. If so, convergence metrics (final similarity, iteration count, convergence speed) serve as a near-free uncertainty signal: they cost roughly 1.09 forward-pass equivalents, since only 6 of 32 layers iterate and the average problem needs about 2 iterations.

We tested this on **Qwen2.5-7B-Instruct** with layers 15–20 as the circuit block and constrained JSON decoding via `outlines`. The model produces `{"reasoning": "...", "answer": N}` directly — no answer extraction noise. Our evaluation set: 100 math problems (arithmetic, algebra, number theory from GSM8K and MATH) and 100 multi-step reasoning problems with difficulty labels and ground-truth answers.

## The Surprise

For math, convergence works exactly as hypothesized. Problems the model gets right converge faster (higher final similarity, fewer iterations) than problems it gets wrong. The ROC AUC for convergence similarity as a correctness predictor is **0.69** — not spectacular, but usefully above chance and available essentially for free.

For reasoning, the signal **inverts**. The correlation between convergence similarity and correctness is *negative* (r = -0.38 at θ=0.95). The model converges *harder* on problems it gets wrong. It confidently stabilizes on incorrect reasoning chains.

![Fig 2: ROC curves](figures/fig2_roc_curves.png)
*Fig 2: ROC curves for convergence metrics as correctness predictors. Left: math (AUC up to 0.69). Right: reasoning (near-chance or inverted — convergence fails as an uncertainty signal).*

This isn't a failure of the method — it's a finding about how transformers work. When a transformer solves an arithmetic problem, the computation is largely **internal**: the answer depends on manipulating numbers through learned circuits. Convergence of hidden states reflects whether those circuits are reaching a stable answer. Quick convergence → stable computation → likely correct.

When a transformer handles a multi-step reasoning problem, the computation is more **context-dependent**: the answer depends on correctly tracking premises, relationships, and implications from the prompt. The model's internal representations can stabilize (converge) while still being wrong about the external reasoning chain. Worse, the model may converge *more* confidently on familiar-seeming patterns that happen to be incorrect.

## Three Regimes

To understand the convergence dynamics more precisely, we swept the similarity threshold θ from 0.50 to 0.999. The results reveal a clean three-regime structure:

**Safe (θ ≤ 0.70):** The threshold is so low that everything converges in one iteration. Accuracy is at baseline (97% math, 92% reasoning). The model barely iterates — there's no variance in convergence behavior to exploit as a signal.

**Plateau (θ = 0.80–0.95):** The threshold is high enough to require ~2 iterations on average. Accuracy degrades mildly (91% math, 77–82% reasoning). This is the useful operating regime — there's variance in convergence behavior, and the convergence metrics are most discriminative here.

**Cliff (θ ≥ 0.96):** Accuracy collapses sharply. The threshold demands such high similarity that the model hits max iterations (4) on most problems, and the forced iteration disrupts rather than refines the representations. Both task types degrade to ~35% accuracy at θ=0.99.

The critical threshold θ* — where accuracy begins its cliff-edge collapse — is **task-dependent**: θ* ≈ 0.96 for math, θ* ≈ 0.95 for reasoning. Reasoning representations are less robust to forced iteration and destabilize at a lower threshold. This aligns with the hypothesis that reasoning representations are more fragile than arithmetic representations.

![Fig 3: Phase transition](figures/fig3_phase_transition.png)
*Fig 3: Accuracy (left) and mean iterations (right) vs. similarity threshold θ. Shaded regions mark the three regimes. Vertical dashed lines indicate the task-specific critical θ*.*

## Where Uncertainty Lives

Moving from sequence-level to token-level analysis reveals where the model's uncertainty concentrates. We classified each generated token by its role in the constrained JSON output: **structural** (JSON syntax: `{`, `"reasoning":`, `}`), **reasoning** (the content of the reasoning field), and **answer** (the numeric answer tokens).

The results were counterintuitive. Structural tokens — which are deterministic given the JSON schema — converge the **slowest**, not the fastest. The model spends the most iterations stabilizing its representation of tokens that have only one valid option. This suggests the adaptive loop is doing deep representation refinement, not just answer computation.

Answer tokens converge fastest. And uncertainty localizes to **mid-reasoning tokens** (positions 50–150), where the model is working through the core logic of the problem.

For problems the model gets wrong, the convergence profile looks similar to correct problems through the early reasoning tokens but diverges in the middle. The model appears to "know" — at the representation level — that something is going wrong before it commits to the wrong answer.

![Fig 4: Token role comparison](figures/fig4_token_roles.png)
*Fig 4: Mean iterations and final similarity by token role, split by correctness (math, θ=0.80). Structural tokens converge slowest; answer tokens converge fastest.*

![Fig 5: Convergence heatmaps](figures/fig5_heatmaps.png)
*Fig 5: Per-token convergence heatmaps for representative problems. Each cell shows cosine similarity at (position, iteration). Top row: correct problems show uniform high similarity. Bottom row: incorrect problems show disruption in mid-reasoning positions.*

## Is the Signal Calibrated?

A useful uncertainty signal should be **calibrated**: when the signal says "80% confident," the model should be correct about 80% of the time. We binned problems by convergence metric value and measured actual accuracy per bin.

The best-calibrated metric differs by task type. For math, **iteration count** is best-calibrated (ECE = 0.19). For reasoning, **convergence speed** is well-calibrated (ECE = 0.10–0.12) — meaning it reliably tracks how likely the model is to be right, even though the absolute AUC is low.

But final similarity for reasoning is **anti-calibrated** (ECE = 0.61). When the signal says "high confidence," accuracy is actually *lower* than when it says "low confidence." This is the confident-wrong phenomenon from Section 3, now quantified: the model's similarity-based convergence signal actively misleads for reasoning tasks.

![Fig 6: Reliability diagrams](figures/fig6_reliability.png)
*Fig 6: Calibration reliability diagrams. The diagonal represents perfect calibration. Left: math. Right: reasoning, showing well-calibrated speed (ECE=0.10) alongside anti-calibrated similarity (ECE=0.61).*

## The Cost Question

How does convergence-based uncertainty compare to established methods? We benchmarked three approaches head-to-head:

1. **Convergence-based** (~1.09 forward passes): metrics extracted during adaptive inference
2. **Sampling-based** (8.0 forward passes): generate N=8 outputs at temperature=0.7, measure answer agreement
3. **Softmax entropy** (1.0 forward pass): mean per-token entropy from the model's output distribution

| Method | Math AUC | Reasoning AUC | Cost (FP) |
|--------|----------|---------------|-----------|
| Sampling (N=8) | **0.874** | **0.829** | 8.00 |
| Conv. Similarity | 0.690 | 0.475 | 1.09 |
| Conv. Speed | 0.660 | 0.524 | 1.09 |
| Conv. Iterations | 0.625 | 0.560 | 1.12 |
| Softmax Entropy | 0.338 | 0.544 | 1.00 |

Sampling dominates in raw discriminative power — AUC > 0.82 for both task types. But it costs 8× more compute.

For math, convergence similarity achieves **79% of sampling's AUC at 14% of the cost**. On a Pareto frontier of AUC vs. compute, convergence is the best option if your budget is under ~2 forward passes.

For reasoning, no cheap method is competitive with sampling. All convergence signals are near chance (AUC ≤ 0.56), and entropy is barely better (AUC = 0.54).

A striking negative result: **softmax entropy is worse than random for math** (AUC = 0.34). The model's token-level confidence distribution contains almost no information about answer correctness for arithmetic problems. Convergence captures something structural — the stability of internal representations across iterations — that per-token softmax probabilities miss entirely.

![Fig 7: Pareto frontier](figures/fig7_pareto.png)
*Fig 7: AUC vs. compute cost for all uncertainty methods. Convergence methods cluster near the origin — low cost, moderate AUC for math. Sampling dominates but at 8× cost.*

![Fig 8: Scatter plots](figures/fig8_scatter.png)
*Fig 8: Convergence similarity vs. sampling agreement for each problem. Blue = correct, red = incorrect. For math (left), the two signals agree — high convergence + high sampling agreement strongly predicts correctness. For reasoning (right), the relationship breaks down.*

## What This Means

**For practitioners:** If you're running inference on computation-heavy tasks (arithmetic, algebra, structured problems with clear answers) and can't afford 8× sampling, convergence-based uncertainty is a viable alternative. Run adaptive inference with layers in the 40–60% depth range, set threshold θ ≈ 0.80–0.90, and use convergence speed or final similarity as your confidence score. But don't use it for context-heavy reasoning tasks — the signal will actively mislead you.

**For researchers:** The task-dependent behavior of convergence tells us something about transformer representations. Math representations have a property that reasoning representations lack: *correctness correlates with stability*. When a transformer "knows" an arithmetic answer, its internal state settles quickly. When it "knows" a reasoning answer, its internal state may settle just as firmly on a wrong answer. This asymmetry suggests that math and reasoning engage fundamentally different computational modes within the same model.

The **confident-wrong phenomenon** — convergence anti-correlating with correctness for reasoning — is perhaps the most interesting finding. It implies that transformers can reach stable attractor states that are confidently incorrect. The model isn't uncertain and guessing; it's certain and wrong. This has implications for interpretability: internal stability signals may be unreliable precisely when they matter most.

**Limitations:** This study uses a single model family (Qwen2.5-7B), a specific circuit block choice (layers 15–20), and relatively small evaluation sets (100 problems per task type). The three-regime structure and task-dependent θ* should be validated across model families and scales. The confident-wrong phenomenon may interact with model size — larger models might show different convergence dynamics for reasoning.

![Fig 9: Summary table](figures/fig9_summary_table.png)
*Fig 9: Complete summary of uncertainty method comparison across tasks.*

## Methods

**Model:** Qwen2.5-7B-Instruct (32 layers). Circuit block: layers 15–20 (6 layers). Loaded with `device_map="auto"` on NVIDIA L40S (48GB VRAM).

**Adaptive inference:** Cosine similarity halting with `max_iterations=4`. Default threshold θ=0.80 for convergence trace collection. Phase transition sweep: 13 thresholds from θ=0.50 to θ=0.999.

**Datasets:** 100 math problems from GSM8K and MATH (arithmetic, algebra, number theory) and 100 multi-step reasoning problems, each with difficulty labels and step counts.

**Constrained decoding:** `outlines` library for guaranteed-valid JSON: `{"reasoning": "<string>", "answer": <integer>}`. This eliminates answer-extraction noise as a confound.

**Sampling baseline:** N=8 samples per problem at temperature=0.7 with JSON constraint. Uncertainty = fraction of samples agreeing on the majority answer.

**Entropy baseline:** Per-token softmax entropy from raw logits (before JSON constraint is applied to select the token). Aggregated as mean entropy across the generated sequence.

**Hardware:** Data collection on RunPod (NVIDIA L40S). Analysis and figure generation on Mac M1 Max (32GB).

**Code:** All code, data, and figures are available in this repository.
```

- [ ] **Step 2: Verify all figure references are correct**

Run: `grep -o 'figures/fig[0-9]_[a-z_]*\.png' WRITEUP.md | sort`

Expected output (9 unique figure paths):
```
figures/fig1_method_diagram.png
figures/fig2_roc_curves.png
figures/fig3_phase_transition.png
figures/fig4_token_roles.png
figures/fig5_heatmaps.png
figures/fig6_reliability.png
figures/fig7_pareto.png
figures/fig8_scatter.png
figures/fig9_summary_table.png
```

- [ ] **Step 3: Commit**

```bash
git add WRITEUP.md
git commit -m "docs: add writeup — convergence as uncertainty signal"
```

---

### Task 7: Integration and Final Verification

**Files:**
- Modify: `IMPLEMENTATION_PLAN.md`

- [ ] **Step 1: Run generate_figures.py from clean state**

```bash
rm -rf figures/
python scripts/generate_figures.py
```

Expected: All 9 figures generated, 9 `Saved:` lines in output.

- [ ] **Step 2: Verify all 9 figures exist and have reasonable size**

Run: `ls -la figures/`

Expected: 9 PNG files, each between 30KB and 500KB:
- `fig1_method_diagram.png`
- `fig2_roc_curves.png`
- `fig3_phase_transition.png`
- `fig4_token_roles.png`
- `fig5_heatmaps.png`
- `fig6_reliability.png`
- `fig7_pareto.png`
- `fig8_scatter.png`
- `fig9_summary_table.png`

- [ ] **Step 3: Verify WRITEUP.md figure references match generated files**

Run: `for f in $(grep -oP 'figures/fig\d+_[a-z_]+\.png' WRITEUP.md | sort -u); do [ -f "$f" ] && echo "OK: $f" || echo "MISSING: $f"; done`

Expected: All 9 lines show `OK:`

- [ ] **Step 4: Run the statistical summary to verify it works**

Run: `python -c "from src.analysis.statistical_summary import StatisticalSummary; ss = StatisticalSummary(); ss.to_json('results/statistical_summary.json', bootstrap_n=100); print('Summary saved')" `

Expected: `Summary saved` with no errors. Check: `ls -la results/statistical_summary.json` shows a non-empty file.

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/ -v`

Expected: All tests pass (existing + new statistical summary tests).

- [ ] **Step 6: Update IMPLEMENTATION_PLAN.md**

Change Stage 6 status from `Not Started` to `Complete`:

In `IMPLEMENTATION_PLAN.md`, find:
```
**Status:** Not Started
```
(the one under Stage 6) and change to:
```
**Status:** Complete
```

- [ ] **Step 7: Commit everything**

```bash
git add figures/ results/statistical_summary.json IMPLEMENTATION_PLAN.md
git commit -m "Complete Stage 6: writeup, figures, and statistical summary"
```
