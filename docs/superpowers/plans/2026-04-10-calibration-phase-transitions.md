# Stage 4: Calibration and Phase Transitions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine if convergence is a well-calibrated confidence signal and whether there's a sharp phase transition in representation quality at a critical threshold θ*.

**Architecture:** A `CalibrationAnalyzer` class bins traces by convergence metric value and computes accuracy per bin, Expected Calibration Error (ECE), and reliability diagram data. A `scripts/phase_transition_sweep.py` runs fine-grained threshold sweeps on RunPod. A `scripts/plot_calibration.py` produces reliability diagrams and phase transition plots. All analysis classes are testable with synthetic data.

**Tech Stack:** Existing trace JSON files, numpy, scipy (curve_fit for sigmoid), matplotlib

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/analysis/calibration.py` | `CalibrationAnalyzer`: bin by metric, ECE, reliability data |
| Create | `tests/analysis/test_calibration.py` | Tests for CalibrationAnalyzer |
| Create | `scripts/phase_transition_sweep.py` | Run fine-grained threshold sweep on RunPod |
| Create | `scripts/plot_calibration.py` | Reliability diagrams, phase transition plots, ECE |

---

### Task 1: CalibrationAnalyzer

**Files:**
- Create: `src/analysis/calibration.py`
- Create: `tests/analysis/test_calibration.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_calibration.py`:

```python
"""Tests for CalibrationAnalyzer."""
import numpy as np
import pytest

from src.analysis.calibration import CalibrationAnalyzer


def _make_traces_calibrated(n=100, seed=42):
    """Traces where convergence IS calibrated: high-similarity bins are more accurate."""
    rng = np.random.RandomState(seed)
    traces = []
    for i in range(n):
        # Higher similarity → higher probability of being correct
        sim = rng.uniform(0.80, 0.99)
        prob_correct = (sim - 0.80) / (0.99 - 0.80)  # linear 0→1
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
        correct = rng.random() < 0.5  # 50% regardless of similarity
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
        # Well-calibrated data should have low ECE
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
        # Not strictly guaranteed but very likely with these seeds
        assert ece_cal < ece_uncal or ece_cal < 0.3

    def test_reliability_bins_monotonic_for_calibrated(self):
        traces = _make_traces_calibrated(n=300, seed=42)
        analyzer = CalibrationAnalyzer(traces, metric="avg_final_similarity")
        bins = analyzer.reliability_bins(n_bins=5)
        accuracies = [b["accuracy"] for b in bins]
        # For well-calibrated data, accuracy should generally increase with bin
        # Check that last bin > first bin (loose monotonicity)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/analysis/test_calibration.py -v`

Expected: `ModuleNotFoundError: No module named 'src.analysis.calibration'`

- [ ] **Step 3: Implement CalibrationAnalyzer**

Create `src/analysis/calibration.py`:

```python
"""Calibration analysis for convergence-based uncertainty signals.

Bins traces by a convergence metric and measures whether the metric value
is calibrated — i.e., whether higher metric values correspond to higher
accuracy. Computes Expected Calibration Error (ECE) and reliability
diagram data.
"""
import numpy as np


class CalibrationAnalyzer:
    """Analyze calibration of a convergence metric as a confidence signal.

    Args:
        traces: List of trace dicts with "correct" and "summary" fields.
        metric: Which convergence metric to use as the confidence signal.
            One of: "avg_final_similarity", "avg_iterations",
            "avg_convergence_speed", "pct_early_halt".
        higher_is_confident: If True (default), higher metric values mean
            higher confidence. Set False for "avg_iterations" where lower
            means more confident.
    """

    def __init__(self, traces, metric="avg_final_similarity",
                 higher_is_confident=None):
        self.traces = traces
        self.metric = metric
        self._correct = np.array([t["correct"] for t in traces], dtype=float)
        self._values = np.array([
            t["summary"].get(metric, 0) for t in traces
        ], dtype=float)

        # Auto-detect direction: iterations and convergence_speed are inverted
        if higher_is_confident is None:
            higher_is_confident = metric not in ("avg_iterations",)
        self.higher_is_confident = higher_is_confident

        # Normalize values to [0, 1] range for calibration
        vmin, vmax = self._values.min(), self._values.max()
        if vmax > vmin:
            self._confidence = (self._values - vmin) / (vmax - vmin)
        else:
            self._confidence = np.full_like(self._values, 0.5)

        if not self.higher_is_confident:
            self._confidence = 1.0 - self._confidence

    def reliability_bins(self, n_bins=10):
        """Bin traces by confidence and compute accuracy per bin.

        Returns:
            List of dicts with bin_center, accuracy, confidence, count.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (self._confidence >= lo) & (self._confidence < hi)
            if i == n_bins - 1:  # include right edge in last bin
                mask = mask | (self._confidence == hi)
            count = mask.sum()
            if count == 0:
                continue
            accuracy = self._correct[mask].mean()
            confidence = self._confidence[mask].mean()
            bins.append({
                "bin_center": float((lo + hi) / 2),
                "accuracy": float(accuracy),
                "confidence": float(confidence),
                "count": int(count),
            })
        return bins

    def expected_calibration_error(self, n_bins=10):
        """Compute Expected Calibration Error (ECE).

        ECE = sum over bins of (count/total) * |accuracy - confidence|
        """
        bins = self.reliability_bins(n_bins)
        total = sum(b["count"] for b in bins)
        if total == 0:
            return 0.0
        ece = sum(
            (b["count"] / total) * abs(b["accuracy"] - b["confidence"])
            for b in bins
        )
        return float(ece)

    def summary(self, n_bins=10):
        """Produce a complete calibration summary."""
        return {
            "metric": self.metric,
            "n": len(self.traces),
            "ece": self.expected_calibration_error(n_bins),
            "bins": self.reliability_bins(n_bins),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/analysis/test_calibration.py -v`

Expected: All 6 tests pass.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/analysis/calibration.py tests/analysis/test_calibration.py
git commit -m "Add CalibrationAnalyzer for convergence-correctness calibration and ECE"
```

---

### Task 2: Phase Transition Sweep Script

**Files:**
- Create: `scripts/phase_transition_sweep.py`

This script runs `collect_traces.py` at many thresholds and extracts summary statistics for each, producing a JSON file suitable for plotting.

- [ ] **Step 1: Create the sweep script**

Create `scripts/phase_transition_sweep.py`:

```python
"""Run fine-grained threshold sweep for phase transition analysis.

Iterates over thresholds, runs collect_traces.py for each, and produces
a summary JSON with accuracy and mean iterations per threshold.

This script is designed for RunPod — it loads the model once and runs
all thresholds in sequence.

Usage:
    python scripts/phase_transition_sweep.py \
        --model Qwen/Qwen2.5-7B \
        --data data/math_probe_expanded.json \
        --output results/phase_sweep_math.json \
        --thresholds 0.50 0.60 0.70 0.80 0.85 0.90 0.92 0.94 0.95 0.96 0.97 0.98 0.99 \
        --max-iters 4 --block-i 15 --block-j 20
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference.convergence_tracer import ConvergenceTracer
from src.inference.constrained import build_json_processor
from src.evaluation.math_eval import score_answer


PROMPT_TEMPLATE = (
    'Respond with JSON: {{"reasoning": "<your work>", "answer": <number>}}\n\n'
    'Question: {question}\n'
)


def extract_answer(generated):
    try:
        obj = json.loads(generated)
        return float(obj["answer"])
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def run_at_threshold(model, tokenizer, json_processor, probes,
                     threshold, max_iters, block_i, block_j):
    """Run all probes at a given threshold, return summary dict."""
    tracer = ConvergenceTracer(
        model, block_i, block_j,
        threshold=threshold, max_iterations=max_iters,
    )

    correct = 0
    total_score = 0.0
    total_iters = 0.0
    total_tokens = 0

    for probe in probes:
        prompt = PROMPT_TEMPLATE.format(question=probe["question"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text=probe["question"],
            score=0.0,
            max_new_tokens=256,
            logits_processor=json_processor,
        )

        predicted = extract_answer(trace.generated)
        score = score_answer(predicted, probe["answer"])
        if score > 0.99:
            correct += 1
        total_score += score

        summary = trace.summary()
        total_iters += summary["avg_iterations"]
        total_tokens += summary["total_tokens"]

    n = len(probes)
    return {
        "threshold": threshold,
        "accuracy": correct / n,
        "num_correct": correct,
        "mean_score": total_score / n,
        "mean_avg_iterations": total_iters / n,
        "mean_tokens": total_tokens / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase transition threshold sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data", default="data/math_probe_expanded.json")
    parser.add_argument("--output", default="results/phase_sweep_math.json")
    parser.add_argument("--block-i", type=int, default=15)
    parser.add_argument("--block-j", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.50, 0.60, 0.70, 0.80, 0.85,
                                 0.90, 0.92, 0.94, 0.95, 0.96,
                                 0.97, 0.98, 0.99])
    parser.add_argument("--max-probes", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )

    with open(args.data) as f:
        probes = json.load(f)
    if args.max_probes:
        probes = probes[:args.max_probes]
    print(f"Loaded {len(probes)} probes")

    print("Building JSON logits processor...")
    json_processor = build_json_processor(model, tokenizer)

    results = []
    for threshold in sorted(args.thresholds):
        t0 = time.time()
        print(f"\n=== Threshold={threshold:.2f} ===")
        result = run_at_threshold(
            model, tokenizer, json_processor, probes,
            threshold, args.max_iters, args.block_i, args.block_j,
        )
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        results.append(result)
        print(f"  accuracy={result['accuracy']:.3f} "
              f"mean_iters={result['mean_avg_iterations']:.2f} "
              f"[{elapsed:.1f}s]")

    output = {
        "model": args.model,
        "data": args.data,
        "block": [args.block_i, args.block_j],
        "max_iterations": args.max_iters,
        "num_probes": len(probes),
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Print summary table
    print(f"\n{'Threshold':>10s}  {'Accuracy':>8s}  {'Iters':>6s}  {'Time':>6s}")
    for r in results:
        print(f"  {r['threshold']:>8.2f}  {r['accuracy']:>8.3f}  "
              f"{r['mean_avg_iterations']:>6.2f}  {r['elapsed_s']:>6.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/phase_transition_sweep.py
git commit -m "Add fine-grained phase transition threshold sweep script"
```

---

### Task 3: Calibration and Phase Transition Visualization Script

**Files:**
- Create: `scripts/plot_calibration.py`

- [ ] **Step 1: Create the visualization script**

Create `scripts/plot_calibration.py`:

```python
"""Plot calibration reliability diagrams and phase transition curves.

Usage:
    # Calibration from trace files
    python scripts/plot_calibration.py calibration \
        results/traces_7b_math_expanded.json \
        --output-dir plots/calibration

    # Phase transition from sweep results
    python scripts/plot_calibration.py phase-transition \
        results/phase_sweep_math.json results/phase_sweep_reasoning.json \
        --output-dir plots/calibration
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.analysis.calibration import CalibrationAnalyzer


def plot_reliability_diagram(trace_files, output_path, n_bins=10):
    """Plot reliability diagrams for each trace file and metric."""
    metrics = ["avg_final_similarity", "avg_iterations", "avg_convergence_speed"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for ax, metric in zip(axes, metrics):
        for trace_file in trace_files:
            with open(trace_file) as f:
                data = json.load(f)
            traces = data["traces"]
            threshold = data.get("threshold", "?")
            label = f"θ={threshold}"

            analyzer = CalibrationAnalyzer(traces, metric=metric)
            bins = analyzer.reliability_bins(n_bins)
            ece = analyzer.expected_calibration_error(n_bins)

            confidences = [b["confidence"] for b in bins]
            accuracies = [b["accuracy"] for b in bins]
            ax.plot(confidences, accuracies, "o-", label=f"{label} (ECE={ece:.3f})",
                    markersize=4)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(metric.replace("avg_", "").replace("_", " ").title())
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Reliability Diagrams: Convergence Metrics as Confidence")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reliability diagram: {output_path}")


def plot_ece_comparison(trace_files, output_path, n_bins=10):
    """Bar chart of ECE per metric and threshold."""
    metrics = ["avg_final_similarity", "avg_iterations", "avg_convergence_speed"]

    data_points = []
    for trace_file in trace_files:
        with open(trace_file) as f:
            data = json.load(f)
        threshold = data.get("threshold", 0)
        traces = data["traces"]
        eces = {}
        for metric in metrics:
            analyzer = CalibrationAnalyzer(traces, metric=metric)
            eces[metric] = analyzer.expected_calibration_error(n_bins)
        data_points.append({"threshold": threshold, "eces": eces})

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(data_points))
    width = 0.25

    for i, metric in enumerate(metrics):
        vals = [dp["eces"][metric] for dp in data_points]
        short_name = metric.replace("avg_", "").replace("_", " ")
        ax.bar(x + i * width, vals, width, label=short_name)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"θ={dp['threshold']}" for dp in data_points])
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Expected Calibration Error by Metric and Threshold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ECE comparison: {output_path}")


def plot_phase_transition(sweep_files, output_path):
    """Plot accuracy and iterations vs threshold for phase transition analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4488cc", "#cc4444", "#44aa44", "#aa44aa"]

    for idx, sweep_file in enumerate(sweep_files):
        with open(sweep_file) as f:
            data = json.load(f)
        results = data["results"]
        label = os.path.basename(sweep_file).replace("phase_sweep_", "").replace(".json", "")

        thresholds = [r["threshold"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        iterations = [r["mean_avg_iterations"] for r in results]

        color = colors[idx % len(colors)]
        axes[0].plot(thresholds, accuracies, "o-", color=color, label=label, markersize=4)
        axes[1].plot(thresholds, iterations, "o-", color=color, label=label, markersize=4)

    axes[0].set_xlabel("Threshold (θ)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs Threshold")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("Threshold (θ)")
    axes[1].set_ylabel("Mean Iterations")
    axes[1].set_title("Mean Iterations vs Threshold")
    axes[1].legend()

    fig.suptitle("Phase Transition Analysis")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved phase transition plot: {output_path}")


def cmd_calibration(args):
    os.makedirs(args.output_dir, exist_ok=True)
    plot_reliability_diagram(
        args.trace_files,
        os.path.join(args.output_dir, "reliability_diagram.png"),
    )
    plot_ece_comparison(
        args.trace_files,
        os.path.join(args.output_dir, "ece_comparison.png"),
    )

    # Print ECE table
    metrics = ["avg_final_similarity", "avg_iterations", "avg_convergence_speed"]
    print(f"\n{'File':<50s}  ", end="")
    for m in metrics:
        print(f"{m.replace('avg_', ''):>15s}  ", end="")
    print()
    for trace_file in args.trace_files:
        with open(trace_file) as f:
            data = json.load(f)
        traces = data["traces"]
        threshold = data.get("threshold", "?")
        print(f"{'θ=' + str(threshold):<50s}  ", end="")
        for metric in metrics:
            analyzer = CalibrationAnalyzer(traces, metric=metric)
            ece = analyzer.expected_calibration_error()
            print(f"{ece:>15.4f}  ", end="")
        print()


def cmd_phase_transition(args):
    os.makedirs(args.output_dir, exist_ok=True)
    plot_phase_transition(
        args.sweep_files,
        os.path.join(args.output_dir, "phase_transition.png"),
    )


def main():
    parser = argparse.ArgumentParser(description="Calibration and phase transition plots")
    subparsers = parser.add_subparsers(dest="command")

    cal_parser = subparsers.add_parser("calibration")
    cal_parser.add_argument("trace_files", nargs="+")
    cal_parser.add_argument("--output-dir", default="plots/calibration")

    pt_parser = subparsers.add_parser("phase-transition")
    pt_parser.add_argument("sweep_files", nargs="+")
    pt_parser.add_argument("--output-dir", default="plots/calibration")

    args = parser.parse_args()
    if args.command == "calibration":
        cmd_calibration(args)
    elif args.command == "phase-transition":
        cmd_phase_transition(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it loads**

Run: `uv run python scripts/plot_calibration.py --help`

Expected: Help text with subcommands.

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_calibration.py
git commit -m "Add calibration and phase transition visualization script"
```

---

### Task 4: Run Calibration Analysis on Existing Traces

**Files:**
- No new code — runs the analysis on existing data.

- [ ] **Step 1: Run calibration on math traces across thresholds**

```bash
uv run python scripts/plot_calibration.py calibration \
    results/traces_7b_math_expanded.json \
    results/traces_7b_math_t0.90.json \
    results/traces_7b_math_t0.95.json \
    results/traces_7b_math_t0.98.json \
    --output-dir plots/calibration
```

- [ ] **Step 2: Run calibration on reasoning traces**

```bash
uv run python scripts/plot_calibration.py calibration \
    results/traces_7b_reasoning_expanded_t0.80.json \
    results/traces_7b_reasoning_t0.90.json \
    results/traces_7b_reasoning_t0.95.json \
    results/traces_7b_reasoning_t0.98.json \
    --output-dir plots/calibration_reasoning
```

- [ ] **Step 3: Commit plots**

```bash
git add plots/calibration/ plots/calibration_reasoning/
git commit -m "Generate calibration reliability diagrams and ECE comparisons"
```

---

### Task 5: Run Phase Transition Sweep on RunPod

**Files:**
- No new code — runs sweep on RunPod, copies results locally.

- [ ] **Step 1: Sync code to RunPod**

```bash
rsync -avz --no-owner --no-group --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    -e "ssh -p 17599 -i ~/.ssh/id_rsa" \
    /Users/nathan/Projects/ndouglas/adaptive-recursive-inference-experiment/ \
    root@91.199.227.82:/workspace/ari/
```

- [ ] **Step 2: Run math sweep**

```bash
ssh -p 17599 -i ~/.ssh/id_rsa root@91.199.227.82 \
    "cd /workspace/ari && HF_HOME=/workspace/huggingface \
     python3 -u scripts/phase_transition_sweep.py \
       --model Qwen/Qwen2.5-7B \
       --data data/math_probe_expanded.json \
       --output /workspace/phase_sweep_math.json \
       --max-iters 4 --block-i 15 --block-j 20 \
       --thresholds 0.50 0.60 0.70 0.80 0.85 0.90 0.92 0.94 0.95 0.96 0.97 0.98 0.99"
```

- [ ] **Step 3: Run reasoning sweep**

```bash
ssh -p 17599 -i ~/.ssh/id_rsa root@91.199.227.82 \
    "cd /workspace/ari && HF_HOME=/workspace/huggingface \
     python3 -u scripts/phase_transition_sweep.py \
       --model Qwen/Qwen2.5-7B \
       --data data/reasoning_probe_expanded.json \
       --output /workspace/phase_sweep_reasoning.json \
       --max-iters 4 --block-i 15 --block-j 20 \
       --thresholds 0.50 0.60 0.70 0.80 0.85 0.90 0.92 0.94 0.95 0.96 0.97 0.98 0.99"
```

- [ ] **Step 4: Copy results locally**

```bash
scp -P 17599 -i ~/.ssh/id_rsa root@91.199.227.82:/workspace/phase_sweep_math.json results/
scp -P 17599 -i ~/.ssh/id_rsa root@91.199.227.82:/workspace/phase_sweep_reasoning.json results/
```

- [ ] **Step 5: Generate phase transition plots**

```bash
uv run python scripts/plot_calibration.py phase-transition \
    results/phase_sweep_math.json results/phase_sweep_reasoning.json \
    --output-dir plots/calibration
```

- [ ] **Step 6: Commit**

```bash
git add results/phase_sweep_*.json plots/calibration/
git commit -m "Add phase transition sweep results and plots (13 thresholds, math+reasoning)"
```

---

### Task 6: Update Implementation Plan

**Files:**
- Modify: `IMPLEMENTATION_PLAN.md`

- [ ] **Step 1: Update Stage 4 status**

Change Stage 4 status from `Not Started` to `Complete` in `IMPLEMENTATION_PLAN.md`.

- [ ] **Step 2: Commit**

```bash
git add -f IMPLEMENTATION_PLAN.md
git commit -m "Mark Stage 4 (calibration and phase transitions) complete"
```
