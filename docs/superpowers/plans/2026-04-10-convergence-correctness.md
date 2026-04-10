# Stage 2: Convergence–Correctness Correlation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether convergence rate during adaptive inference predicts output correctness, producing ROC curves and correlation statistics stratified by task type.

**Architecture:** A `collect_traces.py` script runs probes through `ConvergenceTracer` on RunPod, saving structured JSON trace files. A `ConvergenceAnalyzer` class computes correlations, ROC curves, and bootstrap CIs from those traces offline. A `analyze_traces.py` script orchestrates the analysis and produces summary tables. The analysis code is fully testable with synthetic data — no model required.

**Tech Stack:** Existing `ConvergenceTracer`, `ConvergenceTrace`, `score_answer()`, scipy.stats, numpy, sklearn.metrics (for ROC AUC)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/expand_datasets.py` | Pull GSM8K samples into local probe format |
| Create | `data/math_probe_expanded.json` | 100 math probes in existing format |
| Create | `scripts/collect_traces.py` | Run probes through ConvergenceTracer, save trace JSON |
| Create | `src/analysis/convergence_stats.py` | `ConvergenceAnalyzer`: correlations, ROC, bootstrap CIs |
| Create | `scripts/analyze_traces.py` | Load traces, run analysis, print tables |
| Create | `tests/analysis/__init__.py` | Package marker |
| Create | `tests/analysis/test_convergence_stats.py` | Tests for ConvergenceAnalyzer |
| Modify | `pyproject.toml` | Add scikit-learn dependency |

---

### Task 1: Add scikit-learn Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add scikit-learn**

```bash
uv add scikit-learn
```

- [ ] **Step 2: Verify import works**

```bash
uv run python -c "from sklearn.metrics import roc_auc_score; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Add scikit-learn for ROC AUC analysis"
```

---

### Task 2: Expand Math Dataset

**Files:**
- Create: `scripts/expand_datasets.py`
- Create: `data/math_probe_expanded.json`

We need ~100 math probes with known answers. GSM8K is the standard source but requires HuggingFace datasets. Simpler approach: generate deterministic arithmetic/algebra probes programmatically with known answers, covering the same categories as our existing probes but at scale.

- [ ] **Step 1: Write the dataset expansion script**

Create `scripts/expand_datasets.py`:

```python
"""Generate expanded math probe datasets for statistical analysis.

Produces deterministic math problems with known answers across categories:
- basic_arithmetic: addition, multiplication, division
- multi_step: chained operations
- powers_roots: squares, cubes, square roots
- modular: modular arithmetic
- combinatorics: choose(n, k)
- number_theory: GCD, LCM, primes

Target: 100 problems with difficulty labels (easy/medium/hard).
"""
import json
import math
import random


def generate_math_probes(seed=42, target=100):
    random.seed(seed)
    probes = []

    # Basic arithmetic (20 problems)
    for _ in range(10):
        a, b = random.randint(10, 999), random.randint(10, 999)
        probes.append({
            "question": f"What is {a} + {b}?",
            "answer": a + b,
            "category": "basic_arithmetic",
            "difficulty": "easy",
        })
    for _ in range(10):
        a, b = random.randint(10, 99), random.randint(10, 99)
        probes.append({
            "question": f"What is {a} * {b}?",
            "answer": a * b,
            "category": "basic_arithmetic",
            "difficulty": "easy",
        })

    # Multi-step (20 problems)
    for _ in range(10):
        a, b, c = random.randint(2, 20), random.randint(2, 20), random.randint(2, 20)
        probes.append({
            "question": f"What is ({a} + {b}) * {c}?",
            "answer": (a + b) * c,
            "category": "multi_step",
            "difficulty": "medium",
        })
    for _ in range(10):
        a, b, c, d = (random.randint(2, 15) for _ in range(4))
        probes.append({
            "question": f"What is ({a} * {b}) + ({c} * {d})?",
            "answer": (a * b) + (c * d),
            "category": "multi_step",
            "difficulty": "medium",
        })

    # Powers and roots (20 problems)
    for _ in range(10):
        base = random.randint(2, 30)
        probes.append({
            "question": f"What is {base} squared?",
            "answer": base ** 2,
            "category": "powers_roots",
            "difficulty": "easy",
        })
    for _ in range(10):
        root = random.randint(2, 20)
        square = root ** 2
        probes.append({
            "question": f"What is the square root of {square}?",
            "answer": root,
            "category": "powers_roots",
            "difficulty": "medium",
        })

    # Modular arithmetic (15 problems)
    for _ in range(15):
        base = random.randint(5, 50)
        exp = random.randint(2, 4)
        mod = random.randint(3, 17)
        probes.append({
            "question": f"What is {base}^{exp} mod {mod}?",
            "answer": pow(base, exp, mod),
            "category": "modular",
            "difficulty": "hard",
        })

    # Combinatorics (10 problems)
    for _ in range(10):
        n = random.randint(4, 12)
        k = random.randint(2, min(4, n))
        probes.append({
            "question": f"How many ways can you choose {k} items from {n}?",
            "answer": math.comb(n, k),
            "category": "combinatorics",
            "difficulty": "medium",
        })

    # Number theory (15 problems)
    for _ in range(8):
        a, b = random.randint(10, 200), random.randint(10, 200)
        probes.append({
            "question": f"What is the GCD of {a} and {b}?",
            "answer": math.gcd(a, b),
            "category": "number_theory",
            "difficulty": "medium",
        })
    for _ in range(7):
        a, b = random.randint(5, 50), random.randint(5, 50)
        lcm = (a * b) // math.gcd(a, b)
        probes.append({
            "question": f"What is the LCM of {a} and {b}?",
            "answer": lcm,
            "category": "number_theory",
            "difficulty": "hard",
        })

    random.shuffle(probes)
    return probes[:target]


if __name__ == "__main__":
    probes = generate_math_probes()
    with open("data/math_probe_expanded.json", "w") as f:
        json.dump(probes, f, indent=2)
    print(f"Generated {len(probes)} math probes")

    # Print category breakdown
    from collections import Counter
    cats = Counter(p["category"] for p in probes)
    diffs = Counter(p["difficulty"] for p in probes)
    print(f"Categories: {dict(cats)}")
    print(f"Difficulties: {dict(diffs)}")
```

- [ ] **Step 2: Run it**

```bash
uv run python scripts/expand_datasets.py
```

Expected: `Generated 100 math probes` with category and difficulty breakdown.

- [ ] **Step 3: Verify the generated file**

```bash
uv run python -c "import json; d=json.load(open('data/math_probe_expanded.json')); print(len(d)); print(d[0])"
```

Expected: 100 probes, each with `question`, `answer`, `category`, `difficulty` fields.

- [ ] **Step 4: Commit**

```bash
git add scripts/expand_datasets.py data/math_probe_expanded.json
git commit -m "Generate expanded math probe dataset (100 problems, 6 categories)"
```

---

### Task 3: Trace Collection Script

**Files:**
- Create: `scripts/collect_traces.py`

This script runs probes through `ConvergenceTracer` with constrained JSON decoding and saves structured trace files. Designed to run on RunPod with the 7B model.

- [ ] **Step 1: Create the trace collection script**

Create `scripts/collect_traces.py`:

```python
"""Collect convergence traces from adaptive inference.

Runs math probes through ConvergenceTracer with constrained JSON decoding,
recording full per-token convergence data + correctness labels.

Usage:
    python scripts/collect_traces.py --model Qwen/Qwen2.5-7B \
        --data data/math_probe_expanded.json \
        --output results/traces_7b_math.json \
        --threshold 0.80 --max-iters 4 \
        --block-i 15 --block-j 20
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
    """Extract numeric answer from constrained JSON output."""
    try:
        obj = json.loads(generated)
        return float(obj["answer"])
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def collect_traces(tracer, tokenizer, probes, json_processor, verbose=True):
    """Run probes through tracer, scoring each and returning trace dicts."""
    all_traces = []
    for i, probe in enumerate(probes):
        prompt = PROMPT_TEMPLATE.format(question=probe["question"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            tracer.loop.model.device
        )

        # Generate with tracing (score=0 placeholder, updated after)
        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text=probe["question"],
            score=0.0,
            max_new_tokens=256,
            logits_processor=json_processor,
        )

        # Score the output
        predicted = extract_answer(trace.generated)
        score = score_answer(predicted, probe["answer"])
        trace.score = score  # update placeholder

        trace_dict = trace.to_dict()
        trace_dict["expected"] = probe["answer"]
        trace_dict["predicted"] = predicted
        trace_dict["correct"] = score > 0.99
        if "category" in probe:
            trace_dict["category"] = probe["category"]
        if "difficulty" in probe:
            trace_dict["difficulty"] = probe["difficulty"]

        all_traces.append(trace_dict)

        if verbose:
            summary = trace.summary()
            status = "OK" if score > 0.99 else f"WRONG ({predicted})"
            print(
                f"  [{i+1}/{len(probes)}] {probe['question'][:50]:50s} "
                f"score={score:.2f} iters={summary['avg_iterations']:.1f} "
                f"{status}"
            )

    return all_traces


def main():
    parser = argparse.ArgumentParser(description="Collect convergence traces")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data", default="data/math_probe_expanded.json")
    parser.add_argument("--output", default="results/traces_7b_math.json")
    parser.add_argument("--block-i", type=int, default=15)
    parser.add_argument("--block-j", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--max-probes", type=int, default=None,
                        help="Limit number of probes (for testing)")
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
    print(f"Loaded {len(probes)} probes from {args.data}")

    print("Building JSON logits processor...")
    json_processor = build_json_processor(model, tokenizer)

    tracer = ConvergenceTracer(
        model, args.block_i, args.block_j,
        threshold=args.threshold, max_iterations=args.max_iters,
    )

    print(f"\nCollecting traces (threshold={args.threshold}, "
          f"max_iters={args.max_iters}, block=({args.block_i},{args.block_j}))...\n")
    t0 = time.time()
    traces = collect_traces(tracer, tokenizer, probes, json_processor)
    elapsed = time.time() - t0

    # Summary
    scores = [t["score"] for t in traces]
    correct = sum(1 for t in traces if t["correct"])
    avg_iters = sum(
        t["summary"]["avg_iterations"] for t in traces
    ) / len(traces)

    print(f"\n=== Summary ===")
    print(f"  Probes: {len(traces)}")
    print(f"  Correct: {correct}/{len(traces)} ({correct/len(traces)*100:.1f}%)")
    print(f"  Mean score: {sum(scores)/len(scores):.4f}")
    print(f"  Mean avg_iterations: {avg_iters:.2f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    output = {
        "model": args.model,
        "block": [args.block_i, args.block_j],
        "threshold": args.threshold,
        "max_iterations": args.max_iters,
        "data_source": args.data,
        "num_probes": len(traces),
        "num_correct": correct,
        "mean_score": sum(scores) / len(scores),
        "mean_avg_iterations": avg_iters,
        "elapsed_s": round(elapsed, 1),
        "traces": traces,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/collect_traces.py
git commit -m "Add trace collection script for convergence-correctness analysis"
```

---

### Task 4: ConvergenceAnalyzer — Core Statistics

**Files:**
- Create: `src/analysis/convergence_stats.py`
- Create: `tests/analysis/__init__.py`
- Create: `tests/analysis/test_convergence_stats.py`

- [ ] **Step 1: Write tests for the analyzer**

Create `tests/analysis/__init__.py` — empty file.

Create `tests/analysis/test_convergence_stats.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/analysis/test_convergence_stats.py -v`

Expected: `ModuleNotFoundError: No module named 'src.analysis.convergence_stats'`

- [ ] **Step 3: Implement ConvergenceAnalyzer**

Create `src/analysis/convergence_stats.py`:

```python
"""Statistical analysis of convergence traces.

ConvergenceAnalyzer takes a list of trace dicts (from collect_traces.py output)
and computes correlations, ROC AUC, and bootstrap confidence intervals to
answer: does convergence predict correctness?
"""
import numpy as np
from scipy.stats import pearsonr, pointbiserialr
from sklearn.metrics import roc_auc_score


CONVERGENCE_METRICS = [
    "avg_iterations",
    "avg_final_similarity",
    "avg_convergence_speed",
    "pct_early_halt",
]


class ConvergenceAnalyzer:
    """Analyze convergence-correctness relationships from trace data.

    Args:
        traces: List of trace dicts, each with "score", "correct", and
            "summary" containing convergence metrics.
    """

    def __init__(self, traces):
        self.traces = traces
        self._scores = np.array([t["score"] for t in traces])
        self._correct = np.array([t["correct"] for t in traces], dtype=float)
        self._metrics = {}
        for key in CONVERGENCE_METRICS:
            vals = [t["summary"].get(key) for t in traces]
            if all(v is not None for v in vals):
                self._metrics[key] = np.array(vals)

    def compute_correlations(self, bootstrap_n=0):
        """Compute correlation between each convergence metric and score.

        Args:
            bootstrap_n: If > 0, compute bootstrap 95% CIs with this many samples.

        Returns:
            Dict mapping metric name -> {"r": float, "p": float, "ci_95": [lo, hi]}.
        """
        results = {}
        for key, values in self._metrics.items():
            r, p = pearsonr(values, self._scores)
            entry = {"r": float(r), "p": float(p)}

            if bootstrap_n > 0:
                boot_rs = []
                rng = np.random.RandomState(42)
                n = len(values)
                for _ in range(bootstrap_n):
                    idx = rng.randint(0, n, size=n)
                    boot_vals = values[idx]
                    boot_scores = self._scores[idx]
                    if np.std(boot_vals) > 0 and np.std(boot_scores) > 0:
                        br, _ = pearsonr(boot_vals, boot_scores)
                        boot_rs.append(br)
                if boot_rs:
                    entry["ci_95"] = [
                        float(np.percentile(boot_rs, 2.5)),
                        float(np.percentile(boot_rs, 97.5)),
                    ]
                else:
                    entry["ci_95"] = [float(r), float(r)]

            results[key] = entry
        return results

    def compute_roc_auc(self):
        """Compute ROC AUC for each metric as a binary classifier of correct/incorrect.

        For metrics where higher = more correct (similarity, speed, early_halt),
        uses the metric directly. For iterations (higher = less correct), inverts.

        Returns:
            Dict mapping metric name -> {"auc": float}.
        """
        results = {}
        # Metrics where higher value means MORE likely correct
        positive_direction = {"avg_final_similarity", "avg_convergence_speed", "pct_early_halt"}

        for key, values in self._metrics.items():
            if len(np.unique(self._correct)) < 2:
                results[key] = {"auc": 0.5}
                continue

            if key in positive_direction:
                scores = values
            else:
                # Invert: for iterations, lower = better, so negate
                scores = -values

            try:
                auc = roc_auc_score(self._correct, scores)
                results[key] = {"auc": float(auc)}
            except ValueError:
                results[key] = {"auc": 0.5}

        return results

    def stratified_correlations(self, group_key="category"):
        """Compute correlations stratified by a grouping field.

        Args:
            group_key: Field name in trace dict to group by (e.g., "category", "difficulty").

        Returns:
            Dict mapping group_value -> {"n": int, metric: {"r": float, "p": float}}.
        """
        groups = {}
        for t in self.traces:
            g = t.get(group_key, "unknown")
            groups.setdefault(g, []).append(t)

        results = {}
        for group_name, group_traces in sorted(groups.items()):
            if len(group_traces) < 5:
                continue
            sub = ConvergenceAnalyzer(group_traces)
            corr = sub.compute_correlations()
            corr["n"] = len(group_traces)
            results[group_name] = corr
        return results

    def summary_table(self):
        """Produce a complete summary: overall + stratified correlations and ROC."""
        return {
            "overall": {
                "n": len(self.traces),
                "correlations": self.compute_correlations(bootstrap_n=1000),
                "roc_auc": self.compute_roc_auc(),
                "accuracy": float(np.mean(self._correct)),
                "mean_score": float(np.mean(self._scores)),
            },
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/analysis/test_convergence_stats.py -v`

Expected: All 7 tests pass.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`

Expected: All 27 tests pass (20 existing + 7 new).

- [ ] **Step 6: Commit**

```bash
git add src/analysis/convergence_stats.py tests/analysis/__init__.py tests/analysis/test_convergence_stats.py
git commit -m "Add ConvergenceAnalyzer for convergence-correctness correlation analysis"
```

---

### Task 5: Analysis Script

**Files:**
- Create: `scripts/analyze_traces.py`

- [ ] **Step 1: Create the analysis script**

Create `scripts/analyze_traces.py`:

```python
"""Analyze convergence traces for correctness correlation.

Loads trace JSON from collect_traces.py and runs statistical analysis:
correlations, ROC AUC, bootstrap CIs, stratified by category/difficulty.

Usage:
    python scripts/analyze_traces.py results/traces_7b_math.json
    python scripts/analyze_traces.py results/traces_7b_math.json --group-by difficulty
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.analysis.convergence_stats import ConvergenceAnalyzer


def print_correlations(corr, label=""):
    """Pretty-print correlation results."""
    if label:
        print(f"\n=== {label} ===")
    print(f"  {'Metric':<25s}  {'r':>8s}  {'p':>10s}  {'CI 95%':>20s}")
    for metric, stats in corr.items():
        if metric == "n":
            continue
        ci = stats.get("ci_95")
        ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        print(f"  {metric:<25s}  {stats['r']:+8.4f}  {stats['p']:10.4f}  {ci_str:>20s}")


def print_roc(roc, label=""):
    """Pretty-print ROC AUC results."""
    if label:
        print(f"\n=== {label} ===")
    print(f"  {'Metric':<25s}  {'AUC':>8s}")
    for metric, stats in roc.items():
        print(f"  {metric:<25s}  {stats['auc']:8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze convergence traces")
    parser.add_argument("trace_file", help="Path to traces JSON from collect_traces.py")
    parser.add_argument("--group-by", default="category",
                        help="Field to stratify by (category, difficulty)")
    parser.add_argument("--output", default=None,
                        help="Save analysis results to JSON")
    args = parser.parse_args()

    with open(args.trace_file) as f:
        data = json.load(f)

    traces = data["traces"]
    print(f"Loaded {len(traces)} traces from {args.trace_file}")
    print(f"  Model: {data.get('model', 'unknown')}")
    print(f"  Threshold: {data.get('threshold', '?')}")
    print(f"  Accuracy: {data.get('num_correct', '?')}/{data.get('num_probes', '?')}")

    analyzer = ConvergenceAnalyzer(traces)

    # Overall correlations with bootstrap CIs
    corr = analyzer.compute_correlations(bootstrap_n=1000)
    print_correlations(corr, "Correlations (overall)")

    # ROC AUC
    roc = analyzer.compute_roc_auc()
    print_roc(roc, "ROC AUC (overall)")

    # Stratified
    strat = analyzer.stratified_correlations(group_key=args.group_by)
    if strat:
        print(f"\n=== Stratified by {args.group_by} ===")
        for group, group_corr in strat.items():
            n = group_corr.pop("n", "?")
            print(f"\n  --- {group} (n={n}) ---")
            for metric, stats in group_corr.items():
                print(f"    {metric:<25s}  r={stats['r']:+.4f}  p={stats['p']:.4f}")

    # Summary table
    summary = analyzer.summary_table()
    print(f"\n=== Headline Numbers ===")
    overall = summary["overall"]
    print(f"  N={overall['n']}  accuracy={overall['accuracy']:.3f}  "
          f"mean_score={overall['mean_score']:.4f}")
    for metric, roc_stat in overall["roc_auc"].items():
        corr_stat = overall["correlations"][metric]
        ci = corr_stat.get("ci_95", [None, None])
        ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci[0] is not None else ""
        print(f"  {metric:<25s}  r={corr_stat['r']:+.4f}  AUC={roc_stat['auc']:.4f}  {ci_str}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/analyze_traces.py
git commit -m "Add trace analysis script with correlation tables and ROC AUC"
```

---

### Task 6: Collect Traces on RunPod

**Files:**
- No new code — this task syncs code to RunPod and runs the collection.

This task is executed manually by the human or via SSH commands. It requires the RunPod pod to be running with Qwen2.5-7B cached.

- [ ] **Step 1: Sync code to RunPod**

```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    -e "ssh -p 17599 -i ~/.ssh/id_rsa" \
    /Users/nathan/Projects/ndouglas/adaptive-recursive-inference-experiment/ \
    root@91.199.227.82:/workspace/ari/
```

- [ ] **Step 2: Install new dependencies on RunPod**

```bash
ssh -p 17599 -i ~/.ssh/id_rsa root@91.199.227.82 \
    "cd /workspace/ari && pip3 install scikit-learn"
```

- [ ] **Step 3: Run trace collection on expanded math dataset**

```bash
ssh -p 17599 -i ~/.ssh/id_rsa root@91.199.227.82 \
    "cd /workspace/ari && HF_HOME=/workspace/huggingface \
     python3 -u scripts/collect_traces.py \
       --model Qwen/Qwen2.5-7B \
       --data data/math_probe_expanded.json \
       --output /workspace/traces_7b_math_expanded.json \
       --threshold 0.80 --max-iters 4 \
       --block-i 15 --block-j 20"
```

- [ ] **Step 4: Also collect traces on the reasoning probes**

```bash
ssh -p 17599 -i ~/.ssh/id_rsa root@91.199.227.82 \
    "cd /workspace/ari && HF_HOME=/workspace/huggingface \
     python3 -u scripts/collect_traces.py \
       --model Qwen/Qwen2.5-7B \
       --data data/reasoning_probe.json \
       --output /workspace/traces_7b_reasoning.json \
       --threshold 0.80 --max-iters 4 \
       --block-i 15 --block-j 20"
```

- [ ] **Step 5: Copy results locally**

```bash
scp -P 17599 -i ~/.ssh/id_rsa root@91.199.227.82:/workspace/traces_7b_math_expanded.json results/
scp -P 17599 -i ~/.ssh/id_rsa root@91.199.227.82:/workspace/traces_7b_reasoning.json results/
```

- [ ] **Step 6: Commit trace data**

```bash
git add results/traces_7b_math_expanded.json results/traces_7b_reasoning.json
git commit -m "Collected convergence traces: 100 math + 38 reasoning probes (7B, theta=0.80)"
```

---

### Task 7: Run Analysis and Interpret Results

**Files:**
- No new code — runs existing analysis script on collected traces.

- [ ] **Step 1: Analyze math traces**

```bash
uv run python scripts/analyze_traces.py results/traces_7b_math_expanded.json --group-by category
```

- [ ] **Step 2: Analyze reasoning traces**

```bash
uv run python scripts/analyze_traces.py results/traces_7b_reasoning.json --group-by category
```

- [ ] **Step 3: Save analysis results**

```bash
uv run python scripts/analyze_traces.py results/traces_7b_math_expanded.json \
    --output results/analysis_math.json
uv run python scripts/analyze_traces.py results/traces_7b_reasoning.json \
    --output results/analysis_reasoning.json
```

- [ ] **Step 4: Commit analysis results**

```bash
git add results/analysis_math.json results/analysis_reasoning.json
git commit -m "Stage 2 analysis: convergence-correctness correlations and ROC AUC"
```

- [ ] **Step 5: Interpret results**

The key numbers to evaluate:
- **ROC AUC > 0.65 for any metric/task type** = convergence predicts correctness (proceed to Stage 3)
- **ROC AUC 0.55-0.65** = weak signal, may need larger dataset or different metrics
- **ROC AUC < 0.55** = no signal, pivot to negative-result writeup

Check whether the signal is task-dependent: does it work better for some categories than others?
