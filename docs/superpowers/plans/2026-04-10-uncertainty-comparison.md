# Stage 5: Uncertainty Method Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Benchmark convergence-based uncertainty against sampling agreement and softmax entropy to show cost-effectiveness.

**Architecture:** Two RunPod collection scripts generate sampling (N=8, temperature=0.7) and entropy (greedy with softmax capture) data. An `UncertaintyComparison` analyzer merges the three data sources and computes ROC AUC per method. A visualization script produces ROC curves, scatter plots, and a Pareto frontier (AUC vs compute cost).

**Tech Stack:** Existing trace JSON files, numpy, scipy, sklearn (roc_auc_score, roc_curve), matplotlib

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/collect_samples.py` | Generate N=8 samples per probe with temperature, compute agreement |
| Create | `scripts/collect_entropy.py` | Greedy generation capturing per-token softmax entropy |
| Create | `src/analysis/uncertainty_comparison.py` | Merge data sources, compute ROC AUC, Pareto data, disagreements |
| Create | `tests/analysis/test_uncertainty_comparison.py` | Tests for UncertaintyComparison |
| Create | `scripts/plot_uncertainty_comparison.py` | ROC curves, scatter plots, Pareto frontier |

---

### Task 1: Sampling Collection Script

**Files:**
- Create: `scripts/collect_samples.py`

This script generates N=8 samples per probe using temperature sampling with constrained JSON decoding. It computes answer agreement as the sampling-based uncertainty signal.

- [ ] **Step 1: Create the sampling script**

Create `scripts/collect_samples.py`:

```python
"""Collect sampling-based uncertainty data.

Generates N samples per probe using temperature sampling with constrained
JSON decoding. Records answer agreement rate as the uncertainty signal.

Usage:
    python scripts/collect_samples.py \
        --model Qwen/Qwen2.5-7B \
        --data data/math_probe_expanded.json \
        --output results/samples_7b_math.json \
        --num-samples 8 --temperature 0.7
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


def collect_samples_for_probe(model, tokenizer, json_processor, probe,
                               num_samples, temperature):
    """Generate num_samples outputs for a single probe, return result dict."""
    prompt = PROMPT_TEMPLATE.format(question=probe["question"])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Greedy (baseline) answer first
    json_processor.reset()
    with torch.no_grad():
        greedy_out = model.generate(
            input_ids, max_new_tokens=256, do_sample=False,
            logits_processor=LogitsProcessorList([json_processor]),
        )
    greedy_text = tokenizer.decode(
        greedy_out[0, input_ids.shape[1]:], skip_special_tokens=True
    )
    greedy_answer = extract_answer(greedy_text)
    greedy_correct = score_answer(greedy_answer, probe["answer"]) > 0.99

    # Sampled outputs
    samples = []
    for _ in range(num_samples):
        json_processor.reset()
        with torch.no_grad():
            sample_out = model.generate(
                input_ids, max_new_tokens=256,
                do_sample=True, temperature=temperature, top_p=0.95,
                logits_processor=LogitsProcessorList([json_processor]),
            )
        sample_text = tokenizer.decode(
            sample_out[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        answer = extract_answer(sample_text)
        samples.append({"generated": sample_text, "answer": answer})

    # Compute agreement
    valid_answers = [s["answer"] for s in samples if s["answer"] is not None]
    if valid_answers:
        counter = Counter(valid_answers)
        majority_answer, majority_count = counter.most_common(1)[0]
        agreement = majority_count / len(valid_answers)
        majority_correct = score_answer(majority_answer, probe["answer"]) > 0.99
    else:
        majority_answer = None
        agreement = 0.0
        majority_correct = False

    return {
        "question": probe["question"],
        "expected": probe["answer"],
        "category": probe.get("category", "unknown"),
        "difficulty": probe.get("difficulty", "unknown"),
        "greedy_answer": greedy_answer,
        "greedy_correct": greedy_correct,
        "samples": samples,
        "agreement": agreement,
        "majority_answer": majority_answer,
        "majority_correct": majority_correct,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect sampling-based uncertainty")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data", default="data/math_probe_expanded.json")
    parser.add_argument("--output", default="results/samples_7b_math.json")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
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
    t0 = time.time()
    for i, probe in enumerate(probes):
        result = collect_samples_for_probe(
            model, tokenizer, json_processor, probe,
            args.num_samples, args.temperature,
        )
        results.append(result)
        status = "OK" if result["greedy_correct"] else "WRONG"
        print(f"  [{i+1}/{len(probes)}] agreement={result['agreement']:.2f} "
              f"greedy={status} majority={'OK' if result['majority_correct'] else 'WRONG'}")

        # Incremental save
        output = {
            "model": args.model,
            "data": args.data,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "num_probes": len(results),
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    agreements = [r["agreement"] for r in results]
    greedy_correct = sum(1 for r in results if r["greedy_correct"])
    majority_correct = sum(1 for r in results if r["majority_correct"])

    print(f"\n=== Summary ===")
    print(f"  Probes: {len(results)}")
    print(f"  Greedy correct: {greedy_correct}/{len(results)}")
    print(f"  Majority correct: {majority_correct}/{len(results)}")
    print(f"  Mean agreement: {sum(agreements)/len(agreements):.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/collect_samples.py
git commit -m "Add sampling-based uncertainty collection script (N=8, temperature)"
```

---

### Task 2: Entropy Collection Script

**Files:**
- Create: `scripts/collect_entropy.py`

Custom generation loop that captures per-token softmax entropy from the baseline model (no adaptive loop). Entropy is computed from raw logits before JSON constraint masking, giving the model's native uncertainty.

- [ ] **Step 1: Create the entropy script**

Create `scripts/collect_entropy.py`:

```python
"""Collect softmax-entropy uncertainty data.

Runs greedy generation with a custom loop that captures per-token softmax
entropy from the baseline model (no adaptive loop). Entropy is computed
from raw logits before the JSON constraint is applied, giving the model's
native uncertainty signal.

Usage:
    python scripts/collect_entropy.py \
        --model Qwen/Qwen2.5-7B \
        --data data/math_probe_expanded.json \
        --output results/entropy_7b_math.json
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


def generate_with_entropy(model, tokenizer, input_ids, json_processor,
                          max_new_tokens=256):
    """Greedy generation capturing per-token entropy from raw logits.

    Returns:
        generated_text: str
        token_entropies: list of float (one per generated token)
    """
    device = next(model.parameters()).device
    current_ids = input_ids.to(device)
    eos_id = tokenizer.eos_token_id
    token_entropies = []

    json_processor.reset()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(current_ids)
            raw_logits = outputs.logits[:, -1, :]

            # Compute entropy from raw (unconstrained) logits
            probs = F.softmax(raw_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            token_entropies.append(entropy.item())

            # Apply JSON constraint for token selection
            constrained_logits = json_processor(current_ids, raw_logits.clone())
            next_token = constrained_logits.argmax(dim=-1, keepdim=True)

            current_ids = torch.cat([current_ids, next_token], dim=1)

            if next_token.item() == eos_id:
                break

    generated_ids = current_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, token_entropies


def main():
    parser = argparse.ArgumentParser(description="Collect softmax-entropy uncertainty")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data", default="data/math_probe_expanded.json")
    parser.add_argument("--output", default="results/entropy_7b_math.json")
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
    t0 = time.time()
    for i, probe in enumerate(probes):
        prompt = PROMPT_TEMPLATE.format(question=probe["question"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        generated, entropies = generate_with_entropy(
            model, tokenizer, input_ids, json_processor,
        )
        answer = extract_answer(generated)
        correct = score_answer(answer, probe["answer"]) > 0.99
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        result = {
            "question": probe["question"],
            "expected": probe["answer"],
            "category": probe.get("category", "unknown"),
            "difficulty": probe.get("difficulty", "unknown"),
            "generated": generated,
            "answer": answer,
            "correct": correct,
            "mean_entropy": mean_entropy,
            "token_entropies": entropies,
        }
        results.append(result)

        status = "OK" if correct else "WRONG"
        print(f"  [{i+1}/{len(probes)}] entropy={mean_entropy:.3f} {status}")

        # Incremental save
        output = {
            "model": args.model,
            "data": args.data,
            "num_probes": len(results),
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    correct_count = sum(1 for r in results if r["correct"])
    mean_ent = sum(r["mean_entropy"] for r in results) / len(results)

    print(f"\n=== Summary ===")
    print(f"  Probes: {len(results)}")
    print(f"  Correct: {correct_count}/{len(results)}")
    print(f"  Mean entropy: {mean_ent:.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/collect_entropy.py
git commit -m "Add softmax-entropy uncertainty collection script"
```

---

### Task 3: UncertaintyComparison Analyzer

**Files:**
- Create: `src/analysis/uncertainty_comparison.py`
- Create: `tests/analysis/test_uncertainty_comparison.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_uncertainty_comparison.py`:

```python
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
        # All methods should have AUC > 0.6 on data with signal
        assert aucs["convergence_similarity"]["auc"] > 0.6
        assert aucs["sampling_agreement"]["auc"] > 0.6
        assert aucs["mean_entropy"]["auc"] > 0.6

    def test_roc_auc_all_without_signal(self):
        data = _make_random_data()
        comp = UncertaintyComparison(data)
        aucs = comp.roc_auc_all()
        # All AUCs should be near 0.5
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
        # Entropy (1 pass) < convergence (~1.1 passes) < sampling (8 passes)
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/analysis/test_uncertainty_comparison.py -v`

Expected: `ModuleNotFoundError: No module named 'src.analysis.uncertainty_comparison'`

- [ ] **Step 3: Implement UncertaintyComparison**

Create `src/analysis/uncertainty_comparison.py`:

```python
"""Comparison of uncertainty estimation methods.

Compares convergence-based, sampling-based, and softmax-entropy uncertainty
signals using ROC AUC, Pareto analysis, and disagreement analysis.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


# Methods and their signal direction: True = higher value means more confident
METHOD_CONFIG = {
    "convergence_similarity": {"key": "convergence_similarity", "higher_is_confident": True},
    "convergence_iterations": {"key": "convergence_iterations", "higher_is_confident": False},
    "convergence_speed": {"key": "convergence_speed", "higher_is_confident": True},
    "sampling_agreement": {"key": "sampling_agreement", "higher_is_confident": True},
    "mean_entropy": {"key": "mean_entropy", "higher_is_confident": False},
}


class UncertaintyComparison:
    """Compare uncertainty methods head-to-head.

    Args:
        matched_data: List of dicts, each with keys: correct,
            convergence_similarity, convergence_iterations, convergence_speed,
            sampling_agreement, mean_entropy.
        block_layers: Number of layers in the adaptive circuit block (for cost calc).
        total_layers: Total model layers (for cost calc).
        num_samples: Number of samples used for sampling method (for cost calc).
    """

    def __init__(self, matched_data, block_layers=6, total_layers=32,
                 num_samples=8):
        self.data = matched_data
        self.block_layers = block_layers
        self.total_layers = total_layers
        self.num_samples = num_samples
        self._correct = np.array([d["correct"] for d in matched_data], dtype=float)

    def _get_scores(self, method):
        """Get confidence scores for a method (higher = more confident)."""
        config = METHOD_CONFIG[method]
        values = np.array([d[config["key"]] for d in self.data])
        if not config["higher_is_confident"]:
            values = -values
        return values

    def roc_auc_all(self):
        """Compute ROC AUC for each method.

        Returns:
            Dict mapping method name -> {"auc": float}.
        """
        results = {}
        for method in METHOD_CONFIG:
            scores = self._get_scores(method)
            if len(np.unique(self._correct)) < 2:
                results[method] = {"auc": 0.5}
                continue
            try:
                auc = roc_auc_score(self._correct, scores)
                results[method] = {"auc": float(auc)}
            except ValueError:
                results[method] = {"auc": 0.5}
        return results

    def _compute_cost(self, method):
        """Compute cost in forward-pass equivalents."""
        if method == "mean_entropy":
            return 1.0
        if method == "sampling_agreement":
            return float(self.num_samples)
        if method.startswith("convergence_"):
            # Cost = 1 full pass + (avg_extra_iters * block_fraction)
            avg_iters = np.mean([d["convergence_iterations"] for d in self.data])
            block_fraction = self.block_layers / self.total_layers
            return 1.0 + (avg_iters - 1) * block_fraction
        return 1.0

    def pareto_data(self):
        """Compute AUC and cost for each method (for Pareto frontier plot).

        Returns:
            List of {"method": str, "auc": float, "cost": float}.
        """
        aucs = self.roc_auc_all()
        result = []
        for method in METHOD_CONFIG:
            result.append({
                "method": method,
                "auc": aucs[method]["auc"],
                "cost": self._compute_cost(method),
            })
        return result

    def disagreement_analysis(self, method_a, method_b,
                               threshold_a=0.5, threshold_b=0.5):
        """Analyze cases where two methods disagree on confidence.

        Normalizes scores to [0,1] and splits at the given thresholds.

        Returns:
            Dict with keys: a_confident_b_not, b_confident_a_not,
            both_confident, neither_confident. Each has count and accuracy.
        """
        scores_a = self._get_scores(method_a)
        scores_b = self._get_scores(method_b)

        # Normalize to [0, 1]
        def normalize(s):
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)

        norm_a = normalize(scores_a)
        norm_b = normalize(scores_b)

        conf_a = norm_a >= threshold_a
        conf_b = norm_b >= threshold_b

        categories = {
            "both_confident": conf_a & conf_b,
            "a_confident_b_not": conf_a & ~conf_b,
            "b_confident_a_not": ~conf_a & conf_b,
            "neither_confident": ~conf_a & ~conf_b,
        }

        result = {}
        for name, mask in categories.items():
            count = int(mask.sum())
            acc = float(self._correct[mask].mean()) if count > 0 else 0.0
            result[name] = {"count": count, "accuracy": acc}
        return result

    def summary_table(self):
        """Produce a complete comparison summary."""
        aucs = self.roc_auc_all()
        methods = []
        for method in METHOD_CONFIG:
            methods.append({
                "method": method,
                "auc": aucs[method]["auc"],
                "cost": self._compute_cost(method),
            })
        methods.sort(key=lambda x: x["auc"], reverse=True)
        return {
            "n": len(self.data),
            "accuracy": float(self._correct.mean()),
            "methods": methods,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/analysis/test_uncertainty_comparison.py -v`

Expected: All 7 tests pass.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/analysis/uncertainty_comparison.py tests/analysis/test_uncertainty_comparison.py
git commit -m "Add UncertaintyComparison analyzer for method benchmarking"
```

---

### Task 4: Comparison Visualization Script

**Files:**
- Create: `scripts/plot_uncertainty_comparison.py`

- [ ] **Step 1: Create the visualization script**

Create `scripts/plot_uncertainty_comparison.py`:

```python
"""Plot uncertainty method comparison: ROC curves, scatter plots, Pareto frontier.

Usage:
    python scripts/plot_uncertainty_comparison.py \
        --convergence-traces results/traces_7b_math_expanded.json \
        --samples results/samples_7b_math.json \
        --entropy results/entropy_7b_math.json \
        --output-dir plots/uncertainty_comparison \
        --label math
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.analysis.uncertainty_comparison import UncertaintyComparison, METHOD_CONFIG


def merge_data_sources(convergence_file, samples_file, entropy_file):
    """Match problems across the three data sources by question text."""
    with open(convergence_file) as f:
        conv_data = json.load(f)
    with open(samples_file) as f:
        samp_data = json.load(f)
    with open(entropy_file) as f:
        ent_data = json.load(f)

    # Index by question
    conv_by_q = {t["prompt"]: t for t in conv_data["traces"]}
    samp_by_q = {r["question"]: r for r in samp_data["results"]}
    ent_by_q = {r["question"]: r for r in ent_data["results"]}

    # Find common questions
    common = set(conv_by_q) & set(samp_by_q) & set(ent_by_q)
    if not common:
        print(f"WARNING: No matching questions found!")
        print(f"  Convergence prompts: {list(conv_by_q.keys())[:3]}")
        print(f"  Sample questions: {list(samp_by_q.keys())[:3]}")
        print(f"  Entropy questions: {list(ent_by_q.keys())[:3]}")
        return []

    matched = []
    for q in sorted(common):
        c = conv_by_q[q]
        s = samp_by_q[q]
        e = ent_by_q[q]
        matched.append({
            "question": q,
            "correct": c["correct"],
            "convergence_similarity": c["summary"]["avg_final_similarity"],
            "convergence_iterations": c["summary"]["avg_iterations"],
            "convergence_speed": c["summary"]["avg_convergence_speed"],
            "sampling_agreement": s["agreement"],
            "mean_entropy": e["mean_entropy"],
        })

    print(f"Matched {len(matched)} problems across all three sources")
    return matched


def plot_roc_curves(matched_data, output_path, label=""):
    """Overlay ROC curves for all methods."""
    correct = np.array([d["correct"] for d in matched_data], dtype=float)
    if len(np.unique(correct)) < 2:
        print("Cannot plot ROC — all same class")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"convergence_similarity": "#4488cc",
              "convergence_iterations": "#66aadd",
              "convergence_speed": "#88ccee",
              "sampling_agreement": "#cc4444",
              "mean_entropy": "#44aa44"}
    styles = {"convergence_similarity": "-",
              "convergence_iterations": "--",
              "convergence_speed": ":",
              "sampling_agreement": "-",
              "mean_entropy": "-"}

    for method, config in METHOD_CONFIG.items():
        values = np.array([d[config["key"]] for d in matched_data])
        if not config["higher_is_confident"]:
            values = -values
        try:
            auc = roc_auc_score(correct, values)
            fpr, tpr, _ = roc_curve(correct, values)
            display_name = method.replace("convergence_", "conv. ").replace("_", " ")
            ax.plot(fpr, tpr, color=colors[method], linestyle=styles[method],
                    label=f"{display_name} (AUC={auc:.3f})", linewidth=2)
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves: Uncertainty Methods{' — ' + label if label else ''}")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curves: {output_path}")


def plot_scatter(matched_data, output_path, label=""):
    """Scatter plots: convergence vs sampling, convergence vs entropy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    correct = np.array([d["correct"] for d in matched_data])
    colors = np.where(correct, "#4488cc", "#cc4444")

    # Convergence similarity vs sampling agreement
    sim = [d["convergence_similarity"] for d in matched_data]
    agr = [d["sampling_agreement"] for d in matched_data]
    axes[0].scatter(sim, agr, c=colors, alpha=0.6, s=20)
    axes[0].set_xlabel("Convergence Similarity")
    axes[0].set_ylabel("Sampling Agreement")
    axes[0].set_title("Convergence vs Sampling")

    # Convergence similarity vs mean entropy
    ent = [d["mean_entropy"] for d in matched_data]
    axes[1].scatter(sim, ent, c=colors, alpha=0.6, s=20)
    axes[1].set_xlabel("Convergence Similarity")
    axes[1].set_ylabel("Mean Entropy")
    axes[1].set_title("Convergence vs Entropy")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4488cc',
               markersize=8, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#cc4444',
               markersize=8, label='Incorrect'),
    ]
    axes[0].legend(handles=legend_elements)

    fig.suptitle(f"Uncertainty Signal Comparison{' — ' + label if label else ''}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot: {output_path}")


def plot_pareto(pareto_points_list, labels, output_path):
    """Plot AUC vs compute cost Pareto frontier.

    Args:
        pareto_points_list: List of pareto data lists (one per dataset).
        labels: Dataset labels (e.g., ["math", "reasoning"]).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"convergence_similarity": "o", "convergence_iterations": "s",
               "convergence_speed": "^", "sampling_agreement": "D",
               "mean_entropy": "v"}
    dataset_colors = ["#4488cc", "#cc4444", "#44aa44"]

    for idx, (points, label) in enumerate(zip(pareto_points_list, labels)):
        color = dataset_colors[idx % len(dataset_colors)]
        for p in points:
            marker = markers.get(p["method"], "o")
            display = p["method"].replace("convergence_", "c.").replace("_", " ")
            ax.scatter(p["cost"], p["auc"], marker=marker, color=color,
                       s=100, zorder=5)
            ax.annotate(display, (p["cost"], p["auc"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7)
        # Connect points for this dataset
        sorted_pts = sorted(points, key=lambda x: x["cost"])
        ax.plot([p["cost"] for p in sorted_pts],
                [p["auc"] for p in sorted_pts],
                color=color, alpha=0.3, linestyle="--", label=label)

    ax.set_xlabel("Cost (forward pass equivalents)")
    ax.set_ylabel("ROC AUC")
    ax.set_title("Uncertainty Quality vs Compute Cost")
    ax.legend()
    ax.axhline(y=0.5, color="gray", alpha=0.2, linestyle=":")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pareto frontier: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Uncertainty method comparison plots")
    parser.add_argument("--convergence-traces", required=True, nargs="+")
    parser.add_argument("--samples", required=True, nargs="+")
    parser.add_argument("--entropy", required=True, nargs="+")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-dir", default="plots/uncertainty_comparison")
    parser.add_argument("--block-layers", type=int, default=6)
    parser.add_argument("--total-layers", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    labels = args.labels or [f"dataset_{i}" for i in range(len(args.convergence_traces))]

    all_pareto = []
    for i, (conv, samp, ent, label) in enumerate(zip(
        args.convergence_traces, args.samples, args.entropy, labels
    )):
        print(f"\n=== {label} ===")
        matched = merge_data_sources(conv, samp, ent)
        if not matched:
            continue

        comp = UncertaintyComparison(
            matched, block_layers=args.block_layers,
            total_layers=args.total_layers, num_samples=args.num_samples,
        )

        # Print summary
        table = comp.summary_table()
        print(f"\n  {'Method':<25s} {'AUC':>6s} {'Cost':>6s}")
        for m in table["methods"]:
            print(f"  {m['method']:<25s} {m['auc']:>6.3f} {m['cost']:>6.2f}")

        # Per-dataset plots
        plot_roc_curves(matched,
                        os.path.join(args.output_dir, f"roc_{label}.png"),
                        label=label)
        plot_scatter(matched,
                     os.path.join(args.output_dir, f"scatter_{label}.png"),
                     label=label)

        # Disagreement analysis
        disagree = comp.disagreement_analysis(
            "convergence_similarity", "sampling_agreement")
        print(f"\n  Disagreement (convergence vs sampling):")
        for cat, info in disagree.items():
            print(f"    {cat}: n={info['count']}, accuracy={info['accuracy']:.3f}")

        all_pareto.append(comp.pareto_data())

    # Combined Pareto frontier
    if all_pareto:
        plot_pareto(all_pareto, labels,
                    os.path.join(args.output_dir, "pareto_frontier.png"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it loads**

Run: `uv run python scripts/plot_uncertainty_comparison.py --help`

Expected: Help text with arguments listed.

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_uncertainty_comparison.py
git commit -m "Add uncertainty comparison visualization script"
```

---

### Task 5: Run Data Collection on RunPod

**Files:**
- No new code — runs collection scripts on RunPod.

- [ ] **Step 1: Sync code to RunPod**

```bash
rsync -avz --no-owner --no-group --exclude='.venv' --exclude='__pycache__' --exclude='.git' --exclude='plots' \
    -e "ssh -p 17599 -i ~/.ssh/id_rsa" \
    /Users/nathan/Projects/ndouglas/adaptive-recursive-inference-experiment/ \
    root@91.199.227.82:/workspace/ari/
```

- [ ] **Step 2: Run math sampling collection**

```bash
ssh -p 17599 -i ~/.ssh/id_rsa root@91.199.227.82 \
    "cd /workspace/ari && HF_HOME=/workspace/huggingface \
     python3 -u scripts/collect_samples.py \
       --model Qwen/Qwen2.5-7B \
       --data data/math_probe_expanded.json \
       --output /workspace/samples_7b_math.json \
       --num-samples 8 --temperature 0.7"
```

- [ ] **Step 3: Run reasoning sampling collection**

```bash
ssh -p 17599 -i ~/.ssh/id_rsa root@91.199.227.82 \
    "cd /workspace/ari && HF_HOME=/workspace/huggingface \
     python3 -u scripts/collect_samples.py \
       --model Qwen/Qwen2.5-7B \
       --data data/reasoning_probe_expanded.json \
       --output /workspace/samples_7b_reasoning.json \
       --num-samples 8 --temperature 0.7"
```

- [ ] **Step 4: Run math entropy collection**

```bash
ssh -p 17599 -i ~/.ssh/id_rsa root@91.199.227.82 \
    "cd /workspace/ari && HF_HOME=/workspace/huggingface \
     python3 -u scripts/collect_entropy.py \
       --model Qwen/Qwen2.5-7B \
       --data data/math_probe_expanded.json \
       --output /workspace/entropy_7b_math.json"
```

- [ ] **Step 5: Run reasoning entropy collection**

```bash
ssh -p 17599 -i ~/.ssh/id_rsa root@91.199.227.82 \
    "cd /workspace/ari && HF_HOME=/workspace/huggingface \
     python3 -u scripts/collect_entropy.py \
       --model Qwen/Qwen2.5-7B \
       --data data/reasoning_probe_expanded.json \
       --output /workspace/entropy_7b_reasoning.json"
```

- [ ] **Step 6: Copy results locally**

```bash
scp -P 17599 -i ~/.ssh/id_rsa root@91.199.227.82:/workspace/samples_7b_math.json results/
scp -P 17599 -i ~/.ssh/id_rsa root@91.199.227.82:/workspace/samples_7b_reasoning.json results/
scp -P 17599 -i ~/.ssh/id_rsa root@91.199.227.82:/workspace/entropy_7b_math.json results/
scp -P 17599 -i ~/.ssh/id_rsa root@91.199.227.82:/workspace/entropy_7b_reasoning.json results/
```

- [ ] **Step 7: Commit results**

```bash
git add results/samples_7b_*.json results/entropy_7b_*.json
git commit -m "Add sampling and entropy uncertainty data (math + reasoning)"
```

---

### Task 6: Run Comparison Analysis and Generate Plots

**Files:**
- No new code — runs analysis on collected data.

- [ ] **Step 1: Generate comparison plots for math**

```bash
uv run python scripts/plot_uncertainty_comparison.py \
    --convergence-traces results/traces_7b_math_expanded.json \
    --samples results/samples_7b_math.json \
    --entropy results/entropy_7b_math.json \
    --labels math \
    --output-dir plots/uncertainty_comparison \
    --block-layers 6 --total-layers 32 --num-samples 8
```

- [ ] **Step 2: Generate comparison plots for reasoning**

```bash
uv run python scripts/plot_uncertainty_comparison.py \
    --convergence-traces results/traces_7b_reasoning_expanded_t0.80.json \
    --samples results/samples_7b_reasoning.json \
    --entropy results/entropy_7b_reasoning.json \
    --labels reasoning \
    --output-dir plots/uncertainty_comparison \
    --block-layers 6 --total-layers 32 --num-samples 8
```

- [ ] **Step 3: Generate combined Pareto frontier**

```bash
uv run python scripts/plot_uncertainty_comparison.py \
    --convergence-traces results/traces_7b_math_expanded.json results/traces_7b_reasoning_expanded_t0.80.json \
    --samples results/samples_7b_math.json results/samples_7b_reasoning.json \
    --entropy results/entropy_7b_math.json results/entropy_7b_reasoning.json \
    --labels math reasoning \
    --output-dir plots/uncertainty_comparison \
    --block-layers 6 --total-layers 32 --num-samples 8
```

- [ ] **Step 4: Commit plots**

```bash
git add plots/uncertainty_comparison/
git commit -m "Generate uncertainty method comparison plots (ROC, scatter, Pareto)"
```

---

### Task 7: Update Implementation Plan

**Files:**
- Modify: `IMPLEMENTATION_PLAN.md`

- [ ] **Step 1: Update Stage 5 status**

Change Stage 5 status from `Not Started` to `Complete` in `IMPLEMENTATION_PLAN.md`.

- [ ] **Step 2: Commit**

```bash
git add -f IMPLEMENTATION_PLAN.md
git commit -m "Mark Stage 5 (uncertainty method comparison) complete"
```
