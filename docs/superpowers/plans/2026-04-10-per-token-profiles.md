# Stage 3: Per-Token Convergence Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify generated tokens by JSON role (structural, reasoning, answer), compare convergence behavior across roles and between correct/incorrect outputs, and produce convergence heatmap visualizations.

**Architecture:** A `TokenRoleClassifier` maps token positions to JSON roles by re-tokenizing generated text with the Qwen tokenizer. A `TokenProfileAnalyzer` aggregates convergence metrics by role and correctness. A `plot_convergence_heatmaps.py` script produces per-problem and aggregate heatmaps using matplotlib. All analysis runs locally on Mac against trace JSON files collected in Stage 2.

**Tech Stack:** Existing trace JSON files, `transformers.AutoTokenizer` (Qwen2.5-1.5B, same vocab as 7B), numpy, matplotlib, scipy

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/analysis/token_roles.py` | `TokenRoleClassifier`: map token positions to JSON roles |
| Create | `src/analysis/token_profiles.py` | `TokenProfileAnalyzer`: aggregate convergence by role, correctness |
| Create | `scripts/plot_convergence_heatmaps.py` | Generate heatmap visualizations from trace files |
| Create | `tests/analysis/test_token_roles.py` | Tests for token role classification |
| Create | `tests/analysis/test_token_profiles.py` | Tests for token profile analysis |

---

### Task 1: TokenRoleClassifier

**Files:**
- Create: `src/analysis/token_roles.py`
- Create: `tests/analysis/test_token_roles.py`

This class re-tokenizes a generated JSON string to map each token position to one of three roles: `structural` (JSON syntax tokens like `{`, `"reasoning":`, `"answer":`), `reasoning` (content of the reasoning field), `answer` (the numeric answer value).

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_token_roles.py`:

```python
"""Tests for TokenRoleClassifier."""
import pytest

from src.analysis.token_roles import TokenRoleClassifier, TokenRole


class TestTokenRoleClassifier:
    @pytest.fixture
    def classifier(self):
        return TokenRoleClassifier("Qwen/Qwen2.5-1.5B")

    def test_classify_simple_json(self, classifier):
        text = '{"reasoning": "2 + 3 = 5", "answer": 5}'
        roles = classifier.classify(text)
        assert len(roles) > 0
        # First token is always structural (opening brace + key)
        assert roles[0] == TokenRole.STRUCTURAL
        # Last token is structural (closing brace)
        assert roles[-1] == TokenRole.STRUCTURAL
        # Should have all three roles
        role_set = set(roles)
        assert TokenRole.STRUCTURAL in role_set
        assert TokenRole.REASONING in role_set
        assert TokenRole.ANSWER in role_set

    def test_structural_tokens_identified(self, classifier):
        text = '{"reasoning": "yes", "answer": 42}'
        roles = classifier.classify(text)
        # Count structural tokens — at minimum the JSON scaffolding
        structural_count = sum(1 for r in roles if r == TokenRole.STRUCTURAL)
        assert structural_count >= 4  # {, "reasoning":, "answer":, }

    def test_answer_tokens_at_end(self, classifier):
        text = '{"reasoning": "the answer is 7", "answer": 7}'
        roles = classifier.classify(text)
        # The answer tokens should come after reasoning tokens
        last_reasoning_idx = max(
            i for i, r in enumerate(roles) if r == TokenRole.REASONING
        )
        first_answer_idx = min(
            i for i, r in enumerate(roles) if r == TokenRole.ANSWER
        )
        assert first_answer_idx > last_reasoning_idx

    def test_returns_correct_length(self, classifier):
        text = '{"reasoning": "test content here", "answer": 123}'
        roles = classifier.classify(text)
        # Should have one role per token
        tokenizer = classifier.tokenizer
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(roles) == len(token_ids)

    def test_role_enum_values(self):
        assert TokenRole.STRUCTURAL.value == "structural"
        assert TokenRole.REASONING.value == "reasoning"
        assert TokenRole.ANSWER.value == "answer"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/analysis/test_token_roles.py -v`

Expected: `ModuleNotFoundError: No module named 'src.analysis.token_roles'`

- [ ] **Step 3: Implement TokenRoleClassifier**

Create `src/analysis/token_roles.py`:

```python
"""Classify generated tokens by their role in constrained JSON output.

The generated text has the structure:
    {"reasoning": "<work>", "answer": <number>}

Each token is classified as:
- STRUCTURAL: JSON syntax (braces, keys, colons, commas, quotes around fields)
- REASONING: content inside the "reasoning" value string
- ANSWER: the numeric answer value tokens
"""
from enum import Enum

from transformers import AutoTokenizer


class TokenRole(Enum):
    STRUCTURAL = "structural"
    REASONING = "reasoning"
    ANSWER = "answer"


class TokenRoleClassifier:
    """Classify tokens in constrained JSON output by their semantic role.

    Uses the Qwen tokenizer to re-tokenize generated text and map each
    token to a JSON role based on character position.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def classify(self, generated_text):
        """Classify each token in generated_text by JSON role.

        Args:
            generated_text: The full generated JSON string,
                e.g. '{"reasoning": "work here", "answer": 42}'

        Returns:
            List of TokenRole, one per token.
        """
        token_ids = self.tokenizer.encode(generated_text, add_special_tokens=False)
        n_tokens = len(token_ids)

        # Find character boundaries for each JSON section
        # Structure: {"reasoning": "...", "answer": ...}
        # We need to find where the reasoning value starts/ends
        # and where the answer value starts/ends.
        reasoning_start, reasoning_end = self._find_reasoning_span(generated_text)
        answer_start, answer_end = self._find_answer_span(generated_text)

        # Map each token to its character span by decoding incrementally
        char_spans = self._token_char_spans(token_ids)

        roles = []
        for start, end in char_spans:
            mid = (start + end) // 2
            if reasoning_start <= mid < reasoning_end:
                roles.append(TokenRole.REASONING)
            elif answer_start <= mid < answer_end:
                roles.append(TokenRole.ANSWER)
            else:
                roles.append(TokenRole.STRUCTURAL)

        return roles

    def _find_reasoning_span(self, text):
        """Find char range of the reasoning value content (inside quotes)."""
        # Find "reasoning": "  then content until closing "
        key = '"reasoning":'
        idx = text.find(key)
        if idx == -1:
            return (0, 0)
        # Skip past key and whitespace to opening quote
        after_key = idx + len(key)
        quote_start = text.find('"', after_key)
        if quote_start == -1:
            return (0, 0)
        # Find matching closing quote (handle escaped quotes)
        pos = quote_start + 1
        while pos < len(text):
            if text[pos] == '\\':
                pos += 2
                continue
            if text[pos] == '"':
                break
            pos += 1
        return (quote_start + 1, pos)

    def _find_answer_span(self, text):
        """Find char range of the answer value (the number after "answer":)."""
        key = '"answer":'
        idx = text.rfind(key)  # rfind in case "answer" appears in reasoning
        if idx == -1:
            return (0, 0)
        after_key = idx + len(key)
        # Skip whitespace
        pos = after_key
        while pos < len(text) and text[pos] == ' ':
            pos += 1
        # Answer value runs until } or end of string
        end = pos
        while end < len(text) and text[end] not in ('}', ','):
            end += 1
        return (pos, end)

    def _token_char_spans(self, token_ids):
        """Compute (start, end) character span for each token.

        Decodes incrementally to find where each token's characters fall.
        """
        spans = []
        prev_len = 0
        for i in range(len(token_ids)):
            decoded = self.tokenizer.decode(token_ids[:i + 1])
            cur_len = len(decoded)
            spans.append((prev_len, cur_len))
            prev_len = cur_len
        return spans
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/analysis/test_token_roles.py -v`

Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/analysis/token_roles.py tests/analysis/test_token_roles.py
git commit -m "Add TokenRoleClassifier for per-token JSON role classification"
```

---

### Task 2: TokenProfileAnalyzer

**Files:**
- Create: `src/analysis/token_profiles.py`
- Create: `tests/analysis/test_token_profiles.py`

This class takes trace dicts and a `TokenRoleClassifier`, groups convergence metrics by token role, and compares correct vs incorrect outputs.

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_token_profiles.py`:

```python
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
        # Correct should have higher similarity
        for role in ["reasoning", "answer"]:
            if role in result["correct"] and role in result["incorrect"]:
                assert result["correct"][role]["mean_final_similarity"] > \
                       result["incorrect"][role]["mean_final_similarity"]

    def test_positional_profile(self, analyzer):
        gen = '{"reasoning": "x = 5", "answer": 5}'
        n_tokens = len(analyzer.classifier.tokenizer.encode(gen, add_special_tokens=False))
        # Create token traces with increasing iterations
        tts = [_make_token_trace(0, i % 4 + 1, 0.9 + i * 0.001, 0.05)
               for i in range(n_tokens)]
        trace = _make_trace(gen, tts, correct=True)

        profile = analyzer.positional_profile([trace])
        assert "positions" in profile
        assert "mean_iterations" in profile
        assert "mean_final_similarity" in profile
        assert len(profile["positions"]) == n_tokens
        assert len(profile["mean_iterations"]) == n_tokens
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/analysis/test_token_profiles.py -v`

Expected: `ModuleNotFoundError: No module named 'src.analysis.token_profiles'`

- [ ] **Step 3: Implement TokenProfileAnalyzer**

Create `src/analysis/token_profiles.py`:

```python
"""Analyze convergence profiles grouped by token role.

Aggregates per-token convergence metrics (iterations, similarity, speed)
by their JSON role (structural/reasoning/answer) and compares profiles
between correct and incorrect outputs.
"""
import numpy as np

from src.analysis.token_roles import TokenRoleClassifier, TokenRole


class TokenProfileAnalyzer:
    """Aggregate convergence behavior by token role and correctness.

    Args:
        model_name: HuggingFace model name for tokenizer (e.g., "Qwen/Qwen2.5-1.5B").
    """

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B"):
        self.classifier = TokenRoleClassifier(model_name)

    def _classify_trace(self, trace):
        """Return list of (TokenRole, token_trace_dict) pairs for a trace."""
        generated = trace["generated"]
        token_traces = trace["token_traces"]
        roles = self.classifier.classify(generated)
        # Handle length mismatch (trace may include EOS token)
        n = min(len(roles), len(token_traces))
        return [(roles[i], token_traces[i]) for i in range(n)]

    def aggregate_by_role(self, traces):
        """Compute mean convergence metrics per token role across all traces.

        Args:
            traces: List of trace dicts with "generated" and "token_traces".

        Returns:
            Dict mapping role name -> {mean_iterations, mean_final_similarity,
            mean_convergence_speed, count}.
        """
        role_data = {r.value: [] for r in TokenRole}

        for trace in traces:
            classified = self._classify_trace(trace)
            for role, tt in classified:
                role_data[role.value].append(tt)

        result = {}
        for role_name, tts in role_data.items():
            if not tts:
                continue
            result[role_name] = {
                "mean_iterations": float(np.mean([t["iterations"] for t in tts])),
                "mean_final_similarity": float(np.mean(
                    [t["final_similarity"] for t in tts if t["final_similarity"] is not None]
                )),
                "mean_convergence_speed": float(np.mean(
                    [t["convergence_speed"] for t in tts if t["convergence_speed"] is not None]
                )),
                "count": len(tts),
            }
        return result

    def compare_correct_vs_incorrect(self, traces):
        """Aggregate by role, split by correctness.

        Returns:
            {"correct": {role: metrics}, "incorrect": {role: metrics}}
        """
        correct = [t for t in traces if t["correct"]]
        incorrect = [t for t in traces if not t["correct"]]
        result = {}
        if correct:
            result["correct"] = self.aggregate_by_role(correct)
        if incorrect:
            result["incorrect"] = self.aggregate_by_role(incorrect)
        return result

    def positional_profile(self, traces):
        """Compute convergence metrics at each token position, averaged across traces.

        Pads shorter traces with NaN so all traces contribute up to their length.

        Returns:
            {"positions": list[int], "mean_iterations": list[float],
             "mean_final_similarity": list[float]}
        """
        max_len = max(len(t["token_traces"]) for t in traces)
        all_iters = np.full((len(traces), max_len), np.nan)
        all_sims = np.full((len(traces), max_len), np.nan)

        for i, trace in enumerate(traces):
            tts = trace["token_traces"]
            for j, tt in enumerate(tts):
                all_iters[i, j] = tt["iterations"]
                if tt["final_similarity"] is not None:
                    all_sims[i, j] = tt["final_similarity"]

        return {
            "positions": list(range(max_len)),
            "mean_iterations": [float(x) for x in np.nanmean(all_iters, axis=0)],
            "mean_final_similarity": [float(x) for x in np.nanmean(all_sims, axis=0)],
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/analysis/test_token_profiles.py -v`

Expected: All 3 tests pass.

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest -v`

Expected: All tests pass (existing + new).

- [ ] **Step 6: Commit**

```bash
git add src/analysis/token_profiles.py tests/analysis/test_token_profiles.py
git commit -m "Add TokenProfileAnalyzer for per-token convergence analysis by role"
```

---

### Task 3: Convergence Heatmap Visualization Script

**Files:**
- Create: `scripts/plot_convergence_heatmaps.py`

This script loads trace JSON files and produces:
1. Per-problem convergence heatmaps (position × iteration, colored by similarity)
2. Aggregate role comparison bar charts (structural vs reasoning vs answer)
3. Correct vs incorrect comparison plots

- [ ] **Step 1: Create the visualization script**

Create `scripts/plot_convergence_heatmaps.py`:

```python
"""Visualize per-token convergence profiles from trace data.

Produces:
1. Convergence heatmaps: token position (x) × iteration (y), colored by similarity
2. Role comparison: mean metrics by token role (structural/reasoning/answer)
3. Correct vs incorrect: role metrics split by correctness

Usage:
    python scripts/plot_convergence_heatmaps.py results/traces_7b_math_t0.98.json
    python scripts/plot_convergence_heatmaps.py results/traces_7b_math_t0.98.json --output-dir plots/token_profiles
    python scripts/plot_convergence_heatmaps.py results/traces_7b_math_t0.98.json --n-heatmaps 6
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.analysis.token_profiles import TokenProfileAnalyzer
from src.analysis.token_roles import TokenRoleClassifier, TokenRole


def plot_single_heatmap(trace, classifier, ax, title=None):
    """Plot a position × iteration heatmap for one trace."""
    tts = trace["token_traces"]
    max_iters = max(tt["iterations"] for tt in tts)
    n_tokens = len(tts)

    # Build similarity matrix: (iteration, position)
    matrix = np.full((max_iters, n_tokens), np.nan)
    for j, tt in enumerate(tts):
        for k, sim in enumerate(tt["similarities"]):
            matrix[k, j] = sim

    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.7, vmax=1.0,
                   interpolation="nearest", origin="lower")
    ax.set_xlabel("Token position")
    ax.set_ylabel("Iteration")

    # Add role annotations along the top
    roles = classifier.classify(trace["generated"])
    n = min(len(roles), n_tokens)
    role_colors = {
        TokenRole.STRUCTURAL: "#888888",
        TokenRole.REASONING: "#4488cc",
        TokenRole.ANSWER: "#cc4444",
    }
    for i in range(n):
        ax.axvspan(i - 0.5, i + 0.5, ymax=1.05, ymin=1.0,
                   color=role_colors.get(roles[i], "#888888"),
                   clip_on=False)

    if title:
        ax.set_title(title, fontsize=9)
    return im


def plot_heatmap_grid(traces, classifier, output_path, n=6):
    """Plot a grid of convergence heatmaps for selected traces."""
    # Pick n/2 correct and n/2 incorrect (or as many as available)
    correct = [t for t in traces if t["correct"]]
    incorrect = [t for t in traces if not t["correct"]]
    n_correct = min(len(correct), n // 2)
    n_incorrect = min(len(incorrect), n - n_correct)
    n_correct = min(len(correct), n - n_incorrect)
    selected = correct[:n_correct] + incorrect[:n_incorrect]

    if not selected:
        print("No traces to plot")
        return

    rows = (len(selected) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 3.5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, trace in enumerate(selected):
        ax = axes[idx // 2, idx % 2]
        status = "CORRECT" if trace["correct"] else "WRONG"
        q = trace.get("prompt", "")[:40]
        title = f"{status} | {q}..."
        im = plot_single_heatmap(trace, classifier, ax, title)

    # Hide unused axes
    for idx in range(len(selected), rows * 2):
        axes[idx // 2, idx % 2].set_visible(False)

    # Legend for role colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#888888", label="structural"),
        Patch(facecolor="#4488cc", label="reasoning"),
        Patch(facecolor="#cc4444", label="answer"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.suptitle("Per-Token Convergence Heatmaps (similarity by position × iteration)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap grid: {output_path}")


def plot_role_comparison(analyzer, traces, output_path):
    """Bar chart comparing convergence metrics across token roles."""
    result = analyzer.compare_correct_vs_incorrect(traces)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["mean_iterations", "mean_final_similarity", "mean_convergence_speed"]
    titles = ["Mean Iterations", "Mean Final Similarity", "Mean Convergence Speed"]
    roles = ["structural", "reasoning", "answer"]
    x = np.arange(len(roles))
    width = 0.35

    for ax, metric, title in zip(axes, metrics, titles):
        correct_vals = [result.get("correct", {}).get(r, {}).get(metric, 0) for r in roles]
        incorrect_vals = [result.get("incorrect", {}).get(r, {}).get(metric, 0) for r in roles]

        ax.bar(x - width / 2, correct_vals, width, label="Correct", color="#4488cc")
        ax.bar(x + width / 2, incorrect_vals, width, label="Incorrect", color="#cc4444")
        ax.set_xticks(x)
        ax.set_xticklabels(roles)
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Convergence Metrics by Token Role: Correct vs Incorrect")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved role comparison: {output_path}")


def plot_positional_profile(analyzer, traces, output_path):
    """Line plot of convergence metrics across token positions, split by correctness."""
    correct = [t for t in traces if t["correct"]]
    incorrect = [t for t in traces if not t["correct"]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for group, label, color in [
        (correct, "Correct", "#4488cc"),
        (incorrect, "Incorrect", "#cc4444"),
    ]:
        if not group:
            continue
        profile = analyzer.positional_profile(group)
        positions = profile["positions"]

        axes[0].plot(positions, profile["mean_iterations"], label=label,
                     color=color, alpha=0.8, linewidth=1.5)
        axes[1].plot(positions, profile["mean_final_similarity"], label=label,
                     color=color, alpha=0.8, linewidth=1.5)

    axes[0].set_ylabel("Mean Iterations")
    axes[0].set_title("Iteration Count by Token Position")
    axes[0].legend()

    axes[1].set_ylabel("Mean Final Similarity")
    axes[1].set_xlabel("Token Position")
    axes[1].set_title("Final Similarity by Token Position")
    axes[1].legend()

    fig.suptitle("Positional Convergence Profile: Correct vs Incorrect")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved positional profile: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot per-token convergence profiles")
    parser.add_argument("trace_file", help="Path to traces JSON from collect_traces.py")
    parser.add_argument("--output-dir", default="plots/token_profiles",
                        help="Directory for output plots")
    parser.add_argument("--n-heatmaps", type=int, default=6,
                        help="Number of individual heatmaps to show")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B",
                        help="Tokenizer model name")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.trace_file) as f:
        data = json.load(f)
    traces = data["traces"]
    threshold = data.get("threshold", "?")
    print(f"Loaded {len(traces)} traces (threshold={threshold})")

    classifier = TokenRoleClassifier(args.model_name)
    analyzer = TokenProfileAnalyzer(args.model_name)

    # Role comparison stats (printed)
    role_agg = analyzer.aggregate_by_role(traces)
    print("\n=== Convergence by Token Role ===")
    print(f"  {'Role':<12s}  {'Count':>6s}  {'Iters':>6s}  {'Sim':>6s}  {'Speed':>8s}")
    for role_name, stats in sorted(role_agg.items()):
        print(f"  {role_name:<12s}  {stats['count']:>6d}  "
              f"{stats['mean_iterations']:>6.2f}  "
              f"{stats['mean_final_similarity']:>6.4f}  "
              f"{stats['mean_convergence_speed']:>8.5f}")

    # Correct vs incorrect
    comparison = analyzer.compare_correct_vs_incorrect(traces)
    for group_name in ["correct", "incorrect"]:
        if group_name not in comparison:
            continue
        print(f"\n  --- {group_name} ---")
        for role_name, stats in sorted(comparison[group_name].items()):
            print(f"    {role_name:<12s}  n={stats['count']:>5d}  "
                  f"iters={stats['mean_iterations']:.2f}  "
                  f"sim={stats['mean_final_similarity']:.4f}  "
                  f"speed={stats['mean_convergence_speed']:.5f}")

    # Generate plots
    basename = os.path.splitext(os.path.basename(args.trace_file))[0]

    plot_heatmap_grid(traces, classifier,
                      os.path.join(args.output_dir, f"{basename}_heatmaps.png"),
                      n=args.n_heatmaps)

    plot_role_comparison(analyzer, traces,
                         os.path.join(args.output_dir, f"{basename}_roles.png"))

    plot_positional_profile(analyzer, traces,
                            os.path.join(args.output_dir, f"{basename}_positional.png"))

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs with --help**

Run: `uv run python scripts/plot_convergence_heatmaps.py --help`

Expected: Help text with arguments.

- [ ] **Step 3: Commit**

```bash
git add scripts/plot_convergence_heatmaps.py
git commit -m "Add per-token convergence heatmap and profile visualization script"
```

---

### Task 4: Generate Visualizations

**Files:**
- No new code — this task runs the visualization script on collected traces.

- [ ] **Step 1: Run on math traces at θ=0.98 (most variation)**

```bash
uv run python scripts/plot_convergence_heatmaps.py \
    results/traces_7b_math_t0.98.json \
    --output-dir plots/token_profiles \
    --n-heatmaps 6
```

Expected: Three PNG files in `plots/token_profiles/`, plus printed role comparison stats.

- [ ] **Step 2: Run on reasoning traces at θ=0.98**

```bash
uv run python scripts/plot_convergence_heatmaps.py \
    results/traces_7b_reasoning_t0.98.json \
    --output-dir plots/token_profiles \
    --n-heatmaps 6
```

Expected: Three more PNGs for reasoning data.

- [ ] **Step 3: Run on math traces at θ=0.80 (baseline comparison)**

```bash
uv run python scripts/plot_convergence_heatmaps.py \
    results/traces_7b_math_expanded.json \
    --output-dir plots/token_profiles \
    --n-heatmaps 6
```

Expected: Baseline comparison plots.

- [ ] **Step 4: Run on reasoning expanded at θ=0.80**

```bash
uv run python scripts/plot_convergence_heatmaps.py \
    results/traces_7b_reasoning_expanded_t0.80.json \
    --output-dir plots/token_profiles \
    --n-heatmaps 6
```

- [ ] **Step 5: Commit plots**

```bash
git add plots/token_profiles/
git commit -m "Generate per-token convergence profile visualizations across thresholds"
```

---

### Task 5: Update Implementation Plan

**Files:**
- Modify: `IMPLEMENTATION_PLAN.md`

- [ ] **Step 1: Update Stage 3 status**

Change Stage 3 status from `Not Started` to `Complete` in `IMPLEMENTATION_PLAN.md`.

- [ ] **Step 2: Commit**

```bash
git add IMPLEMENTATION_PLAN.md
git commit -m "Mark Stage 3 (per-token convergence profiles) complete"
```
