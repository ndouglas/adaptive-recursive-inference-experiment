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
