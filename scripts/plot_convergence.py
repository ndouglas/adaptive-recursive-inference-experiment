"""Plot convergence curves from static loop experiment results.

Generates:
1. Score vs iteration count (math and EQ) for each block
2. Combined comparison of blocks
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_scores_vs_iterations(results_list, labels, output_path):
    """Plot math and EQ scores vs total passes for multiple blocks."""
    fig, (ax_math, ax_eq) = plt.subplots(1, 2, figsize=(14, 5))

    for results, label in zip(results_list, labels):
        baseline = results["baseline"]
        passes = [1]
        math_scores = [baseline["math_score"]]
        eq_scores = [baseline["eq_score"]]

        for k in sorted(results["iterations"], key=int):
            r = results["iterations"][k]
            passes.append(r["total_passes"])
            math_scores.append(r["math_score"])
            eq_scores.append(r["eq_score"])

        ax_math.plot(passes, math_scores, "o-", label=label, linewidth=2, markersize=6)
        ax_eq.plot(passes, eq_scores, "o-", label=label, linewidth=2, markersize=6)

    ax_math.axhline(y=results_list[0]["baseline"]["math_score"], color="gray",
                     linestyle="--", alpha=0.5, label="baseline")
    ax_math.set_xlabel("Total Passes")
    ax_math.set_ylabel("Math Score")
    ax_math.set_title("Math Score vs Iteration Count")
    ax_math.legend()
    ax_math.set_ylim(0, 1)
    ax_math.grid(True, alpha=0.3)

    ax_eq.axhline(y=results_list[0]["baseline"]["eq_score"], color="gray",
                   linestyle="--", alpha=0.5, label="baseline")
    ax_eq.set_xlabel("Total Passes")
    ax_eq.set_ylabel("EQ Score")
    ax_eq.set_title("EQ Score vs Iteration Count")
    ax_eq.legend()
    ax_eq.set_ylim(0, 1)
    ax_eq.grid(True, alpha=0.3)

    fig.suptitle("Static Loop: Score vs Iteration Count (Qwen2.5-1.5B)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_deltas_vs_iterations(results_list, labels, output_path):
    """Plot score deltas vs total passes, highlighting the sweet spot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for results, label in zip(results_list, labels):
        baseline = results["baseline"]
        passes = []
        combined_deltas = []

        for k in sorted(results["iterations"], key=int):
            r = results["iterations"][k]
            passes.append(r["total_passes"])
            md = r["math_score"] - baseline["math_score"]
            ed = r["eq_score"] - baseline["eq_score"]
            combined_deltas.append(md + ed)

        ax.plot(passes, combined_deltas, "o-", label=label, linewidth=2, markersize=6)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Total Passes")
    ax.set_ylabel("Combined Delta (Math + EQ)")
    ax.set_title("Combined Score Delta vs Iteration Count (Qwen2.5-1.5B)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot convergence curves")
    parser.add_argument("--results", nargs="+", required=True,
                        help="Paths to static_loop result JSON files")
    parser.add_argument("--labels", nargs="+",
                        help="Labels for each result file")
    parser.add_argument("--output-dir", default="plots/convergence")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results_list = [load_results(p) for p in args.results]
    labels = args.labels or [
        f"({r['block_i']},{r['block_j']})" for r in results_list
    ]

    plot_scores_vs_iterations(
        results_list, labels,
        os.path.join(args.output_dir, "scores_vs_iterations.png"),
    )
    plot_deltas_vs_iterations(
        results_list, labels,
        os.path.join(args.output_dir, "deltas_vs_iterations.png"),
    )


if __name__ == "__main__":
    main()
