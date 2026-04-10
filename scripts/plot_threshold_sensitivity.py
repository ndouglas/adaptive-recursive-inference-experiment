"""Plot threshold sensitivity analysis from adaptive eval results."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/adaptive_eval.json")
    parser.add_argument("--output-dir", default="plots/adaptive")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)

    baseline = data["baseline"]
    thresholds = []
    math_scores = []
    eq_scores = []
    avg_iters = []

    for k in sorted(data["thresholds"], key=float):
        r = data["thresholds"][k]
        thresholds.append(r["threshold"])
        math_scores.append(r["math_score"])
        eq_scores.append(r["eq_score"])
        avg_iters.append(r["avg_iterations"])

    thresholds = np.array(thresholds)
    math_scores = np.array(math_scores)
    eq_scores = np.array(eq_scores)
    avg_iters = np.array(avg_iters)

    # --- Plot 1: Score vs Threshold (dual y-axis with iterations) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(thresholds, math_scores, "o-", color="tab:blue", linewidth=2,
             markersize=7, label="Math Score")
    ax1.plot(thresholds, eq_scores, "s-", color="tab:orange", linewidth=2,
             markersize=7, label="EQ Score")
    ax1.axhline(y=baseline["math_score"], color="tab:blue", linestyle="--",
                alpha=0.4, label=f"Math baseline ({baseline['math_score']:.3f})")
    ax1.axhline(y=baseline["eq_score"], color="tab:orange", linestyle="--",
                alpha=0.4, label=f"EQ baseline ({baseline['eq_score']:.3f})")
    ax1.set_xlabel("Cosine Similarity Threshold")
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, avg_iters, "d--", color="tab:green", linewidth=1.5,
             markersize=6, alpha=0.7, label="Avg Iterations")
    ax2.set_ylabel("Average Iterations", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylim(0, data["max_iterations"] + 0.5)
    ax2.legend(loc="upper right")

    block_i = data["block_i"]
    block_j = data["block_j"]
    fig.suptitle(f"Adaptive Halting: Threshold Sensitivity — Block ({block_i},{block_j})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "threshold_sensitivity.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.output_dir}/threshold_sensitivity.png")

    # --- Plot 2: Combined delta vs avg iterations (efficiency frontier) ---
    fig, ax = plt.subplots(figsize=(8, 6))

    math_deltas = math_scores - baseline["math_score"]
    eq_deltas = eq_scores - baseline["eq_score"]
    combined_deltas = math_deltas + eq_deltas

    scatter = ax.scatter(avg_iters, combined_deltas, c=thresholds, cmap="viridis",
                         s=100, edgecolors="black", zorder=5)
    ax.plot(avg_iters, combined_deltas, "-", color="gray", alpha=0.4, zorder=1)

    for i, t in enumerate(thresholds):
        ax.annotate(f"{t:.3f}", (avg_iters[i], combined_deltas[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Average Iterations per Token")
    ax.set_ylabel("Combined Score Delta (Math + EQ)")
    ax.set_title(f"Efficiency Frontier — Block ({block_i},{block_j})")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax, label="Threshold")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "efficiency_frontier.png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.output_dir}/efficiency_frontier.png")


if __name__ == "__main__":
    main()
