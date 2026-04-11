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

    arrow_kw = dict(arrowstyle="-|>", lw=2, color="#555")
    for x1, x2, y in [(2.0, 2.8, 2.0), (5.0, 5.8, 2.0),
                       (8.8, 9.6, 2.0), (11.8, 12.5, 2.0)]:
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=arrow_kw)

    loop_x = 7.3
    ax.annotate("", xy=(loop_x, 3.3), xytext=(loop_x + 0.8, 3.3),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#E65100",
                                connectionstyle="arc3,rad=-1.2"))
    ax.text(7.3, 3.8, "cos(h_i, h_{i-1}) < θ ?  → repeat", ha="center",
            fontsize=9, color="#E65100", style="italic")

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
