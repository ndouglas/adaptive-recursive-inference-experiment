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

    ax1.axvspan(0.0, 0.70, alpha=0.06, color="green", label="Safe")
    ax1.axvspan(0.70, 0.95, alpha=0.06, color="gold", label="Plateau")
    ax1.axvspan(0.95, 1.0, alpha=0.06, color="red", label="Cliff")

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
    fig_roc_curves(out_dir, results_dir)
    fig_phase_transition(out_dir, results_dir)
    print(f"\nDone. {len([f for f in os.listdir(out_dir) if f.endswith('.png')])} figures in {out_dir}/")


if __name__ == "__main__":
    main()
