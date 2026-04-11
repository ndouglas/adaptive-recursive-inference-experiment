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

from src.analysis.token_profiles import TokenProfileAnalyzer
from src.analysis.token_roles import TokenRoleClassifier, TokenRole
from src.analysis.calibration import CalibrationAnalyzer
from src.analysis.uncertainty_comparison import UncertaintyComparison

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

        ax.bar(x - width / 2, correct_vals, width,
               label="Correct", color=COLORS["correct"], alpha=0.85)
        ax.bar(x + width / 2, incorrect_vals, width,
               label="Incorrect", color=COLORS["incorrect"], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(role_labels)
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle("Convergence by Token Role (Math, θ=0.80)", fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig4_token_roles.png"))


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
        if sims:
            for i in range(len(sims), max_iters):
                matrix[i, j] = sims[-1]
    return matrix


def _select_representative(traces, correct):
    """Pick a median-convergence trace from the correct/incorrect group."""
    group = [t for t in traces if t["correct"] == correct]
    if not group:
        return None
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

        try:
            roles = classifier.classify(trace["generated"])
            n = min(len(roles), matrix.shape[1])
            for i in range(n):
                ax.axvspan(i - 0.5, i + 0.5, ymax=1.05, ymin=1.0,
                           color=role_colors.get(roles[i], "#888"),
                           clip_on=False)
        except Exception:
            pass

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label="Cosine Similarity")

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


# ---------------------------------------------------------------------------
# Fig 6: Reliability Diagrams
# ---------------------------------------------------------------------------
def fig_reliability(out_dir, results_dir):
    """Calibration reliability diagrams for best and worst metrics per task."""
    math_traces = load_json(os.path.join(results_dir, "traces_7b_math_expanded.json"))["traces"]
    reas_traces = load_json(os.path.join(results_dir, "traces_7b_reasoning_expanded_t0.80.json"))["traces"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

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


# ---------------------------------------------------------------------------
# Fig 7: Pareto Frontier
# ---------------------------------------------------------------------------
def fig_pareto(out_dir, results_dir):
    """AUC vs compute cost for all methods, both task types."""
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


# ---------------------------------------------------------------------------
# Fig 9: Summary Table
# ---------------------------------------------------------------------------
def fig_summary_table(out_dir, results_dir):
    """Render the key results as a matplotlib table figure."""
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

    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#E3F2FD")
        table[0, j].set_text_props(fontweight="bold")

    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[i, j].set_facecolor("#F5F5F5")

    fig.suptitle("Summary: Uncertainty Method Comparison", fontsize=14)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig9_summary_table.png"))


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
    fig_token_roles(out_dir, results_dir, args.model_name)
    fig_heatmaps(out_dir, results_dir, args.model_name)
    fig_reliability(out_dir, results_dir)
    fig_pareto(out_dir, results_dir)
    fig_scatter(out_dir, results_dir)
    fig_summary_table(out_dir, results_dir)
    print(f"\nDone. {len([f for f in os.listdir(out_dir) if f.endswith('.png')])} figures in {out_dir}/")


if __name__ == "__main__":
    main()
