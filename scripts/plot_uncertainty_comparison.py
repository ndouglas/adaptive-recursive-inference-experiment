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

    conv_by_q = {t["prompt"]: t for t in conv_data["traces"]}
    samp_by_q = {r["question"]: r for r in samp_data["results"]}
    ent_by_q = {r["question"]: r for r in ent_data["results"]}

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

    sim = [d["convergence_similarity"] for d in matched_data]
    agr = [d["sampling_agreement"] for d in matched_data]
    axes[0].scatter(sim, agr, c=colors, alpha=0.6, s=20)
    axes[0].set_xlabel("Convergence Similarity")
    axes[0].set_ylabel("Sampling Agreement")
    axes[0].set_title("Convergence vs Sampling")

    ent = [d["mean_entropy"] for d in matched_data]
    axes[1].scatter(sim, ent, c=colors, alpha=0.6, s=20)
    axes[1].set_xlabel("Convergence Similarity")
    axes[1].set_ylabel("Mean Entropy")
    axes[1].set_title("Convergence vs Entropy")

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
    """Plot AUC vs compute cost Pareto frontier."""
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

        table = comp.summary_table()
        print(f"\n  {'Method':<25s} {'AUC':>6s} {'Cost':>6s}")
        for m in table["methods"]:
            print(f"  {m['method']:<25s} {m['auc']:>6.3f} {m['cost']:>6.2f}")

        plot_roc_curves(matched,
                        os.path.join(args.output_dir, f"roc_{label}.png"),
                        label=label)
        plot_scatter(matched,
                     os.path.join(args.output_dir, f"scatter_{label}.png"),
                     label=label)

        disagree = comp.disagreement_analysis(
            "convergence_similarity", "sampling_agreement")
        print(f"\n  Disagreement (convergence vs sampling):")
        for cat, info in disagree.items():
            print(f"    {cat}: n={info['count']}, accuracy={info['accuracy']:.3f}")

        all_pareto.append(comp.pareto_data())

    if all_pareto:
        plot_pareto(all_pareto, labels,
                    os.path.join(args.output_dir, "pareto_frontier.png"))


if __name__ == "__main__":
    main()
