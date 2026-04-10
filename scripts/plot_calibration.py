"""Plot calibration reliability diagrams and phase transition curves.

Usage:
    # Calibration from trace files
    python scripts/plot_calibration.py calibration \
        results/traces_7b_math_expanded.json \
        --output-dir plots/calibration

    # Phase transition from sweep results
    python scripts/plot_calibration.py phase-transition \
        results/phase_sweep_math.json results/phase_sweep_reasoning.json \
        --output-dir plots/calibration
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.analysis.calibration import CalibrationAnalyzer

N_BINS = 10


def plot_reliability_diagram(loaded_data, output_path, n_bins=N_BINS):
    """Plot reliability diagrams for each trace file and metric.

    Args:
        loaded_data: List of (traces, threshold_label) tuples.
    """
    metrics = ["avg_final_similarity", "avg_iterations", "avg_convergence_speed"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for ax, metric in zip(axes, metrics):
        for traces, label in loaded_data:
            analyzer = CalibrationAnalyzer(traces, metric=metric)
            bins = analyzer.reliability_bins(n_bins)

            if not bins:
                print(f"Warning: no reliability bins for {label} / {metric}, skipping.")
                continue

            ece = analyzer.expected_calibration_error(n_bins)

            confidences = [b["confidence"] for b in bins]
            accuracies = [b["accuracy"] for b in bins]
            ax.plot(confidences, accuracies, "o-", label=f"{label} (ECE={ece:.3f})",
                    markersize=4)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(metric.replace("avg_", "").replace("_", " ").title())
        ax.legend(fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    fig.suptitle("Reliability Diagrams: Convergence Metrics as Confidence")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reliability diagram: {output_path}")


def plot_ece_comparison(loaded_data, output_path, n_bins=N_BINS):
    """Bar chart of ECE per metric and threshold.

    Args:
        loaded_data: List of (traces, threshold_label) tuples.
    """
    metrics = ["avg_final_similarity", "avg_iterations", "avg_convergence_speed"]

    data_points = []
    for traces, label in loaded_data:
        eces = {}
        for metric in metrics:
            analyzer = CalibrationAnalyzer(traces, metric=metric)
            eces[metric] = analyzer.expected_calibration_error(n_bins)
        data_points.append({"threshold": label, "eces": eces})

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(data_points))
    width = 0.25

    for i, metric in enumerate(metrics):
        vals = [dp["eces"][metric] for dp in data_points]
        short_name = metric.replace("avg_", "").replace("_", " ")
        ax.bar(x + i * width, vals, width, label=short_name)

    ax.set_xticks(x + width)
    ax.set_xticklabels([dp["threshold"] for dp in data_points])
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Expected Calibration Error by Metric and Threshold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ECE comparison: {output_path}")


def plot_phase_transition(sweep_files, output_path):
    """Plot accuracy and iterations vs threshold for phase transition analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4488cc", "#cc4444", "#44aa44", "#aa44aa"]

    for idx, sweep_file in enumerate(sweep_files):
        with open(sweep_file) as f:
            data = json.load(f)
        results = data["results"]
        label = os.path.basename(sweep_file).replace("phase_sweep_", "").replace(".json", "")

        thresholds = [r["threshold"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        iterations = [r.get("mean_avg_iterations", float("nan")) for r in results]

        color = colors[idx % len(colors)]
        axes[0].plot(thresholds, accuracies, "o-", color=color, label=label, markersize=4)
        axes[1].plot(thresholds, iterations, "o-", color=color, label=label, markersize=4)

    axes[0].set_xlabel("Threshold (θ)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs Threshold")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("Threshold (θ)")
    axes[1].set_ylabel("Mean Iterations")
    axes[1].set_title("Mean Iterations vs Threshold")
    axes[1].legend()

    fig.suptitle("Phase Transition Analysis")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved phase transition plot: {output_path}")


def cmd_calibration(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all trace files once; share the data with chart functions and the ECE table.
    loaded_data = []
    for trace_file in args.trace_files:
        with open(trace_file) as f:
            data = json.load(f)
        threshold = data.get("threshold", "?")
        loaded_data.append((data["traces"], f"θ={threshold}"))

    plot_reliability_diagram(
        loaded_data,
        os.path.join(args.output_dir, "reliability_diagram.png"),
    )
    plot_ece_comparison(
        loaded_data,
        os.path.join(args.output_dir, "ece_comparison.png"),
    )

    # Print ECE table
    metrics = ["avg_final_similarity", "avg_iterations", "avg_convergence_speed"]
    print(f"\n{'Threshold':<50s}  ", end="")
    for m in metrics:
        print(f"{m.replace('avg_', ''):>15s}  ", end="")
    print()
    for traces, label in loaded_data:
        print(f"{label:<50s}  ", end="")
        for metric in metrics:
            analyzer = CalibrationAnalyzer(traces, metric=metric)
            ece = analyzer.expected_calibration_error(N_BINS)
            print(f"{ece:>15.4f}  ", end="")
        print()


def cmd_phase_transition(args):
    os.makedirs(args.output_dir, exist_ok=True)
    plot_phase_transition(
        args.sweep_files,
        os.path.join(args.output_dir, "phase_transition.png"),
    )


def main():
    parser = argparse.ArgumentParser(description="Calibration and phase transition plots")
    subparsers = parser.add_subparsers(dest="command")

    cal_parser = subparsers.add_parser("calibration")
    cal_parser.add_argument("trace_files", nargs="+")
    cal_parser.add_argument("--output-dir", default="plots/calibration")

    pt_parser = subparsers.add_parser("phase-transition")
    pt_parser.add_argument("sweep_files", nargs="+")
    pt_parser.add_argument("--output-dir", default="plots/calibration")

    args = parser.parse_args()
    if args.command == "calibration":
        cmd_calibration(args)
    elif args.command == "phase-transition":
        cmd_phase_transition(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
