"""Analyze convergence traces for correctness correlation.

Loads trace JSON from collect_traces.py and runs statistical analysis:
correlations, ROC AUC, bootstrap CIs, stratified by category/difficulty.

Usage:
    python scripts/analyze_traces.py results/traces_7b_math.json
    python scripts/analyze_traces.py results/traces_7b_math.json --group-by difficulty
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.analysis.convergence_stats import ConvergenceAnalyzer


def print_correlations(corr, label=""):
    """Pretty-print correlation results."""
    if label:
        print(f"\n=== {label} ===")
    print(f"  {'Metric':<25s}  {'r':>8s}  {'p':>10s}  {'CI 95%':>20s}")
    for metric, stats in corr.items():
        if metric == "n":
            continue
        ci = stats.get("ci_95")
        ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
        print(f"  {metric:<25s}  {stats['r']:+8.4f}  {stats['p']:10.4f}  {ci_str:>20s}")


def print_roc(roc, label=""):
    """Pretty-print ROC AUC results."""
    if label:
        print(f"\n=== {label} ===")
    print(f"  {'Metric':<25s}  {'AUC':>8s}")
    for metric, stats in roc.items():
        print(f"  {metric:<25s}  {stats['auc']:8.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze convergence traces")
    parser.add_argument("trace_file", help="Path to traces JSON from collect_traces.py")
    parser.add_argument("--group-by", default="category",
                        help="Field to stratify by (category, difficulty)")
    parser.add_argument("--output", default=None,
                        help="Save analysis results to JSON")
    args = parser.parse_args()

    with open(args.trace_file) as f:
        data = json.load(f)

    traces = data["traces"]
    print(f"Loaded {len(traces)} traces from {args.trace_file}")
    print(f"  Model: {data.get('model', 'unknown')}")
    print(f"  Threshold: {data.get('threshold', '?')}")
    print(f"  Accuracy: {data.get('num_correct', '?')}/{data.get('num_probes', '?')}")

    analyzer = ConvergenceAnalyzer(traces)

    # Overall correlations with bootstrap CIs
    corr = analyzer.compute_correlations(bootstrap_n=1000)
    print_correlations(corr, "Correlations (overall)")

    # ROC AUC
    roc = analyzer.compute_roc_auc()
    print_roc(roc, "ROC AUC (overall)")

    # Stratified
    strat = analyzer.stratified_correlations(group_key=args.group_by)
    if strat:
        print(f"\n=== Stratified by {args.group_by} ===")
        for group, group_corr in strat.items():
            n = group_corr.pop("n", "?")
            print(f"\n  --- {group} (n={n}) ---")
            for metric, stats in group_corr.items():
                print(f"    {metric:<25s}  r={stats['r']:+.4f}  p={stats['p']:.4f}")

    # Summary table
    summary = analyzer.summary_table()
    print(f"\n=== Headline Numbers ===")
    overall = summary["overall"]
    print(f"  N={overall['n']}  accuracy={overall['accuracy']:.3f}  "
          f"mean_score={overall['mean_score']:.4f}")
    for metric, roc_stat in overall["roc_auc"].items():
        corr_stat = overall["correlations"][metric]
        ci = corr_stat.get("ci_95", [None, None])
        ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci[0] is not None else ""
        print(f"  {metric:<25s}  r={corr_stat['r']:+.4f}  AUC={roc_stat['auc']:.4f}  {ci_str}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
