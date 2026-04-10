"""Visualize reasoning probe results: baseline vs adaptive by difficulty and category."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/reasoning_eval.json")
    parser.add_argument("--output-dir", default="plots/reasoning")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results) as f:
        data = json.load(f)

    bl_results = data["baseline"]["results"]
    ad_results = data["adaptive"]["results"]
    threshold = data["threshold"]
    block_i, block_j = data["block"]

    # --- Plot 1: Score by difficulty tier ---
    tiers = [
        ("Easy\n(2 steps)", {2}),
        ("Medium\n(3 steps)", {3}),
        ("Hard\n(4-5 steps)", {4, 5}),
        ("Overall", {2, 3, 4, 5}),
    ]

    tier_names = []
    bl_scores = []
    ad_scores = []
    ad_iters = []

    for name, steps in tiers:
        bl_tier = [r for r in bl_results if r["steps"] in steps]
        ad_tier = [r for r in ad_results if r["steps"] in steps]
        tier_names.append(name)
        bl_scores.append(sum(r["score"] for r in bl_tier) / len(bl_tier))
        ad_scores.append(sum(r["score"] for r in ad_tier) / len(ad_tier))
        ad_iters.append(sum(r["avg_iters"] for r in ad_tier) / len(ad_tier))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tier_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, bl_scores, width, label="Baseline", color="tab:blue", alpha=0.8)
    bars2 = ax.bar(x + width/2, ad_scores, width, label=f"Adaptive (θ={threshold})", color="tab:orange", alpha=0.8)

    # Delta labels above adaptive bars
    for i, (bl, ad) in enumerate(zip(bl_scores, ad_scores)):
        delta = ad - bl
        color = "green" if delta > 0 else "red"
        ax.annotate(f"{delta:+.3f}", (x[i] + width/2, ad + 0.01),
                    ha="center", fontsize=10, fontweight="bold", color=color)

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(tier_names)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")
    ax.set_title(f"Adaptive Halting: Score by Difficulty Tier — Block ({block_i},{block_j})")

    fig.tight_layout()
    path = os.path.join(args.output_dir, "score_by_difficulty.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Plot 2: Delta by category ---
    categories = sorted(set(r["category"] for r in bl_results))
    cat_labels = []
    cat_deltas = []
    cat_iters = []

    for cat in categories:
        bl_cat = [r for r in bl_results if r["category"] == cat]
        ad_cat = [r for r in ad_results if r["category"] == cat]
        bl_s = sum(r["score"] for r in bl_cat) / len(bl_cat)
        ad_s = sum(r["score"] for r in ad_cat) / len(ad_cat)
        cat_labels.append(cat.replace("_", "\n"))
        cat_deltas.append(ad_s - bl_s)
        cat_iters.append(sum(r["avg_iters"] for r in ad_cat) / len(ad_cat))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:green" if d > 0 else "tab:red" for d in cat_deltas]
    bars = ax.bar(cat_labels, cat_deltas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)

    for bar, d in zip(bars, cat_deltas):
        y = bar.get_height()
        ax.annotate(f"{d:+.3f}", (bar.get_x() + bar.get_width()/2, y),
                    ha="center", va="bottom" if y >= 0 else "top",
                    fontsize=10, fontweight="bold")

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel("Score Delta (Adaptive − Baseline)")
    ax.set_title(f"Adaptive Halting: Score Delta by Category — Block ({block_i},{block_j})")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    path = os.path.join(args.output_dir, "delta_by_category.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Plot 3: Per-problem scatter (steps vs delta) ---
    fig, ax = plt.subplots(figsize=(10, 6))

    steps_all = []
    deltas_all = []
    iters_all = []

    for bl, ad in zip(bl_results, ad_results):
        steps_all.append(bl["steps"])
        deltas_all.append(ad["score"] - bl["score"])
        iters_all.append(ad["avg_iters"])

    # Jitter steps for visibility
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(steps_all))
    steps_jittered = np.array(steps_all, dtype=float) + jitter

    scatter = ax.scatter(steps_jittered, deltas_all, c=iters_all, cmap="YlOrRd",
                         s=80, edgecolors="black", linewidth=0.5, zorder=5)
    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)

    # Trend line (mean delta per step count)
    for s in sorted(set(steps_all)):
        tier_deltas = [d for st, d in zip(steps_all, deltas_all) if st == s]
        ax.plot(s, np.mean(tier_deltas), "D", color="black", markersize=10, zorder=10)

    ax.set_xlabel("Problem Steps")
    ax.set_ylabel("Score Delta (Adaptive − Baseline)")
    ax.set_title(f"Per-Problem Score Delta vs Difficulty — Block ({block_i},{block_j})")
    ax.set_xticks(sorted(set(steps_all)))
    ax.grid(True, alpha=0.2)

    cbar = fig.colorbar(scatter, ax=ax, label="Avg Iterations Used")

    fig.tight_layout()
    path = os.path.join(args.output_dir, "delta_vs_steps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
