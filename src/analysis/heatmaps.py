import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def load_sweep_results(results_path):
    """Load sweep results and extract baseline + config scores."""
    with open(results_path) as f:
        data = json.load(f)

    baseline = data["baseline"]
    configs = {k: v for k, v in data.items() if k != "baseline"}
    return baseline, configs


def build_delta_matrix(baseline, configs, metric, num_layers):
    """Build an N×N matrix of score deltas for a given metric.

    Only the upper triangle (j > i) is populated; rest is NaN.
    """
    baseline_score = baseline[metric]
    matrix = np.full((num_layers, num_layers + 1), np.nan)

    for key, result in configs.items():
        i, j = result["i"], result["j"]
        matrix[i, j] = result[metric] - baseline_score

    return matrix


def plot_heatmap(matrix, title, output_path, num_layers, best_ij=None):
    """Plot a single heatmap with diverging colormap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Symmetric color scale around zero
    valid = matrix[~np.isnan(matrix)]
    if len(valid) == 0:
        return
    vmax = max(abs(valid.min()), abs(valid.max()))
    if vmax == 0:
        vmax = 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(
        matrix,
        cmap="RdBu_r",
        norm=norm,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )

    if best_ij is not None:
        i, j = best_ij
        ax.plot(j, i, marker="*", color="gold", markersize=15, markeredgecolor="black")

    ax.set_xlabel("End Layer j")
    ax.set_ylabel("Start Layer i")
    ax.set_title(title)
    ax.set_xticks(range(0, num_layers + 1, 2))
    ax.set_yticks(range(0, num_layers, 2))

    cbar = fig.colorbar(im, ax=ax, label="Score Delta vs Baseline")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def find_best_config(configs, metric):
    """Find the (i, j) config with the highest score for a given metric."""
    best_key = max(configs, key=lambda k: configs[k][metric])
    r = configs[best_key]
    return (r["i"], r["j"])


def generate_heatmaps(results_path, output_dir, num_layers):
    """Generate math, EQ, and combined heatmaps from sweep results."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    baseline, configs = load_sweep_results(results_path)

    # Math delta heatmap
    math_matrix = build_delta_matrix(baseline, configs, "math_score", num_layers)
    best_math = find_best_config(configs, "math_score")
    plot_heatmap(
        math_matrix,
        f"Math Score Delta (baseline={baseline['math_score']:.4f})",
        os.path.join(output_dir, "heatmap_math.png"),
        num_layers,
        best_ij=best_math,
    )

    # EQ delta heatmap
    eq_matrix = build_delta_matrix(baseline, configs, "eq_score", num_layers)
    best_eq = find_best_config(configs, "eq_score")
    plot_heatmap(
        eq_matrix,
        f"EQ Score Delta (baseline={baseline['eq_score']:.4f})",
        os.path.join(output_dir, "heatmap_eq.png"),
        num_layers,
        best_ij=best_eq,
    )

    # Combined heatmap (z-normalized sum)
    math_valid = math_matrix[~np.isnan(math_matrix)]
    eq_valid = eq_matrix[~np.isnan(eq_matrix)]

    math_z = np.where(
        np.isnan(math_matrix), np.nan,
        (math_matrix - math_valid.mean()) / (math_valid.std() or 1)
    )
    eq_z = np.where(
        np.isnan(eq_matrix), np.nan,
        (eq_matrix - eq_valid.mean()) / (eq_valid.std() or 1)
    )
    combined = math_z + eq_z

    # Find best combined
    best_combined_val = np.nanmax(combined)
    best_idx = np.unravel_index(np.nanargmax(combined), combined.shape)
    best_combined = (best_idx[0], best_idx[1])

    plot_heatmap(
        combined,
        "Combined Score Delta (z-normalized math + EQ)",
        os.path.join(output_dir, "heatmap_combined.png"),
        num_layers,
        best_ij=best_combined,
    )

    # Print summary
    print(f"\nBaseline: math={baseline['math_score']:.4f}, eq={baseline['eq_score']:.4f}")
    print(f"Best math:     ({best_math[0]},{best_math[1]})")
    print(f"Best EQ:       ({best_eq[0]},{best_eq[1]})")
    print(f"Best combined: ({best_combined[0]},{best_combined[1]})")

    # Top 10 by combined z-score
    scored = []
    for key, r in configs.items():
        i, j = r["i"], r["j"]
        if not np.isnan(combined[i, j]):
            scored.append((i, j, combined[i, j], r["math_score"], r["eq_score"]))
    scored.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 10 configs by combined z-score:")
    for i, j, z, ms, es in scored[:10]:
        md = ms - baseline["math_score"]
        ed = es - baseline["eq_score"]
        print(f"  ({i:2d},{j:2d}): z={z:+.3f}  math={md:+.4f}  eq={ed:+.4f}")

    return {
        "best_math": best_math,
        "best_eq": best_eq,
        "best_combined": best_combined,
    }
