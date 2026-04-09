import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.analysis.heatmaps import generate_heatmaps


def main():
    parser = argparse.ArgumentParser(description="Generate heatmaps from sweep results")
    parser.add_argument("--results", default="results/sweep_results.json",
                        help="Path to sweep results JSON")
    parser.add_argument("--output", default="plots",
                        help="Output directory for heatmap PNGs")
    parser.add_argument("--num-layers", type=int, default=28,
                        help="Number of layers in the model")
    args = parser.parse_args()

    generate_heatmaps(args.results, args.output, args.num_layers)


if __name__ == "__main__":
    main()
