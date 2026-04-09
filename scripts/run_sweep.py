import argparse
import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.sweep import run_sweep, enumerate_configs


def main():
    parser = argparse.ArgumentParser(description="Run (i,j) layer duplication sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B",
                        help="Model name or path")
    parser.add_argument("--results", default="results/sweep_results.json",
                        help="Results output path")
    parser.add_argument("--max-math", type=int, default=None,
                        help="Max math questions per config (default: all 16)")
    parser.add_argument("--max-eq", type=int, default=None,
                        help="Max EQ scenarios per config (default: all 16)")
    parser.add_argument("--test", type=int, default=None,
                        help="Only run first N configs (for quick testing)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded: {model.config.num_hidden_layers} layers on {model.device}")

    with open("data/math_probe.json") as f:
        math_questions = json.load(f)
    with open("data/eq_probe.json") as f:
        eq_scenarios = json.load(f)

    configs = None
    if args.test:
        all_configs = enumerate_configs(model.config.num_hidden_layers)
        configs = all_configs[:args.test]
        print(f"Test mode: running {args.test}/{len(all_configs)} configs")

    math_count = args.max_math or len(math_questions)
    eq_count = args.max_eq or len(eq_scenarios)
    print(f"Probes: {math_count} math, {eq_count} EQ per config")

    results = run_sweep(
        model, tokenizer,
        math_questions, eq_scenarios,
        args.results,
        max_math=args.max_math,
        max_eq=args.max_eq,
        configs=configs,
    )

    # Print top 5 by combined score
    baseline = results.get("baseline", {})
    b_math = baseline.get("math_score", 0)
    b_eq = baseline.get("eq_score", 0)

    print(f"\nBaseline: math={b_math:.4f}, eq={b_eq:.4f}")
    print("\nTop 5 configs by combined delta (math_delta + eq_delta):")

    scored = []
    for key, r in results.items():
        if key == "baseline":
            continue
        math_delta = r["math_score"] - b_math
        eq_delta = r["eq_score"] - b_eq
        scored.append((key, r["i"], r["j"], math_delta, eq_delta, math_delta + eq_delta))

    scored.sort(key=lambda x: x[5], reverse=True)
    for key, i, j, md, ed, combined in scored[:5]:
        print(f"  ({i},{j}): math={md:+.4f} eq={ed:+.4f} combined={combined:+.4f}")


if __name__ == "__main__":
    main()
