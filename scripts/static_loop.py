"""Static loop experiment: run the circuit block 1-8 times and measure convergence.

For each iteration count, records:
- Cosine similarity trajectory between consecutive exit states
- Math and EQ benchmark scores
- Per-prompt convergence data

Uses the 1.5B model locally (MPS) with block (15,20) from the 7B sweep.
For the 1.5B model, (25,27) was the best combined config, but we use (15,20)
to validate the 7B finding and because it'll be our target for adaptive halting.
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference.layer_duplicator import LayerDuplicator
from src.inference.iteration_monitor import IterationMonitor
from src.evaluation.math_eval import run_math_eval
from src.evaluation.eq_eval import run_eq_eval


def run_monitored_generation(model, tokenizer, prompt, block_i, block_j, iterations):
    """Run a single generation with the iteration monitor attached."""
    dup = LayerDuplicator(block_i, block_j, iterations=iterations)

    with dup.apply(model) as modified:
        monitor = IterationMonitor(block_i, block_j, iterations)
        monitor.attach(modified)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = modified.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=1.0,
            )

        monitor.detach()

    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated, monitor.summary()


def main():
    parser = argparse.ArgumentParser(description="Static loop convergence experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--block-i", type=int, default=15)
    parser.add_argument("--block-j", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=7,
                        help="Max extra iterations (total passes = max-iters + 1)")
    parser.add_argument("--max-math", type=int, default=8)
    parser.add_argument("--max-eq", type=int, default=8)
    parser.add_argument("--output", default="results/static_loop.json")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    num_layers = model.config.num_hidden_layers
    print(f"Loaded: {num_layers} layers, block ({args.block_i},{args.block_j})")

    with open("data/math_probe.json") as f:
        math_questions = json.load(f)
    with open("data/eq_probe.json") as f:
        eq_scenarios = json.load(f)

    math_subset = math_questions[:args.max_math]
    eq_subset = eq_scenarios[:args.max_eq]

    # --- Baseline (0 extra iterations = normal forward pass) ---
    print("\n=== Baseline (0 extra iterations) ===")
    _, math_baseline = run_math_eval(model, tokenizer, math_subset, verbose=False)
    _, eq_baseline = run_eq_eval(model, tokenizer, eq_subset, verbose=False)
    print(f"  math={math_baseline:.4f}  eq={eq_baseline:.4f}")

    results = {
        "model": args.model,
        "block_i": args.block_i,
        "block_j": args.block_j,
        "baseline": {"math_score": math_baseline, "eq_score": eq_baseline},
        "iterations": {},
    }

    # --- Test prompts for convergence monitoring ---
    test_prompts = [
        "What is 23 times 47?",
        "What is the square root of 2025?",
        "Explain why the sky is blue in one sentence.",
        "What is the capital of France?",
    ]

    # --- Sweep iteration counts ---
    for iters in range(1, args.max_iters + 1):
        total_passes = iters + 1
        print(f"\n=== {total_passes} passes ({iters} extra iterations) ===")
        t0 = time.time()

        # Benchmark scores
        dup = LayerDuplicator(args.block_i, args.block_j, iterations=iters)
        with dup.apply(model) as modified:
            _, math_score = run_math_eval(modified, tokenizer, math_subset, verbose=False)
            _, eq_score = run_eq_eval(modified, tokenizer, eq_subset, verbose=False)

        elapsed = time.time() - t0
        math_delta = math_score - math_baseline
        eq_delta = eq_score - eq_baseline
        print(f"  math={math_score:.4f} ({math_delta:+.4f})  eq={eq_score:.4f} ({eq_delta:+.4f})  [{elapsed:.1f}s]")

        # Convergence monitoring on test prompts (prompt-only, first forward call)
        prompt_data = []
        for prompt in test_prompts:
            generated, summary = run_monitored_generation(
                model, tokenizer, prompt,
                args.block_i, args.block_j, iters
            )
            prompt_data.append({
                "prompt": prompt,
                "generated": generated,
                "summary": summary,
            })
            traj = summary["exit_trajectory"]
            traj_str = " → ".join(f"{s:.6f}" for s in traj)
            final = summary.get("final_similarity")
            final_str = f"  final={final:.6f}" if final is not None else ""
            print(f"  [{prompt[:40]:40s}] {traj_str}{final_str}")

        results["iterations"][str(iters)] = {
            "total_passes": total_passes,
            "math_score": math_score,
            "eq_score": eq_score,
            "elapsed_s": round(elapsed, 1),
            "prompts": prompt_data,
        }

    # --- Save ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # --- Summary table ---
    print("\n=== Summary ===")
    print(f"{'Passes':>8s}  {'Math':>8s}  {'EQ':>8s}  {'Math Δ':>8s}  {'EQ Δ':>8s}")
    print(f"{'1 (base)':>8s}  {math_baseline:8.4f}  {eq_baseline:8.4f}  {0:+8.4f}  {0:+8.4f}")
    for iters in range(1, args.max_iters + 1):
        r = results["iterations"][str(iters)]
        md = r["math_score"] - math_baseline
        ed = r["eq_score"] - eq_baseline
        print(f"{r['total_passes']:8d}  {r['math_score']:8.4f}  {r['eq_score']:8.4f}  {md:+8.4f}  {ed:+8.4f}")


if __name__ == "__main__":
    main()
