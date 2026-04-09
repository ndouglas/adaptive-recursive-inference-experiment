"""Remote job runner for RunPod (or any GPU machine).

Wraps the sweep with timing, GPU utilization monitoring, and cost estimation.
Designed to be run on the remote machine after setup_runpod.sh.

Usage:
    python3 scripts/run_remote_job.py --model Qwen/Qwen2.5-7B --results /workspace/sweep_7b.json
    python3 scripts/run_remote_job.py --model Qwen/Qwen2.5-7B --test 5  # quick validation
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.sweep import run_sweep, enumerate_configs


def gpu_summary():
    """Print GPU info and return VRAM in GB."""
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected — this will be very slow.")
        return 0.0
    props = torch.cuda.get_device_properties(0)
    vram_gb = (getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)) / (1024 ** 3)
    print(f"GPU: {props.name} — {vram_gb:.1f} GB VRAM")
    return vram_gb


def estimate_cost(elapsed_s, hourly_rate):
    """Estimate cost from elapsed time and hourly rate."""
    return (elapsed_s / 3600) * hourly_rate


def main():
    parser = argparse.ArgumentParser(description="Run sweep job on remote GPU")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B",
                        help="Model name or path")
    parser.add_argument("--results", default="/workspace/sweep_results.json",
                        help="Results output path")
    parser.add_argument("--max-math", type=int, default=None,
                        help="Max math questions per config")
    parser.add_argument("--max-eq", type=int, default=None,
                        help="Max EQ scenarios per config")
    parser.add_argument("--test", type=int, default=None,
                        help="Only run first N configs (validation mode)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype (default: float16)")
    parser.add_argument("--hourly-rate", type=float, default=0.0,
                        help="GPU hourly rate in USD for cost estimation")
    args = parser.parse_args()

    print("=" * 60)
    print("ARI Sweep Job")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Dtype:   {args.dtype}")
    print(f"Results: {args.results}")
    print()

    vram_gb = gpu_summary()

    # Load model
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"\nLoading model ({args.dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    )
    load_time = time.time() - t0
    num_layers = model.config.num_hidden_layers
    print(f"Loaded in {load_time:.1f}s — {num_layers} layers")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"VRAM used: {allocated:.1f} / {vram_gb:.1f} GB ({allocated/vram_gb*100:.0f}%)")

    # Load probes
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    with open(os.path.join(data_dir, "math_probe.json")) as f:
        math_questions = json.load(f)
    with open(os.path.join(data_dir, "eq_probe.json")) as f:
        eq_scenarios = json.load(f)

    # Configure sweep
    configs = None
    if args.test:
        all_configs = enumerate_configs(num_layers)
        configs = all_configs[:args.test]
        print(f"\nValidation mode: {args.test}/{len(all_configs)} configs")
    else:
        total = len(enumerate_configs(num_layers))
        print(f"\nFull sweep: {total} configs")

    math_count = args.max_math or len(math_questions)
    eq_count = args.max_eq or len(eq_scenarios)
    print(f"Probes: {math_count} math, {eq_count} EQ per config")

    # Run sweep
    print()
    sweep_start = time.time()
    results = run_sweep(
        model, tokenizer,
        math_questions, eq_scenarios,
        args.results,
        max_math=args.max_math,
        max_eq=args.max_eq,
        configs=configs,
    )
    sweep_elapsed = time.time() - sweep_start

    # Summary
    baseline = results.get("baseline", {})
    b_math = baseline.get("math_score", 0)
    b_eq = baseline.get("eq_score", 0)

    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Baseline: math={b_math:.4f}, eq={b_eq:.4f}")
    print(f"Sweep time: {sweep_elapsed/60:.1f} min ({sweep_elapsed:.0f}s)")

    if args.hourly_rate > 0:
        total_time = time.time() - t0
        cost = estimate_cost(total_time, args.hourly_rate)
        print(f"Estimated cost: ${cost:.2f} (at ${args.hourly_rate:.2f}/hr)")

    # Top 5 by combined delta
    scored = []
    for key, r in results.items():
        if key == "baseline":
            continue
        md = r["math_score"] - b_math
        ed = r["eq_score"] - b_eq
        scored.append((r["i"], r["j"], md, ed, md + ed))

    scored.sort(key=lambda x: x[4], reverse=True)
    print("\nTop 5 by combined delta:")
    for i, j, md, ed, combined in scored[:5]:
        print(f"  ({i:2d},{j:2d}): math={md:+.4f} eq={ed:+.4f} combined={combined:+.4f}")

    print(f"\nResults saved to: {args.results}")


if __name__ == "__main__":
    main()
