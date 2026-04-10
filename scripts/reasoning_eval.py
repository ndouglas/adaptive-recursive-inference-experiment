"""Evaluate adaptive halting on multi-step reasoning probes.

Tests the difficulty hypothesis: harder problems should benefit more from
extra iterations. Compares baseline vs adaptive across step-count tiers.
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference.adaptive_loop import AdaptiveLoop
from src.evaluation.math_eval import run_math_eval


def score_answer(generated, expected):
    """Score a generated answer against expected (partial credit via relative error)."""
    numbers = re.findall(r'-?\d+\.?\d*', generated)
    if not numbers:
        return 0.0, None
    predicted = float(numbers[0])
    if expected == 0:
        return (1.0 if predicted == 0 else 0.0), predicted
    rel_error = abs(predicted - expected) / abs(expected)
    return max(0.0, 1.0 - rel_error), predicted


def eval_baseline(model, tokenizer, probes):
    """Run probes through the unmodified model."""
    results = []
    for p in probes:
        prompt = f"Question: {p['question']}\nAnswer (number only):"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        generated = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)
        score, predicted = score_answer(generated, p["answer"])
        results.append({
            "question": p["question"],
            "expected": p["answer"],
            "predicted": predicted,
            "generated": generated.strip()[:80],
            "score": score,
            "steps": p["steps"],
            "category": p["category"],
        })
    return results


def eval_adaptive(loop, tokenizer, probes):
    """Run probes through the adaptive loop."""
    results = []
    for p in probes:
        prompt = f"Question: {p['question']}\nAnswer (number only):"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(loop.model.device)
        output_ids = loop.generate(input_ids, tokenizer, max_new_tokens=32)
        generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:],
                                     skip_special_tokens=True)
        score, predicted = score_answer(generated, p["answer"])
        diag = loop.diagnostics_summary()
        results.append({
            "question": p["question"],
            "expected": p["answer"],
            "predicted": predicted,
            "generated": generated.strip()[:80],
            "score": score,
            "steps": p["steps"],
            "category": p["category"],
            "avg_iters": diag["avg_iterations"],
            "max_iters_used": diag["max_iterations"],
        })
    return results


def summarize_by_tier(results, tiers):
    """Group results by step-count tiers and compute per-tier stats."""
    summary = {}
    for tier_name, step_range in tiers.items():
        tier_results = [r for r in results if r["steps"] in step_range]
        if not tier_results:
            continue
        scores = [r["score"] for r in tier_results]
        avg_iters = [r.get("avg_iters", 0) for r in tier_results]
        summary[tier_name] = {
            "count": len(tier_results),
            "avg_score": sum(scores) / len(scores),
            "avg_iters": sum(avg_iters) / len(avg_iters) if any(avg_iters) else None,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Reasoning probe evaluation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--block-i", type=int, default=25)
    parser.add_argument("--block-j", type=int, default=27)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--output", default="results/reasoning_eval.json")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )

    with open("data/reasoning_probe.json") as f:
        probes = json.load(f)

    tiers = {
        "easy (2 steps)": {2},
        "medium (3 steps)": {3},
        "hard (4-5 steps)": {4, 5},
    }

    # --- Baseline ---
    print("\n=== Baseline ===")
    t0 = time.time()
    baseline_results = eval_baseline(model, tokenizer, probes)
    baseline_time = time.time() - t0
    baseline_overall = sum(r["score"] for r in baseline_results) / len(baseline_results)
    print(f"  Overall: {baseline_overall:.4f} [{baseline_time:.1f}s]")

    baseline_tiers = summarize_by_tier(baseline_results, tiers)
    for tier, stats in baseline_tiers.items():
        print(f"  {tier}: {stats['avg_score']:.4f} ({stats['count']} probes)")

    # --- Adaptive ---
    print(f"\n=== Adaptive (threshold={args.threshold}, max_iters={args.max_iters}, "
          f"block=({args.block_i},{args.block_j})) ===")
    loop = AdaptiveLoop(model, args.block_i, args.block_j,
                        threshold=args.threshold, max_iterations=args.max_iters)
    t0 = time.time()
    adaptive_results = eval_adaptive(loop, tokenizer, probes)
    adaptive_time = time.time() - t0
    adaptive_overall = sum(r["score"] for r in adaptive_results) / len(adaptive_results)
    print(f"  Overall: {adaptive_overall:.4f} [{adaptive_time:.1f}s]")

    adaptive_tiers = summarize_by_tier(adaptive_results, tiers)
    for tier, stats in adaptive_tiers.items():
        print(f"  {tier}: {stats['avg_score']:.4f} ({stats['count']} probes, "
              f"avg {stats['avg_iters']:.1f} iters)")

    # --- Comparison ---
    print(f"\n=== Comparison ===")
    print(f"{'Tier':<20s}  {'Baseline':>8s}  {'Adaptive':>8s}  {'Delta':>8s}  {'Avg It':>8s}")
    print(f"{'Overall':<20s}  {baseline_overall:8.4f}  {adaptive_overall:8.4f}  "
          f"{adaptive_overall - baseline_overall:+8.4f}  {'':>8s}")
    for tier in tiers:
        if tier not in baseline_tiers or tier not in adaptive_tiers:
            continue
        bl = baseline_tiers[tier]["avg_score"]
        ad = adaptive_tiers[tier]["avg_score"]
        ai = adaptive_tiers[tier]["avg_iters"]
        delta = ad - bl
        print(f"{tier:<20s}  {bl:8.4f}  {ad:8.4f}  {delta:+8.4f}  {ai:8.1f}")

    # --- Per-category breakdown ---
    print(f"\n=== By Category ===")
    categories = sorted(set(r["category"] for r in probes))
    print(f"{'Category':<25s}  {'Baseline':>8s}  {'Adaptive':>8s}  {'Delta':>8s}  {'Avg It':>8s}")
    for cat in categories:
        bl_cat = [r for r in baseline_results if r["category"] == cat]
        ad_cat = [r for r in adaptive_results if r["category"] == cat]
        bl_score = sum(r["score"] for r in bl_cat) / len(bl_cat)
        ad_score = sum(r["score"] for r in ad_cat) / len(ad_cat)
        ai = sum(r["avg_iters"] for r in ad_cat) / len(ad_cat)
        print(f"{cat:<25s}  {bl_score:8.4f}  {ad_score:8.4f}  "
              f"{ad_score - bl_score:+8.4f}  {ai:8.1f}")

    # --- Save ---
    output = {
        "model": args.model,
        "block": (args.block_i, args.block_j),
        "threshold": args.threshold,
        "max_iterations": args.max_iters,
        "baseline": {"results": baseline_results, "overall": baseline_overall,
                      "tiers": baseline_tiers, "time_s": round(baseline_time, 1)},
        "adaptive": {"results": adaptive_results, "overall": adaptive_overall,
                      "tiers": adaptive_tiers, "time_s": round(adaptive_time, 1)},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
