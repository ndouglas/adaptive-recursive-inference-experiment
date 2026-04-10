"""Adaptive halting evaluation with threshold sensitivity analysis.

Runs the adaptive loop across a range of thresholds, recording:
- Math and EQ benchmark scores
- Average iterations per token
- Per-prompt halting behavior
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference.adaptive_loop import AdaptiveLoop
from src.evaluation.math_eval import run_math_eval
from src.evaluation.eq_eval import run_eq_eval


def run_adaptive_math_eval(loop, tokenizer, questions, verbose=False):
    """Run math eval using the adaptive loop instead of model.generate()."""
    import re
    results = []
    total_score = 0.0
    total_iters = 0
    total_tokens = 0

    for q in questions:
        prompt = f"Question: {q['question']}\nAnswer (number only):"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(loop.model.device)

        output_ids = loop.generate(input_ids, tokenizer, max_new_tokens=32)
        generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        diag = loop.diagnostics_summary()
        total_iters += sum(loop.token_iterations)
        total_tokens += len(loop.token_iterations)

        # Score: extract answer number (handles chain-of-thought output)
        expected = q["answer"]
        predicted = None
        for pattern in [
            r'(?:answer|result|equals|=)\s*[:is]*\s*(-?\d[\d,]*\.?\d*)',
            r'(-?\d[\d,]*\.?\d*)\s*$',
        ]:
            match = re.search(pattern, generated, re.IGNORECASE)
            if match:
                predicted = float(match.group(1).replace(',', ''))
                break
        if predicted is None:
            numbers = re.findall(r'-?\d[\d,]*\.?\d*', generated)
            if numbers:
                predicted = float(numbers[-1].replace(',', ''))

        if predicted is not None:
            if expected == 0:
                score = 1.0 if predicted == 0 else 0.0
            else:
                rel_error = abs(predicted - expected) / abs(expected)
                score = max(0.0, 1.0 - rel_error)
        else:
            score = 0.0

        results.append({
            "question": q["question"],
            "expected": expected,
            "predicted": predicted,
            "generated": generated.strip(),
            "score": score,
            "avg_iters": diag["avg_iterations"],
        })
        total_score += score

        if verbose:
            print(f"  [{score:.2f}] {q['question'][:50]:50s} → {generated.strip()[:30]:30s} (avg {diag['avg_iterations']:.1f} iters)")

    return results, total_score / len(questions), total_iters / max(total_tokens, 1)


def run_adaptive_eq_eval(loop, tokenizer, scenarios, verbose=False):
    """Run EQ eval using the adaptive loop."""
    from scipy.stats import pearsonr
    results = []
    total_score = 0.0
    total_iters = 0
    total_tokens = 0

    for scenario in scenarios:
        prompt = (
            f"Scenario: {scenario['scenario']}\n"
            f"Rate each emotion from 0 to 100:\n"
            f"- {scenario['emotions'][0]}:\n"
            f"- {scenario['emotions'][1]}:\n"
            f"- {scenario['emotions'][2]}:\n"
            f"- {scenario['emotions'][3]}:\n"
            f"Respond with just four numbers, one per line."
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(loop.model.device)
        output_ids = loop.generate(input_ids, tokenizer, max_new_tokens=64)
        generated = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        diag = loop.diagnostics_summary()
        total_iters += sum(loop.token_iterations)
        total_tokens += len(loop.token_iterations)

        # Parse and score
        import re
        numbers = re.findall(r'\b(\d{1,3})\b', generated)
        ratings = []
        for n in numbers:
            val = int(n)
            if 0 <= val <= 100:
                ratings.append(val)
            if len(ratings) == 4:
                break

        predicted = ratings if len(ratings) == 4 else None
        reference = scenario["reference"]

        if predicted is None or len(set(predicted)) == 1 or len(set(reference)) == 1:
            score = 0.0
        else:
            r, _ = pearsonr(predicted, reference)
            score = max(0.0, r)

        results.append({
            "scenario": scenario["scenario"][:60],
            "score": score,
            "avg_iters": diag["avg_iterations"],
        })
        total_score += score

        if verbose:
            print(f"  [{score:.2f}] {scenario['scenario'][:50]:50s} (avg {diag['avg_iterations']:.1f} iters)")

    return results, total_score / len(scenarios), total_iters / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="Adaptive halting threshold sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--block-i", type=int, default=25)
    parser.add_argument("--block-j", type=int, default=27)
    parser.add_argument("--max-iters", type=int, default=4,
                        help="Max iterations safety cap")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999],
                        help="Thresholds to test")
    parser.add_argument("--max-math", type=int, default=8)
    parser.add_argument("--max-eq", type=int, default=8)
    parser.add_argument("--output", default="results/adaptive_eval.json")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    with open("data/math_probe.json") as f:
        math_questions = json.load(f)[:args.max_math]
    with open("data/eq_probe.json") as f:
        eq_scenarios = json.load(f)[:args.max_eq]

    # Baseline (no duplication)
    print("\n=== Baseline ===")
    _, math_baseline = run_math_eval(model, tokenizer, math_questions, verbose=False)
    _, eq_baseline = run_eq_eval(model, tokenizer, eq_scenarios, verbose=False)
    print(f"  math={math_baseline:.4f}  eq={eq_baseline:.4f}")

    results = {
        "model": args.model,
        "block_i": args.block_i,
        "block_j": args.block_j,
        "max_iterations": args.max_iters,
        "baseline": {"math_score": math_baseline, "eq_score": eq_baseline},
        "thresholds": {},
    }

    for threshold in sorted(args.thresholds):
        print(f"\n=== Threshold={threshold:.3f}, max_iters={args.max_iters} ===")
        t0 = time.time()

        loop = AdaptiveLoop(model, args.block_i, args.block_j,
                            threshold=threshold, max_iterations=args.max_iters)

        math_results, math_score, math_avg_iters = run_adaptive_math_eval(
            loop, tokenizer, math_questions, verbose=True)
        eq_results, eq_score, eq_avg_iters = run_adaptive_eq_eval(
            loop, tokenizer, eq_scenarios, verbose=True)

        elapsed = time.time() - t0
        avg_iters = (math_avg_iters + eq_avg_iters) / 2

        math_delta = math_score - math_baseline
        eq_delta = eq_score - eq_baseline
        print(f"  math={math_score:.4f} ({math_delta:+.4f})  eq={eq_score:.4f} ({eq_delta:+.4f})")
        print(f"  avg_iters={avg_iters:.2f}  [{elapsed:.1f}s]")

        results["thresholds"][str(threshold)] = {
            "threshold": threshold,
            "math_score": math_score,
            "eq_score": eq_score,
            "avg_iterations": avg_iters,
            "math_avg_iters": math_avg_iters,
            "eq_avg_iters": eq_avg_iters,
            "elapsed_s": round(elapsed, 1),
            "math_results": math_results,
            "eq_results": eq_results,
        }

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'Threshold':>10s}  {'Math':>8s}  {'EQ':>8s}  {'Math D':>8s}  {'EQ D':>8s}  {'Avg It':>8s}")
    for threshold in sorted(args.thresholds):
        r = results["thresholds"][str(threshold)]
        md = r["math_score"] - math_baseline
        ed = r["eq_score"] - eq_baseline
        print(f"{threshold:10.3f}  {r['math_score']:8.4f}  {r['eq_score']:8.4f}  "
              f"{md:+8.4f}  {ed:+8.4f}  {r['avg_iterations']:8.2f}")


if __name__ == "__main__":
    main()
