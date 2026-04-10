"""Run fine-grained threshold sweep for phase transition analysis.

Iterates over thresholds, runs collect_traces.py for each, and produces
a summary JSON with accuracy and mean iterations per threshold.

This script is designed for RunPod — it loads the model once and runs
all thresholds in sequence.

Usage:
    python scripts/phase_transition_sweep.py \
        --model Qwen/Qwen2.5-7B \
        --data data/math_probe_expanded.json \
        --output results/phase_sweep_math.json \
        --thresholds 0.50 0.60 0.70 0.80 0.85 0.90 0.92 0.94 0.95 0.96 0.97 0.98 0.99 \
        --max-iters 4 --block-i 15 --block-j 20
"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference.convergence_tracer import ConvergenceTracer
from src.inference.constrained import build_json_processor
from src.evaluation.math_eval import score_answer


PROMPT_TEMPLATE = (
    'Respond with JSON: {{"reasoning": "<your work>", "answer": <number>}}\n\n'
    'Question: {question}\n'
)


def extract_answer(generated):
    try:
        obj = json.loads(generated)
        return float(obj["answer"])
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def run_at_threshold(model, tokenizer, json_processor, probes,
                     threshold, max_iters, block_i, block_j):
    """Run all probes at a given threshold, return summary dict."""
    tracer = ConvergenceTracer(
        model, block_i, block_j,
        threshold=threshold, max_iterations=max_iters,
    )

    correct = 0
    total_score = 0.0
    total_iters = 0.0
    total_tokens = 0

    for probe in probes:
        prompt = PROMPT_TEMPLATE.format(question=probe["question"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text=probe["question"],
            score=0.0,
            max_new_tokens=256,
            logits_processor=json_processor,
        )

        predicted = extract_answer(trace.generated)
        score = score_answer(predicted, probe["answer"])
        if score > 0.99:
            correct += 1
        total_score += score

        summary = trace.summary()
        total_iters += summary["avg_iterations"]
        total_tokens += summary["total_tokens"]

    n = len(probes)
    return {
        "threshold": threshold,
        "accuracy": correct / n,
        "num_correct": correct,
        "mean_score": total_score / n,
        "mean_avg_iterations": total_iters / n,
        "mean_tokens": total_tokens / n,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase transition threshold sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data", default="data/math_probe_expanded.json")
    parser.add_argument("--output", default="results/phase_sweep_math.json")
    parser.add_argument("--block-i", type=int, default=15)
    parser.add_argument("--block-j", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0.50, 0.60, 0.70, 0.80, 0.85,
                                 0.90, 0.92, 0.94, 0.95, 0.96,
                                 0.97, 0.98, 0.99])
    parser.add_argument("--max-probes", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto",
    )

    with open(args.data) as f:
        probes = json.load(f)
    if args.max_probes:
        probes = probes[:args.max_probes]
    print(f"Loaded {len(probes)} probes")

    print("Building JSON logits processor...")
    json_processor = build_json_processor(model, tokenizer)

    results = []
    for threshold in sorted(args.thresholds):
        t0 = time.time()
        print(f"\n=== Threshold={threshold:.2f} ===")
        result = run_at_threshold(
            model, tokenizer, json_processor, probes,
            threshold, args.max_iters, args.block_i, args.block_j,
        )
        elapsed = time.time() - t0
        result["elapsed_s"] = round(elapsed, 1)
        results.append(result)
        print(f"  accuracy={result['accuracy']:.3f} "
              f"mean_iters={result['mean_avg_iterations']:.2f} "
              f"[{elapsed:.1f}s]")

    output = {
        "model": args.model,
        "data": args.data,
        "block": [args.block_i, args.block_j],
        "max_iterations": args.max_iters,
        "num_probes": len(probes),
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")

    # Print summary table
    print(f"\n{'Threshold':>10s}  {'Accuracy':>8s}  {'Iters':>6s}  {'Time':>6s}")
    for r in results:
        print(f"  {r['threshold']:>8.2f}  {r['accuracy']:>8.3f}  "
              f"{r['mean_avg_iterations']:>6.2f}  {r['elapsed_s']:>6.1f}s")


if __name__ == "__main__":
    main()
