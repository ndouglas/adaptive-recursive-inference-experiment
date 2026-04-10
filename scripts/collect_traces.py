"""Collect convergence traces from adaptive inference.

Runs math probes through ConvergenceTracer with constrained JSON decoding,
recording full per-token convergence data + correctness labels.

Usage:
    python scripts/collect_traces.py --model Qwen/Qwen2.5-7B \
        --data data/math_probe_expanded.json \
        --output results/traces_7b_math.json \
        --threshold 0.80 --max-iters 4 \
        --block-i 15 --block-j 20
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
    """Extract numeric answer from constrained JSON output."""
    try:
        obj = json.loads(generated)
        return float(obj["answer"])
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None


def collect_traces(tracer, tokenizer, probes, json_processor, verbose=True):
    """Run probes through tracer, scoring each and returning trace dicts."""
    all_traces = []
    for i, probe in enumerate(probes):
        prompt = PROMPT_TEMPLATE.format(question=probe["question"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(
            tracer.loop.model.device
        )

        # Generate with tracing (score=0 placeholder, updated after)
        trace = tracer.trace_generation(
            input_ids=input_ids,
            tokenizer=tokenizer,
            prompt_text=probe["question"],
            score=0.0,
            max_new_tokens=256,
            logits_processor=json_processor,
        )

        # Score the output
        predicted = extract_answer(trace.generated)
        score = score_answer(predicted, probe["answer"])
        trace.score = score  # update placeholder

        trace_dict = trace.to_dict()
        trace_dict["expected"] = probe["answer"]
        trace_dict["predicted"] = predicted
        trace_dict["correct"] = score > 0.99
        if "category" in probe:
            trace_dict["category"] = probe["category"]
        if "difficulty" in probe:
            trace_dict["difficulty"] = probe["difficulty"]

        all_traces.append(trace_dict)

        if verbose:
            summary = trace.summary()
            status = "OK" if score > 0.99 else f"WRONG ({predicted})"
            print(
                f"  [{i+1}/{len(probes)}] {probe['question'][:50]:50s} "
                f"score={score:.2f} iters={summary['avg_iterations']:.1f} "
                f"{status}"
            )

    return all_traces


def main():
    parser = argparse.ArgumentParser(description="Collect convergence traces")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data", default="data/math_probe_expanded.json")
    parser.add_argument("--output", default="results/traces_7b_math.json")
    parser.add_argument("--block-i", type=int, default=15)
    parser.add_argument("--block-j", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--max-probes", type=int, default=None,
                        help="Limit number of probes (for testing)")
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
    print(f"Loaded {len(probes)} probes from {args.data}")

    print("Building JSON logits processor...")
    json_processor = build_json_processor(model, tokenizer)

    tracer = ConvergenceTracer(
        model, args.block_i, args.block_j,
        threshold=args.threshold, max_iterations=args.max_iters,
    )

    print(f"\nCollecting traces (threshold={args.threshold}, "
          f"max_iters={args.max_iters}, block=({args.block_i},{args.block_j}))...\n")
    t0 = time.time()
    traces = collect_traces(tracer, tokenizer, probes, json_processor)
    elapsed = time.time() - t0

    # Summary
    scores = [t["score"] for t in traces]
    correct = sum(1 for t in traces if t["correct"])
    avg_iters = sum(
        t["summary"]["avg_iterations"] for t in traces
    ) / len(traces)

    print(f"\n=== Summary ===")
    print(f"  Probes: {len(traces)}")
    print(f"  Correct: {correct}/{len(traces)} ({correct/len(traces)*100:.1f}%)")
    print(f"  Mean score: {sum(scores)/len(scores):.4f}")
    print(f"  Mean avg_iterations: {avg_iters:.2f}")
    print(f"  Elapsed: {elapsed:.1f}s")

    output = {
        "model": args.model,
        "block": [args.block_i, args.block_j],
        "threshold": args.threshold,
        "max_iterations": args.max_iters,
        "data_source": args.data,
        "num_probes": len(traces),
        "num_correct": correct,
        "mean_score": sum(scores) / len(scores),
        "mean_avg_iterations": avg_iters,
        "elapsed_s": round(elapsed, 1),
        "traces": traces,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
