"""Collect sampling-based uncertainty data.

Generates N samples per probe using temperature sampling with constrained
JSON decoding. Records answer agreement rate as the uncertainty signal.

Usage:
    python scripts/collect_samples.py \
        --model Qwen/Qwen2.5-7B \
        --data data/math_probe_expanded.json \
        --output results/samples_7b_math.json \
        --num-samples 8 --temperature 0.7
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


def collect_samples_for_probe(model, tokenizer, json_processor, probe,
                               num_samples, temperature):
    """Generate num_samples outputs for a single probe, return result dict."""
    prompt = PROMPT_TEMPLATE.format(question=probe["question"])
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Greedy (baseline) answer first
    json_processor.reset()
    with torch.no_grad():
        greedy_out = model.generate(
            input_ids, max_new_tokens=256, do_sample=False,
            logits_processor=LogitsProcessorList([json_processor]),
        )
    greedy_text = tokenizer.decode(
        greedy_out[0, input_ids.shape[1]:], skip_special_tokens=True
    )
    greedy_answer = extract_answer(greedy_text)
    greedy_correct = score_answer(greedy_answer, probe["answer"]) > 0.99

    # Sampled outputs
    samples = []
    for _ in range(num_samples):
        json_processor.reset()
        with torch.no_grad():
            sample_out = model.generate(
                input_ids, max_new_tokens=256,
                do_sample=True, temperature=temperature, top_p=0.95,
                logits_processor=LogitsProcessorList([json_processor]),
            )
        sample_text = tokenizer.decode(
            sample_out[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        answer = extract_answer(sample_text)
        samples.append({"generated": sample_text, "answer": answer})

    # Compute agreement
    valid_answers = [s["answer"] for s in samples if s["answer"] is not None]
    if valid_answers:
        counter = Counter(valid_answers)
        majority_answer, majority_count = counter.most_common(1)[0]
        agreement = majority_count / len(valid_answers)
        majority_correct = score_answer(majority_answer, probe["answer"]) > 0.99
    else:
        majority_answer = None
        agreement = 0.0
        majority_correct = False

    return {
        "question": probe["question"],
        "expected": probe["answer"],
        "category": probe.get("category", "unknown"),
        "difficulty": probe.get("difficulty", "unknown"),
        "greedy_answer": greedy_answer,
        "greedy_correct": greedy_correct,
        "samples": samples,
        "agreement": agreement,
        "majority_answer": majority_answer,
        "majority_correct": majority_correct,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect sampling-based uncertainty")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data", default="data/math_probe_expanded.json")
    parser.add_argument("--output", default="results/samples_7b_math.json")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
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
    t0 = time.time()
    for i, probe in enumerate(probes):
        result = collect_samples_for_probe(
            model, tokenizer, json_processor, probe,
            args.num_samples, args.temperature,
        )
        results.append(result)
        status = "OK" if result["greedy_correct"] else "WRONG"
        print(f"  [{i+1}/{len(probes)}] agreement={result['agreement']:.2f} "
              f"greedy={status} majority={'OK' if result['majority_correct'] else 'WRONG'}")

        # Incremental save
        output = {
            "model": args.model,
            "data": args.data,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "num_probes": len(results),
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    agreements = [r["agreement"] for r in results]
    greedy_correct = sum(1 for r in results if r["greedy_correct"])
    majority_correct = sum(1 for r in results if r["majority_correct"])

    print(f"\n=== Summary ===")
    print(f"  Probes: {len(results)}")
    print(f"  Greedy correct: {greedy_correct}/{len(results)}")
    print(f"  Majority correct: {majority_correct}/{len(results)}")
    print(f"  Mean agreement: {sum(agreements)/len(agreements):.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
