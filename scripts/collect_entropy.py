"""Collect softmax-entropy uncertainty data.

Runs greedy generation with a custom loop that captures per-token softmax
entropy from the baseline model (no adaptive loop). Entropy is computed
from raw logits before the JSON constraint is applied, giving the model's
native uncertainty signal.

Usage:
    python scripts/collect_entropy.py \
        --model Qwen/Qwen2.5-7B \
        --data data/math_probe_expanded.json \
        --output results/entropy_7b_math.json
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def generate_with_entropy(model, tokenizer, input_ids, json_processor,
                          max_new_tokens=256):
    """Greedy generation capturing per-token entropy from raw logits.

    Returns:
        generated_text: str
        token_entropies: list of float (one per generated token)
    """
    device = next(model.parameters()).device
    current_ids = input_ids.to(device)
    eos_id = tokenizer.eos_token_id
    token_entropies = []

    json_processor.reset()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(current_ids)
            raw_logits = outputs.logits[:, -1, :]

            # Compute entropy from raw (unconstrained) logits
            probs = F.softmax(raw_logits.float(), dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            token_entropies.append(entropy.item())

            # Apply JSON constraint for token selection
            constrained_logits = json_processor(current_ids, raw_logits.clone())
            next_token = constrained_logits.argmax(dim=-1, keepdim=True)

            current_ids = torch.cat([current_ids, next_token], dim=1)

            if next_token.item() == eos_id:
                break

    generated_ids = current_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text, token_entropies


def main():
    parser = argparse.ArgumentParser(description="Collect softmax-entropy uncertainty")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--data", default="data/math_probe_expanded.json")
    parser.add_argument("--output", default="results/entropy_7b_math.json")
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
        prompt = PROMPT_TEMPLATE.format(question=probe["question"])
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        generated, entropies = generate_with_entropy(
            model, tokenizer, input_ids, json_processor,
        )
        answer = extract_answer(generated)
        correct = score_answer(answer, probe["answer"]) > 0.99
        mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0

        result = {
            "question": probe["question"],
            "expected": probe["answer"],
            "category": probe.get("category", "unknown"),
            "difficulty": probe.get("difficulty", "unknown"),
            "generated": generated,
            "answer": answer,
            "correct": correct,
            "mean_entropy": mean_entropy,
            "token_entropies": entropies,
        }
        results.append(result)

        status = "OK" if correct else "WRONG"
        print(f"  [{i+1}/{len(probes)}] entropy={mean_entropy:.3f} {status}")

        # Incremental save
        output = {
            "model": args.model,
            "data": args.data,
            "num_probes": len(results),
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    correct_count = sum(1 for r in results if r["correct"])
    mean_ent = sum(r["mean_entropy"] for r in results) / len(results)

    print(f"\n=== Summary ===")
    print(f"  Probes: {len(results)}")
    print(f"  Correct: {correct_count}/{len(results)}")
    print(f"  Mean entropy: {mean_ent:.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
