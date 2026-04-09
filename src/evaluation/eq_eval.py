import re
import torch
import numpy as np
from scipy.stats import pearsonr


def parse_ratings(text):
    """Extract up to 4 numbers (0-100) from model output text."""
    numbers = re.findall(r'\b(\d{1,3})\b', text)
    ratings = []
    for n in numbers:
        val = int(n)
        if 0 <= val <= 100:
            ratings.append(val)
        if len(ratings) == 4:
            break
    return ratings if len(ratings) == 4 else None


def score_scenario(predicted, reference):
    """Pearson correlation between predicted and reference ratings."""
    if predicted is None:
        return 0.0
    # Constant input means no variance — correlation is undefined
    if len(set(predicted)) == 1 or len(set(reference)) == 1:
        return 0.0
    r, _ = pearsonr(predicted, reference)
    # Clamp negative correlations to 0 (anticorrelated = no credit)
    return max(0.0, r)


def run_eq_eval(model, tokenizer, scenarios):
    """Run EQ evaluation on a list of scenarios. Returns (results, aggregate_score)."""
    results = []
    for s in scenarios:
        emotion_list = ", ".join(s["emotions"])
        prompt = (
            f"Read the following scenario and rate how strongly the character "
            f"feels each emotion on a scale of 0 to 100.\n\n"
            f"Scenario: {s['scenario']}\n"
            f"Character: {s['character']}\n"
            f"Emotions: {emotion_list}\n\n"
            f"Respond with exactly four numbers (0-100), one for each emotion, "
            f"separated by commas. Nothing else.\n"
            f"Ratings:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
            )

        new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        predicted = parse_ratings(response)
        score = score_scenario(predicted, s["reference"])
        results.append({
            "character": s["character"],
            "emotions": s["emotions"],
            "reference": s["reference"],
            "response": response,
            "predicted": predicted,
            "score": score,
        })
        print(f"  {s['character']}: {s['scenario'][:60]}...")
        print(f"    Emotions:  {s['emotions']}")
        print(f"    Reference: {s['reference']}")
        print(f"    Response:  {response}")
        print(f"    Parsed:    {predicted}, Score: {score:.4f}")

    valid = [r for r in results if r["predicted"] is not None]
    parse_rate = len(valid) / len(results)
    aggregate = sum(r["score"] for r in results) / len(results) if results else 0.0

    print(f"\n  Parse rate: {len(valid)}/{len(results)} ({parse_rate:.0%})")
    return results, aggregate
