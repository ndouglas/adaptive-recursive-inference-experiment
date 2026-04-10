import re
import torch


def parse_number(text):
    """Extract the answer number from model output text.

    Handles both terse ("45") and chain-of-thought ("To find... = 45") output.
    First strips trailing question repetitions, then checks for answer-signaling
    patterns, then falls back to the first number.
    """
    # Strip trailing question repetitions (model often echoes next question)
    text = re.split(r'\nQuestion:', text)[0]
    text = text.strip()

    # Check for answer-signaling patterns
    for pattern in [
        r'(?:answer|result)\s*(?:is|:)\s*(-?\d[\d,]*\.?\d*)',
        r'=\s*(-?\d[\d,]*\.?\d*)',
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1).replace(',', ''))

    # Fall back to first number
    match = re.search(r'-?\d[\d,]*\.?\d*', text)
    if match:
        return float(match.group().replace(',', ''))
    return None


def score_answer(predicted, expected):
    """Partial-credit scoring based on relative error."""
    if predicted is None:
        return 0.0
    if expected == 0:
        return 1.0 if predicted == 0 else 0.0
    relative_error = abs(predicted - expected) / abs(expected)
    score = max(0.0, 1.0 - relative_error)
    return score


def run_math_eval(model, tokenizer, questions, verbose=True):
    """Run math evaluation on a list of questions. Returns (results, aggregate_score)."""
    results = []
    for q in questions:
        prompt = f"Answer with just the number, nothing else.\n\nQuestion: {q['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )

        # Decode only the new tokens (skip the prompt)
        new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        predicted = parse_number(response)
        score = score_answer(predicted, q["answer"])
        results.append({
            "question": q["question"],
            "expected": q["answer"],
            "response": response,
            "predicted": predicted,
            "score": score,
        })
        if verbose:
            print(f"  {q['question']}")
            print(f"    Expected: {q['answer']}, Got: {response} (parsed: {predicted}), Score: {score:.4f}")

    aggregate = sum(r["score"] for r in results) / len(results)
    return results, aggregate
