import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.evaluation.math_eval import run_math_eval
from src.inference.layer_duplicator import LayerDuplicator

# Load model
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
)

# Load probe set
with open("data/math_probe.json") as f:
    questions = json.load(f)

# --- Baseline ---
print("=== Baseline (no duplication) ===")
baseline_results, baseline_score = run_math_eval(model, tokenizer, questions)
print(f"\nBaseline aggregate score: {baseline_score:.4f}")

# --- Duplicated (12, 17) ---
print("\n=== Duplicated (12, 17) — mid-block reasoning layers ===")
dup = LayerDuplicator(12, 17)
with dup.apply(model) as modified_model:
    dup_results, dup_score = run_math_eval(modified_model, tokenizer, questions)
print(f"\nDuplicated aggregate score: {dup_score:.4f}")

# --- Summary ---
print(f"\n=== Summary ===")
print(f"Baseline:   {baseline_score:.4f}")
print(f"Duplicated: {dup_score:.4f}")
print(f"Delta:      {dup_score - baseline_score:+.4f}")

# Save results
os.makedirs("results", exist_ok=True)
output = {
    "baseline": {"score": baseline_score, "results": baseline_results},
    "duplicated_12_17": {"score": dup_score, "results": dup_results},
}
with open("results/math_eval_baseline.json", "w") as f:
    json.dump(output, f, indent=2)
print("\nSaved to results/math_eval_baseline.json")
