import os
import sys
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference.layer_duplicator import LayerDuplicator
from src.utils.cosine_analysis import adjacent_layer_similarity

# Load model
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# --- Baseline (no duplication) ---
print("=== Baseline ===")
with torch.no_grad():
    baseline_outputs = model(**inputs, output_hidden_states=True)
baseline_sim = adjacent_layer_similarity(baseline_outputs.hidden_states)
print(f"Layers in forward pass: {len(baseline_outputs.hidden_states) - 1}")

# Generate text for comparison
with torch.no_grad():
    gen_ids = model.generate(**inputs, max_new_tokens=50)
print(f"Generated: {tokenizer.decode(gen_ids[0], skip_special_tokens=True)}")

# --- Good config: duplicate middle reasoning layers (12, 17) ---
configs = [
    ((12, 17), "Mid-block (12,17) — reasoning layers"),
    ((0, 3), "Early (0,3) — encoder layers"),
    ((25, 28), "Late (25,28) — decoder layers"),
]

fig, axes = plt.subplots(len(configs), 1, figsize=(10, 4 * len(configs)))

for ax, ((i, j), description) in zip(axes, configs):
    print(f"\n=== {description} ===")
    dup = LayerDuplicator(i, j)

    with dup.apply(model) as modified_model:
        with torch.no_grad():
            dup_outputs = modified_model(**inputs, output_hidden_states=True)
        dup_sim = adjacent_layer_similarity(dup_outputs.hidden_states)
        num_dup_layers = len(dup_outputs.hidden_states) - 1
        print(f"Layers in forward pass: {num_dup_layers}")

        # Generate text
        with torch.no_grad():
            gen_ids = modified_model.generate(**inputs, max_new_tokens=50)
        print(f"Generated: {tokenizer.decode(gen_ids[0], skip_special_tokens=True)}")

    # Plot baseline and duplicated on same axes
    ax.plot(range(1, len(baseline_sim) + 1), baseline_sim,
            marker="o", markersize=3, label="Baseline (28 layers)", alpha=0.7)
    ax.plot(range(1, len(dup_sim) + 1), dup_sim,
            marker="s", markersize=3, label=f"Duplicated ({num_dup_layers} layers)")

    # Mark the duplicated region
    ax.axvspan(i + 1, j, alpha=0.1, color="green", label=f"Original block [{i},{j})")
    ax.axvspan(j + 1, j + (j - i), alpha=0.1, color="orange", label=f"Duplicate block")

    ax.set_xlabel("Layer (execution order)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Adjacent-Layer Similarity — {description}")
    ax.legend(fontsize=8)

os.makedirs("results", exist_ok=True)
fig.tight_layout()
fig.savefig("results/layer_duplication.png", dpi=150)
print(f"\nSaved to results/layer_duplication.png")
