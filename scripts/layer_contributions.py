import os
import sys
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.layer_contributions import (
    contribution_magnitudes,
    contribution_alignment,
    inter_layer_contribution_similarity,
)

# Load model
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
)

prompts = [
    "The capital of France is",
    "If a train leaves Chicago traveling at 60 miles per hour",
    "Write a poem about",
    "What is 347 times 829?",
]

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

for prompt in prompts:
    print(f"\n--- {prompt} ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    mags = contribution_magnitudes(hidden_states)
    aligns = contribution_alignment(hidden_states)
    inter_sim = inter_layer_contribution_similarity(hidden_states)

    for i, (m, a) in enumerate(zip(mags, aligns)):
        print(f"  Layer {i+1:2d}: magnitude = {m:.4f}, alignment = {a:.6f}")

    label = prompt[:40]
    ax1.plot(range(1, 29), mags, marker="o", markersize=3, label=label)
    ax2.plot(range(1, 29), aligns, marker="o", markersize=3, label=label)
    ax3.plot(range(2, 29), inter_sim, marker="o", markersize=3, label=label)

ax1.set_xlabel("Layer")
ax1.set_ylabel("L2 Norm of Delta")
ax1.set_title("Contribution Magnitude (how much each layer changes the hidden state)")
ax1.legend()

ax2.set_xlabel("Layer")
ax2.set_ylabel("Cosine Similarity (delta vs stream)")
ax2.set_title("Contribution Alignment (reinforcing vs redirecting)")
ax2.legend()

ax3.set_xlabel("Layer")
ax3.set_ylabel("Cosine Similarity (adjacent deltas)")
ax3.set_title("Inter-Layer Contribution Similarity (circuit boundary detector)")
ax3.legend()

os.makedirs("results", exist_ok=True)
fig.tight_layout()
fig.savefig("results/layer_contributions.png", dpi=150)
print("\nSaved to results/layer_contributions.png")
