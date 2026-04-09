import os
import sys
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.cosine_analysis import adjacent_layer_similarity, convergence_to_final

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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for prompt in prompts:
    print(f"\n--- {prompt} ---")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    adj_sim = adjacent_layer_similarity(hidden_states)
    conv_sim = convergence_to_final(hidden_states)

    for i, sim in enumerate(adj_sim):
        print(f"  Layer {i+1:2d}: adjacent sim = {sim:.6f}")

    label = prompt[:40]
    ax1.plot(range(1, len(hidden_states)), adj_sim, marker="o", markersize=3, label=label)
    ax2.plot(range(len(hidden_states)), conv_sim, marker="o", markersize=3, label=label)

ax1.set_xlabel("Layer")
ax1.set_ylabel("Cosine Similarity")
ax1.set_title("Adjacent-Layer Similarity")
ax1.legend()

ax2.set_xlabel("Layer")
ax2.set_ylabel("Cosine Similarity to Final Layer")
ax2.set_title("Convergence to Final Representation")
ax2.legend()

os.makedirs("results", exist_ok=True)
fig.tight_layout()
fig.savefig("results/cosine_analysis.png", dpi=150)
print("\nSaved to results/cosine_analysis.png")
