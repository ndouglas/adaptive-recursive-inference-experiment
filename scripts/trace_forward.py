import torch
import matplotlib.pyplot as plt
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
  model_name,
  dtype=torch.float16,
  device_map="auto",
)

# Inspect the architecture
print(model)

num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")

num_layers = model.config.num_hidden_layers
print(f"Layers: {num_layers}")

# Trace hidden states for multiple prompts

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
  print(f"Hidden states: {len(hidden_states)} x {hidden_states[0].shape}")

  norms = []
  deltas = []

  for i, hs in enumerate(hidden_states):
    norm = hs.float().norm(dim=-1).mean().item()
    norms.append(norm)
    if i > 0:
      delta = (hs - hidden_states[i - 1]).float().norm(dim=-1).mean().item()
      deltas.append(delta)
      print(f"  Layer {i:2d}: norm = {norm:.4f}, delta = {delta:.4f}")

  label = prompt[:40]
  ax1.plot(range(len(hidden_states)), norms, marker='o', markersize=3, label=label)
  ax2.plot(range(1, len(hidden_states)), deltas, marker='o', markersize=3, label=label)

ax1.set_xlabel("Layer")
ax1.set_ylabel("L2 Norm")
ax1.set_title("Hidden State Magnitude Across Layers")
ax1.legend()

ax2.set_xlabel("Layer")
ax2.set_ylabel("Delta L2 Norm")
ax2.set_title("Per-Layer Change in Hidden State")
ax2.legend()

os.makedirs("results", exist_ok=True)
fig.tight_layout()
fig.savefig("results/forward_trace.png", dpi=150)
print("\nSaved to results/forward_trace.png")