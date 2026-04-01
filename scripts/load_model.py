import torch
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

# Generate text

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
  outputs = model.generate(
    **inputs,
    max_new_tokens=50,
  )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {result}")

# Memory check
print(f"MPS memory allocated: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")
print(f"MPS driver memory: {torch.mps.driver_allocated_memory() / 1024**2:.1f} MB")

