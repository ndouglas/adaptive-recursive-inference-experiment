import torch
import torch.nn.functional as F

def adjacent_layer_similarity(hidden_states):
  similarities = []
  for i in range(1, len(hidden_states)):
    cos_sim = F.cosine_similarity(
      hidden_states[i].float(),
      hidden_states[i - 1].float(),
      dim=-1,
    )
    similarities.append(cos_sim.mean().item())
  return similarities

def convergence_to_final(hidden_states):
  final = hidden_states[-1].float()
  similarities = []
  for hs in hidden_states:
    cos_sim = F.cosine_similarity(hs.float(), final, dim=-1)
    similarities.append(cos_sim.mean().item())
  return similarities

