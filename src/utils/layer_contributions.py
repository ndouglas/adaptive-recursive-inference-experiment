import torch.nn.functional as F

def contribution_magnitudes(hidden_states):
  magnitudes = []
  for i in range(1, len(hidden_states)):
    delta = (hidden_states[i] - hidden_states[i - 1]).float()
    mag = delta.norm(dim=-1).mean().item()
    magnitudes.append(mag)
  return magnitudes

def contribution_alignment(hidden_states):
  alignments = []
  for i in range(1, len(hidden_states)):
    delta = (hidden_states[i] - hidden_states[i - 1]).float()
    stream = hidden_states[i].float()
    cos_sim = F.cosine_similarity(delta, stream, dim=-1)
    alignments.append(cos_sim.mean().item())
  return alignments

def inter_layer_contribution_similarity(hidden_states):
  deltas = []
  for i in range(1, len(hidden_states)):
    delta = (hidden_states[i] - hidden_states[i - 1]).float()
    deltas.append(delta)

  similarities = []
  for i in range(1, len(deltas)):
    cos_sim = F.cosine_similarity(deltas[i], deltas[i - 1], dim=-1)
    similarities.append(cos_sim.mean().item())
  return similarities
