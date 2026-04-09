# Adaptive Recursive Inference

Extending [dnhkng's RYS (Repeat Your Self)](https://github.com/dnhkng/RYS) technique from static layer duplication to **adaptive recursive inference** with cosine-similarity-based halting.

## Background

RYS duplicates contiguous mid-stack transformer layers at inference time, running them twice in the forward pass. This improves benchmark performance without modifying weights -- the hypothesis being that "reasoning" layers benefit from a second pass through the same circuit.

This project makes that process dynamic: instead of a fixed duplication config, the model loops through its reasoning layers until the hidden state converges (measured by cosine similarity between passes). The core hypothesis is that convergence behaves like a laser cavity -- coherent features amplify while incoherent features attenuate, producing a detectable phase transition.

## Approach

1. **Characterize model anatomy** -- trace forward passes, measure layer contributions, identify the three-phase structure (encoder / reasoning / decoder)
2. **Build proxy evaluation tasks** -- math (structured reasoning) and EQ (emotional/contextual reasoning) probes that test orthogonal capabilities
3. **Sweep all (i, j) layer duplication configs** -- generate heatmaps showing which blocks improve performance, identifying functional circuit boundaries
4. **Implement adaptive halting** -- replace static duplication with a cosine-similarity convergence loop, measuring per-token halting behavior
5. **Measure convergence dynamics** -- characterize the phase transition, measure across model scales, validate on held-out benchmarks

## Project Structure

```
src/
  inference/       # Layer duplication (LayerDuplicator)
  evaluation/      # Math and EQ probes, sweep runner
  analysis/        # Heatmap generation
  utils/           # Cosine analysis, layer contribution metrics
scripts/           # Runnable experiments (trace, analyze, evaluate, sweep)
data/              # Probe datasets (math_probe.json, eq_probe.json)
k8s/               # Dockerfile and K8s Job manifests for GPU cluster
results/           # Sweep results, eval outputs
plots/             # Generated heatmaps and analysis plots
```

## Running

Requires Python 3.12+, PyTorch, and Transformers. Uses `uv` for dependency management.

```bash
# Environment verification
uv run python scripts/verify_environment.py

# Forward pass analysis
uv run python scripts/trace_forward.py
uv run python scripts/cosine_analysis.py
uv run python scripts/layer_contributions.py

# Layer duplication experiments
uv run python scripts/layer_duplication.py

# Proxy task evaluation (baseline vs duplicated)
uv run python scripts/math_eval.py
uv run python scripts/eq_eval.py

# Full (i,j) sweep (long-running, ~1hr on GPU)
uv run python scripts/run_sweep.py --max-math 8 --max-eq 8

# Generate heatmaps from sweep results
uv run python scripts/generate_heatmaps.py
```

## Compute

- **Mac M1 Max (64GB)** -- development, analysis, interactive exploration via MPS
- **Velaryon (RTX 2070S)** -- batch sweeps via K8s Jobs on the GoldenTooth cluster
- **RunPod (A100+)** -- 7B+ model experiments

## Status

Work in progress. Currently running the (i,j) sweep on Qwen2.5-1.5B.
