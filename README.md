# Adaptive Recursive Inference

Extending [dnhkng's RYS (Repeat Your Self)](https://github.com/dnhkng/RYS) technique from static layer duplication to **adaptive recursive inference** with cosine-similarity-based halting -- and using convergence dynamics as a cheap uncertainty signal.

**For the full writeup, see [WRITEUP.md](WRITEUP.md).**

## Key Findings

- **Convergence predicts math correctness** (AUC=0.69) at 14% the cost of sampling-based uncertainty (8 samples)
- **Convergence inverts for reasoning** -- the model converges *harder* on wrong answers (r=-0.38)
- **Three-regime phase transition**: Safe (θ≤0.70), Plateau (θ=0.80-0.95), Cliff (θ≥0.96)
- **Softmax entropy is worse than random for math** (AUC=0.34) -- convergence captures structural information that token-level confidence misses

## Background

RYS duplicates contiguous mid-stack transformer layers at inference time, running them twice in the forward pass. This improves benchmark performance without modifying weights -- the hypothesis being that "reasoning" layers benefit from a second pass through the same circuit.

This project makes that process dynamic: instead of a fixed duplication config, the model loops through its reasoning layers until the hidden state converges (measured by cosine similarity between passes). We then study whether the convergence rate itself serves as a useful uncertainty signal -- and find that it does for math, but actively misleads for reasoning.

## Project Structure

```
WRITEUP.md              # Full writeup with figures and analysis
figures/                 # 9 publication-quality figures
src/
  inference/             # Adaptive recursive inference engine
  evaluation/            # Math and reasoning probes, sweep runner
  analysis/              # Convergence stats, calibration, uncertainty comparison,
                         #   token profiles, statistical summary
  utils/                 # Cosine analysis, layer contribution metrics
scripts/
  generate_figures.py    # Master figure script (produces all 9 figures)
  run_sweep.py           # Phase transition threshold sweep
  collect_traces.py      # Convergence trace collection
  sample_uncertainty.py  # Sampling-based uncertainty baseline
  entropy_baseline.py    # Softmax entropy baseline
data/                    # Probe datasets (math, reasoning)
results/                 # Traces, sweep results, baselines, statistical summary
```

## Running

Requires Python 3.12+, PyTorch, and Transformers. Uses `uv` for dependency management.

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Generate all figures from existing results
uv run python scripts/generate_figures.py

# Generate statistical summary
uv run python -c "from src.analysis.statistical_summary import StatisticalSummary; StatisticalSummary().to_json('results/statistical_summary.json')"
```

### Data Collection (requires GPU)

```bash
# Convergence trace collection (RunPod L40S)
uv run python scripts/collect_traces.py

# Phase transition sweep
uv run python scripts/run_sweep.py

# Sampling and entropy baselines
uv run python scripts/sample_uncertainty.py
uv run python scripts/entropy_baseline.py
```

## Compute

- **Mac M1 Max (32GB)** -- analysis, figure generation, development
- **Velaryon (RTX 2070S)** -- batch sweeps via K8s Jobs on the Goldentooth cluster
- **RunPod (L40S 48GB)** -- data collection with Qwen2.5-7B-Instruct

## Status

Complete through Stage 6 (Writeup and Visualization). See [WRITEUP.md](WRITEUP.md) for the full analysis.
