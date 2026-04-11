# Convergence-as-Uncertainty Signal in Adaptive Recursive Inference

## Revised Project Thesis

Mid-layer convergence dynamics during adaptive recursive inference provide a lightweight, single-pass uncertainty signal that predicts output reliability. We characterize when this signal is informative (computation-heavy tasks like arithmetic and number theory) vs. uninformative (context-heavy tasks like emotional reasoning), and show that it offers per-token uncertainty at a fraction of the cost of sampling-based methods.

## Background and Motivation

The original RYS-extension hypothesis — that adaptive block iteration would universally improve output quality — is not supported by our data. Five evaluation rounds (v1–v5) across two model sizes show:

- **Math/arithmetic:** +7% to +23% improvement, consistent across all versions
- **Number theory:** +12.5% (strongest category-level gain)
- **EQ/emotional reasoning:** -9% to -88%, consistent degradation
- **Multi-step reasoning:** Mixed — hard problems improve (+3.5%), medium problems degrade (-9%)

The interesting finding is not that the mechanism "works" or "doesn't work" — it's that **convergence behavior is task-dependent and predictive**. The model converges quickly on problems it handles well and slowly on problems where it's uncertain. This is a free uncertainty signal embedded in the inference dynamics.

## What's Already Built (Phase 1–3 of original plan)

- Environment: Mac (MPS) + RunPod (A100) pipeline
- Models: Qwen2.5-1.5B and 7B loaded, circuit blocks identified (25-27 for 1.5B, 15-20 for 7B)
- `AdaptiveLoop` with cosine-similarity halting and per-token trajectory capture
- Constrained JSON decoding via `outlines` (eliminates answer extraction as confound)
- Probe datasets: `math_probe.json` (8), `eq_probe.json` (8), `reasoning_probe.json` (32)
- Five rounds of evaluation results (v1–v5) establishing baselines

## Revised Plan: 6 Stages

---

### Stage 1: Convergence Trajectory Instrumentation

**Goal:** Capture rich, structured convergence data suitable for statistical analysis.

**What exists:** `AdaptiveLoop` already records `token_iterations`, `token_similarities`, and `token_trajectories` per generation. `diagnostics_summary()` computes averages.

**What's needed:**

1. A `ConvergenceTrace` dataclass that captures, for each generation:
   - Per-token iteration count
   - Per-token similarity at each iteration (the full trajectory, not just final)
   - Per-token L2 norm of hidden state at each iteration
   - Sequence-level convergence profile: how mean similarity evolves across token positions
   - Which tokens halted early vs. hit max_iterations
   - Wall-clock time per token

2. A `ConvergenceTracer` wrapper that runs generation through `AdaptiveLoop` and produces a `ConvergenceTrace` alongside the generated text and score.

3. Serialization to JSON for offline analysis.

**Changes required:** Modify `AdaptiveLoop.forward()` to optionally capture L2 norms (currently only captures similarity). Build the tracer as a thin wrapper — the loop already does most of the work.

**Success criteria:** Can run any probe through the tracer and get a structured trace object with all fields populated. Traces serialize/deserialize cleanly.

**Compute:** Mac (development and testing on 1.5B)

**Status:** Complete

---

### Stage 2: Convergence–Correctness Correlation

**Goal:** Answer the central question: does convergence rate predict whether the model got the answer right?

**Prerequisite:** Stage 1 (need structured traces)

**Procedure:**

1. **Expand probe datasets.** The current sets (8 math, 8 EQ, 32 reasoning) are too small for statistical significance. Target 50–100 problems per category. Sources:
   - GSM8K for multi-step arithmetic
   - MATH (Hendrycks) for number theory and algebra
   - Custom generation for EQ (or use EmoBench/similar)
   - Keep difficulty labels and step counts for stratification

2. **Collect traces.** Run all probes through the tracer at θ=0.80 (our best threshold) on the 7B model. Record full convergence traces + correctness labels.

3. **Compute correlations:**
   - Mean iterations per problem vs. binary correctness (point-biserial r)
   - Mean final similarity vs. score (Pearson r)
   - Mean convergence speed (similarity gain per iteration) vs. score
   - Stratify by task type (math, reasoning, EQ) and difficulty tier

4. **ROC analysis:** Treat convergence metrics as a binary classifier for "correct vs. incorrect." Compute AUC for each metric, each task type. The headline number: can convergence signal separate right from wrong answers?

5. **Statistical tests:** Bootstrap confidence intervals on all correlations. We need to know whether observed effects are significant or noise from small samples.

**Key question this answers:** "If the model converges quickly, is it more likely to be right?" If AUC > 0.65 for any task type, the signal is useful. If AUC varies significantly across task types, that's the task-dependence finding.

**Success criteria:** Clear correlation tables with CIs, ROC curves per task type, and a definitive answer to whether convergence predicts correctness.

**Compute:** RunPod (7B model, expanded datasets)

**Status:** Complete

---

### Stage 3: Per-Token Convergence Profiles

**Goal:** Move from sequence-level to token-level analysis. Understand which tokens the model is uncertain about.

**Prerequisite:** Stages 1–2 (need traces and correctness labels)

**Procedure:**

1. **Classify tokens by role.** For math/reasoning outputs (constrained JSON), tokens fall into categories:
   - Structural: `{`, `"reasoning"`, `:`, `}` — formatting tokens
   - Reasoning: the content of the reasoning field — the model's "work"
   - Answer: the numeric answer token(s)
   - Use the known JSON structure to classify automatically

2. **Compare convergence by token role:**
   - Do answer tokens converge faster or slower than reasoning tokens?
   - Do structural tokens converge fastest (as expected — they're deterministic)?
   - Is there a "convergence valley" at specific positions (e.g., right before the answer)?

3. **Error analysis:** For problems the model gets wrong, compare the token-level convergence profile to problems it gets right. Are there specific positions where convergence breaks down?

4. **Visualize convergence heatmaps:** Position (x-axis) × iteration (y-axis), colored by similarity. One heatmap per problem, arranged by correctness. Look for visual patterns.

**Key question this answers:** "Is uncertainty localized to specific tokens, or diffuse across the sequence?" If localized, convergence gives per-token confidence — something sampling-based methods can't easily provide.

**Success criteria:** Visualizations showing clear differences in convergence profiles between correct/incorrect outputs and between token roles.

**Compute:** Mac (analysis of traces collected in Stage 2)

**Status:** Complete

---

### Stage 4: Calibration and Phase Transitions

**Goal:** Determine if convergence is a well-calibrated confidence signal, and whether there's a sharp phase transition in representation quality.

**Prerequisite:** Stage 2 (need convergence–correctness data)

**Procedure:**

1. **Calibration analysis:**
   - Bin problems by convergence confidence (e.g., deciles of mean convergence speed)
   - For each bin: what fraction of answers are correct?
   - Plot reliability diagram (predicted confidence vs. observed accuracy)
   - Compute Expected Calibration Error (ECE)
   - Compare to softmax-probability calibration (standard baseline)

2. **Phase transition sweep:**
   - Run the 7B model with thresholds at 0.01 increments from 0.50 to 0.999
   - For each threshold: measure mean score, mean iterations, score variance
   - Look for a sharp transition: is there a critical θ* where score drops abruptly?
   - Fit a sigmoid or step function to the score-vs-threshold curve
   - If a phase transition exists, characterize its width and location per task type

3. **Cross-model comparison (if time permits):**
   - Repeat the phase transition sweep on the 1.5B model
   - Does the transition happen at the same similarity value, or is it model-dependent?
   - A universal critical point would be a strong finding

**Key questions this answers:**
- "Can I trust the convergence signal as a probability?" (calibration)
- "Is there a critical point where representation quality flips?" (phase transition — the original laser cavity hypothesis in its most testable form)

**Success criteria:** Calibration plots with ECE scores. Phase transition plots showing whether the transition is sharp or gradual. If sharp, identification of θ* per task type.

**Compute:** RunPod (fine-grained threshold sweep on 7B)

**Status:** Complete

---

### Stage 5: Comparison to Sampling-Based Uncertainty

**Goal:** Benchmark convergence-based uncertainty against the standard approach (sampling agreement) to show cost-effectiveness.

**Prerequisite:** Stage 2 (need convergence-based uncertainty scores)

**Procedure:**

1. **Generate samples.** For each problem in the expanded dataset, generate N=8 outputs from the baseline model (temperature=0.7 or nucleus sampling). Record all outputs.

2. **Compute sampling-based uncertainty:**
   - For math: fraction of samples agreeing on the same numeric answer
   - For reasoning: answer agreement across samples
   - Self-consistency score (Wang et al., 2022 style)

3. **Compute softmax-entropy uncertainty:**
   - Mean token-level entropy from the baseline model's softmax distribution
   - Sequence-level entropy (geometric mean or similar aggregation)

4. **Head-to-head comparison:**
   - ROC AUC for correct/incorrect classification:
     - Convergence-based (cost: ~1.5 forward passes)
     - Sampling-based (cost: 8 forward passes)
     - Softmax entropy (cost: 1 forward pass)
   - Scatter plots: convergence uncertainty vs. sampling uncertainty (are they correlated?)
   - Cases where they disagree — what does convergence catch that sampling misses, and vice versa?

5. **Cost-quality tradeoff plot:** AUC (y) vs. compute cost in forward passes (x). Show where convergence sits on the Pareto frontier.

**Key question this answers:** "Is convergence-based uncertainty competitive with existing methods?" Even if AUC is lower, the cost advantage (1.5x vs. 8x) could make it Pareto-optimal.

**Success criteria:** Clear comparison table and Pareto plot. Identification of regimes where convergence wins vs. loses.

**Compute:** RunPod (sampling requires multiple generations per problem)

**Status:** Complete

---

### Stage 6: Writeup and Visualization Suite

**Goal:** Produce a coherent narrative and publication-ready figures.

**Prerequisite:** Stages 1–5

**Deliverables:**

1. **Master visualization suite** (Python script producing all figures):
   - Convergence trajectory plots (representative examples: easy/hard, correct/incorrect)
   - ROC curves per task type
   - Calibration reliability diagrams
   - Phase transition plots (score vs. threshold)
   - Token-level convergence heatmaps
   - Cost-quality Pareto frontier
   - Summary table of all experimental results

2. **Statistical summary:**
   - All correlations with 95% bootstrap CIs
   - Effect sizes (Cohen's d for correct vs. incorrect convergence rates)
   - Significance tests (permutation tests preferred over parametric)

3. **Narrative structure** (blog post or short paper):
   - "We extended RYS into adaptive inference and found that it doesn't uniformly improve quality — but the convergence dynamics themselves are informative."
   - "Convergence rate predicts output reliability for computation-heavy tasks (AUC=X) at 1.5x the cost of a single forward pass."
   - "The signal is task-dependent: [characterization]. This tells us something about how transformers process different types of information."
   - "Comparison to sampling shows convergence is [competitive/complementary] at [fraction] of the compute cost."

**Success criteria:** A complete draft with all figures, ready for review.

**Compute:** Mac (analysis and writing)

**Status:** Complete

---

## Decision Points

- **After Stage 2:** If convergence does not predict correctness (AUC < 0.55 across all task types), the project pivots to a negative-result writeup: "why adaptive convergence doesn't work as an uncertainty signal." This is still publishable and honest.

- **After Stage 4:** If no phase transition exists (gradual degradation rather than sharp transition), drop the "laser cavity" framing. The uncertainty signal story stands on its own.

- **After Stage 5:** If convergence is dominated by softmax entropy (same AUC at lower cost), the contribution becomes the characterization of convergence dynamics rather than a practical uncertainty method.
