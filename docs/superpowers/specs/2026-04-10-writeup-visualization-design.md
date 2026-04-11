# Stage 6: Writeup and Visualization Suite — Design Spec

## Format

Hybrid technical post: blog-post tone with paper-like structure (clear sections, numbered figures). GitHub-flavored markdown (`WRITEUP.md` at repo root), inline figure references to `figures/` directory. Intended for posting on GitHub with a LinkedIn link.

## Audience

Layered — ML practitioners first (cheap uncertainty signal, Pareto tradeoff, actionable takeaways), then ML researchers deeper in (phase transitions, per-token convergence structure, confident-wrong phenomenon).

## Narrative Structure: Problem → Surprise → Deep Dive

### Section 1: Opening Hook (~200 words)
The practical question: LLMs don't know when they're wrong. Uncertainty quantification matters but the standard approach (sampling N=8 outputs) costs 8× compute. Can we get a useful signal cheaper?

### Section 2: The Idea (~400 words)
- Adaptive recursive inference: run mid-layers (15-20) of Qwen2.5-7B repeatedly until cosine similarity of hidden states converges past threshold θ
- Convergence rate as a free uncertainty signal — fast convergence = confident, slow = uncertain
- Setup: constrained JSON decoding via `outlines`, 100 math + 100 reasoning probes, expanded from GSM8K/MATH
- **Fig 1: Method diagram** — schematic of adaptive loop within transformer

### Section 3: The Surprise (~500 words)
- Math: convergence predicts correctness (AUC=0.69 for similarity, 0.66 for speed)
- Reasoning: convergence *inverts* — model converges harder on wrong answers (r=-0.38 at θ=0.95)
- The model confidently produces wrong reasoning answers. This is the hook.
- **Fig 2: ROC curves** — two-panel, math vs reasoning, showing the contrast

### Section 4: Three Regimes (~600 words)
- Phase transition sweep (13 thresholds, 0.50-0.99) reveals three regimes:
  - Safe (θ≤0.70): baseline accuracy (math 97%, reasoning 92%), 1 iteration
  - Plateau (θ=0.80-0.95): mild degradation (math 91%, reasoning 77-82%), ~2 iterations
  - Cliff (θ≥0.96): sharp accuracy collapse, iterations saturate at max
- Critical θ* is task-dependent: math θ*≈0.96, reasoning θ*≈0.95
- Transition is gradual, not a sharp step function
- **Fig 3: Phase transition** — two-panel accuracy vs θ with regime annotations

### Section 5: Where Uncertainty Lives (~500 words)
- Per-token convergence profiles by role (structural/reasoning/answer)
- Structural tokens converge slowest (not fastest as expected — they're "deterministic" but the model still iterates on them)
- Answer tokens converge fastest
- Uncertainty localizes to mid-reasoning tokens (positions 50-150)
- The model "knows" it's going wrong before committing
- **Fig 4: Convergence by token role** — bar chart, correct vs incorrect
- **Fig 5: Convergence heatmaps** — 2×2 grid of representative examples

### Section 6: Is the Signal Calibrated? (~400 words)
- Reliability diagrams: binned confidence vs observed accuracy
- Best-calibrated metrics: convergence speed for reasoning (ECE=0.10), iterations for math (ECE=0.19)
- Anti-calibrated: final similarity for reasoning (ECE=0.61) — the confident-wrong problem quantified
- **Fig 6: Reliability diagrams** — two-panel with best and worst metrics

### Section 7: The Cost Question (~500 words)
- Head-to-head comparison: convergence vs sampling (N=8) vs softmax entropy
- Key numbers:
  - Math: sampling AUC=0.874 (8.0 FP), convergence similarity AUC=0.690 (1.09 FP)
  - Reasoning: sampling AUC=0.829 (8.0 FP), convergence iterations AUC=0.560 (1.12 FP)
- Convergence: 79% of sampling's AUC at 14% of the cost (math)
- Entropy surprisingly poor for math (AUC=0.338) — convergence captures structural info beyond token-level confidence
- **Fig 7: Pareto frontier** — AUC vs cost, all methods
- **Fig 8: Scatter plots** — convergence vs sampling agreement, colored by correctness

### Section 8: What This Means (~400 words)
- Practical: convergence is a viable cheap uncertainty signal for computation-heavy tasks, not for reasoning
- Theoretical: transformers process math and reasoning differently at the representation level — math representations stabilize when correct, reasoning representations stabilize regardless
- The confident-wrong finding as a window into model cognition
- Limitations: single model family (Qwen2.5), small-scale (100 probes per type), specific circuit block choice
- Future: multi-model comparison, larger datasets, adaptive threshold selection
- **Fig 9: Summary table** — method × task, AUC, cost, ECE at a glance

### Section 9: Methods Appendix (~300 words)
- Model: Qwen2.5-7B-Instruct, layers 15-20 as circuit block
- Hardware: RunPod L40S (data collection), Mac M1 Max (analysis)
- Datasets: 100 math (GSM8K + MATH), 100 reasoning (custom multi-step)
- Hyperparameters: max_iterations=4, θ=0.80 for traces, temperature=0.7 for sampling, N=8 samples
- Constrained JSON decoding via `outlines`
- Link to GitHub repo for full reproducibility

## Deliverables

### 1. `WRITEUP.md`
- ~4000-5000 words of GitHub-flavored markdown
- Inline figure references: `![Fig N: Caption](figures/filename.png)`
- Self-contained — readable without running any code

### 2. `scripts/generate_figures.py`
- Single script producing all 9 figures into `figures/`
- Consistent matplotlib style: shared color palette, font sizes, figure dimensions
- Multi-panel layouts where comparisons matter
- Reads from `results/` JSON files directly
- Uses existing analysis modules (`ConvergenceAnalyzer`, `CalibrationAnalyzer`, `UncertaintyComparison`, `TokenProfileAnalyzer`)

**Figure inventory:**
1. Method diagram (horizontal flow: input → layers 1-14 → loop box with "layers 15-20, cosine check, repeat" → layers 21-32 → output; drawn with matplotlib patches/arrows, not an external tool)
2. ROC curves — math vs reasoning (two-panel)
3. Phase transition — accuracy vs θ (two-panel with regime annotations)
4. Convergence by token role (bar chart, correct vs incorrect)
5. Convergence heatmaps (2×2 grid of representative examples)
6. Reliability diagrams (two-panel, best and anti-calibrated metrics)
7. Pareto frontier (AUC vs cost, all methods)
8. Scatter — convergence vs sampling (two-panel)
9. Summary table (rendered as matplotlib table figure)

### 3. `src/analysis/statistical_summary.py`
- Loads all result files from `results/`
- Computes/collects: correlations + 95% bootstrap CIs, Cohen's d effect sizes, ECE values, AUC scores, cost data
- Reuses existing analyzers (`ConvergenceAnalyzer`, `CalibrationAnalyzer`, `UncertaintyComparison`)
- Outputs structured dict + optional JSON export
- Single source of truth for every number in the writeup

## Style Constraints

- **Figures**: consistent matplotlib rcParams (font family, sizes, colors). Shared color palette: one color per method across all figures. Figure size: 10×5 for two-panel, 10×10 for 2×2 grid, 8×5 for single panel.
- **Writing**: concise, first person plural ("we"), present tense for results ("convergence predicts..."), past tense for methods ("we ran..."). No jargon without definition on first use.
- **Markdown**: GitHub-compatible. No LaTeX math (use plain text or Unicode). Tables in GFM pipe syntax. Figures as relative image links.
