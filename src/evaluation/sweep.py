import json
import os
import time

from src.inference.layer_duplicator import LayerDuplicator
from src.evaluation.math_eval import run_math_eval
from src.evaluation.eq_eval import run_eq_eval


def enumerate_configs(num_layers):
    """Return all valid (i, j) configurations including baseline.

    Baseline is (None, None). Duplication configs are all (i, j) where
    0 <= i < j <= num_layers. For 28 layers: 406 configs + 1 baseline = 407.
    """
    configs = [(None, None)]
    for i in range(num_layers):
        for j in range(i + 1, num_layers + 1):
            configs.append((i, j))
    return configs


def config_key(i, j):
    """String key for a configuration."""
    if i is None:
        return "baseline"
    return f"{i}_{j}"


def load_existing_results(results_path):
    """Load previously completed results for resume support."""
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return {}


def run_sweep(model, tokenizer, math_questions, eq_scenarios, results_path,
              max_math=None, max_eq=None, configs=None):
    """Run the (i,j) sweep across all configurations.

    Args:
        model: The base model (temporarily modified per config)
        tokenizer: The tokenizer
        math_questions: Full list of math probe questions
        eq_scenarios: Full list of EQ probe scenarios
        results_path: Path to save results JSON (written after each config)
        max_math: Max math questions per config (None = all)
        max_eq: Max EQ scenarios per config (None = all)
        configs: Specific (i,j) tuples to run (None = all valid)

    Returns:
        Dict of all results keyed by config string.
    """
    from tqdm import tqdm

    num_layers = model.config.num_hidden_layers

    if configs is None:
        configs = enumerate_configs(num_layers)

    math_subset = math_questions[:max_math] if max_math else math_questions
    eq_subset = eq_scenarios[:max_eq] if max_eq else eq_scenarios

    existing = load_existing_results(results_path)
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)

    completed = 0
    skipped = 0
    times = []

    for i, j in tqdm(configs, desc="Sweep", unit="cfg"):
        key = config_key(i, j)

        if key in existing:
            skipped += 1
            continue

        t0 = time.time()

        if i is None:
            _, math_score = run_math_eval(model, tokenizer, math_subset, verbose=False)
            _, eq_score = run_eq_eval(model, tokenizer, eq_subset, verbose=False)
        else:
            dup = LayerDuplicator(i, j)
            with dup.apply(model) as modified:
                _, math_score = run_math_eval(modified, tokenizer, math_subset, verbose=False)
                _, eq_score = run_eq_eval(modified, tokenizer, eq_subset, verbose=False)

        elapsed = time.time() - t0
        times.append(elapsed)

        existing[key] = {
            "i": i,
            "j": j,
            "math_score": math_score,
            "eq_score": eq_score,
            "elapsed_s": round(elapsed, 1),
        }

        # Save after each config for resume support
        with open(results_path, "w") as f:
            json.dump(existing, f, indent=2)

        completed += 1
        avg = sum(times) / len(times)
        remaining = (len(configs) - skipped - completed) * avg
        tqdm.write(f"  {key}: math={math_score:.4f} eq={eq_score:.4f} ({elapsed:.1f}s, ~{remaining/60:.0f}m left)")

    print(f"\nSweep done: {completed} new, {skipped} resumed, {len(configs)} total")
    if times:
        print(f"Avg time per config: {sum(times)/len(times):.1f}s")
    return existing
