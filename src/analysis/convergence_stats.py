"""Statistical analysis of convergence traces.

ConvergenceAnalyzer takes a list of trace dicts (from collect_traces.py output)
and computes correlations, ROC AUC, and bootstrap confidence intervals to
answer: does convergence predict correctness?
"""
import numpy as np
from scipy.stats import pearsonr, pointbiserialr
from sklearn.metrics import roc_auc_score


CONVERGENCE_METRICS = [
    "avg_iterations",
    "avg_final_similarity",
    "avg_convergence_speed",
    "pct_early_halt",
]


class ConvergenceAnalyzer:
    """Analyze convergence-correctness relationships from trace data.

    Args:
        traces: List of trace dicts, each with "score", "correct", and
            "summary" containing convergence metrics.
    """

    def __init__(self, traces):
        self.traces = traces
        self._scores = np.array([t["score"] for t in traces])
        self._correct = np.array([t["correct"] for t in traces], dtype=float)
        self._metrics = {}
        for key in CONVERGENCE_METRICS:
            vals = [t["summary"].get(key) for t in traces]
            if all(v is not None for v in vals):
                self._metrics[key] = np.array(vals)

    def compute_correlations(self, bootstrap_n=0):
        """Compute correlation between each convergence metric and score.

        Args:
            bootstrap_n: If > 0, compute bootstrap 95% CIs with this many samples.

        Returns:
            Dict mapping metric name -> {"r": float, "p": float, "ci_95": [lo, hi]}.
        """
        results = {}
        for key, values in self._metrics.items():
            r, p = pearsonr(values, self._scores)
            entry = {"r": float(r), "p": float(p)}

            if bootstrap_n > 0:
                boot_rs = []
                rng = np.random.RandomState(42)
                n = len(values)
                for _ in range(bootstrap_n):
                    idx = rng.randint(0, n, size=n)
                    boot_vals = values[idx]
                    boot_scores = self._scores[idx]
                    if np.std(boot_vals) > 0 and np.std(boot_scores) > 0:
                        br, _ = pearsonr(boot_vals, boot_scores)
                        boot_rs.append(br)
                if boot_rs:
                    entry["ci_95"] = [
                        float(np.percentile(boot_rs, 2.5)),
                        float(np.percentile(boot_rs, 97.5)),
                    ]
                else:
                    entry["ci_95"] = [float(r), float(r)]

            results[key] = entry
        return results

    def compute_roc_auc(self):
        """Compute ROC AUC for each metric as a binary classifier of correct/incorrect.

        For metrics where higher = more correct (similarity, speed, early_halt),
        uses the metric directly. For iterations (higher = less correct), inverts.

        Returns:
            Dict mapping metric name -> {"auc": float}.
        """
        results = {}
        # Metrics where higher value means MORE likely correct
        positive_direction = {"avg_final_similarity", "avg_convergence_speed", "pct_early_halt"}

        for key, values in self._metrics.items():
            if len(np.unique(self._correct)) < 2:
                results[key] = {"auc": 0.5}
                continue

            if key in positive_direction:
                scores = values
            else:
                # Invert: for iterations, lower = better, so negate
                scores = -values

            try:
                auc = roc_auc_score(self._correct, scores)
                results[key] = {"auc": float(auc)}
            except ValueError:
                results[key] = {"auc": 0.5}

        return results

    def stratified_correlations(self, group_key="category"):
        """Compute correlations stratified by a grouping field.

        Args:
            group_key: Field name in trace dict to group by (e.g., "category", "difficulty").

        Returns:
            Dict mapping group_value -> {"n": int, metric: {"r": float, "p": float}}.
        """
        groups = {}
        for t in self.traces:
            g = t.get(group_key, "unknown")
            groups.setdefault(g, []).append(t)

        results = {}
        for group_name, group_traces in sorted(groups.items()):
            if len(group_traces) < 5:
                continue
            sub = ConvergenceAnalyzer(group_traces)
            corr = sub.compute_correlations()
            corr["n"] = len(group_traces)
            results[group_name] = corr
        return results

    def summary_table(self):
        """Produce a complete summary: overall + stratified correlations and ROC."""
        return {
            "overall": {
                "n": len(self.traces),
                "correlations": self.compute_correlations(bootstrap_n=1000),
                "roc_auc": self.compute_roc_auc(),
                "accuracy": float(np.mean(self._correct)),
                "mean_score": float(np.mean(self._scores)),
            },
        }
