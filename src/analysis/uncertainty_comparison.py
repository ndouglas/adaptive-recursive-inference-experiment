"""Comparison of uncertainty estimation methods.

Compares convergence-based, sampling-based, and softmax-entropy uncertainty
signals using ROC AUC, Pareto analysis, and disagreement analysis.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


METHOD_CONFIG = {
    "convergence_similarity": {"key": "convergence_similarity", "higher_is_confident": True},
    "convergence_iterations": {"key": "convergence_iterations", "higher_is_confident": False},
    "convergence_speed": {"key": "convergence_speed", "higher_is_confident": True},
    "sampling_agreement": {"key": "sampling_agreement", "higher_is_confident": True},
    "mean_entropy": {"key": "mean_entropy", "higher_is_confident": False},
}


class UncertaintyComparison:
    """Compare uncertainty methods head-to-head.

    Args:
        matched_data: List of dicts, each with keys: correct,
            convergence_similarity, convergence_iterations, convergence_speed,
            sampling_agreement, mean_entropy.
        block_layers: Number of layers in the adaptive circuit block (for cost calc).
        total_layers: Total model layers (for cost calc).
        num_samples: Number of samples used for sampling method (for cost calc).
    """

    def __init__(self, matched_data, block_layers=6, total_layers=32,
                 num_samples=8):
        self.data = matched_data
        self.block_layers = block_layers
        self.total_layers = total_layers
        self.num_samples = num_samples
        self._correct = np.array([d["correct"] for d in matched_data], dtype=float)

    def _get_scores(self, method):
        """Get confidence scores for a method (higher = more confident)."""
        config = METHOD_CONFIG[method]
        values = np.array([d[config["key"]] for d in self.data])
        if not config["higher_is_confident"]:
            values = -values
        return values

    def roc_auc_all(self):
        """Compute ROC AUC for each method.

        Returns:
            Dict mapping method name -> {"auc": float}.
        """
        results = {}
        for method in METHOD_CONFIG:
            scores = self._get_scores(method)
            if len(np.unique(self._correct)) < 2:
                results[method] = {"auc": 0.5}
                continue
            try:
                auc = roc_auc_score(self._correct, scores)
                results[method] = {"auc": float(auc)}
            except ValueError:
                results[method] = {"auc": 0.5}
        return results

    def _compute_cost(self, method):
        """Compute cost in forward-pass equivalents."""
        if method == "mean_entropy":
            return 1.0
        if method == "sampling_agreement":
            return float(self.num_samples)
        if method.startswith("convergence_"):
            avg_iters = np.mean([d["convergence_iterations"] for d in self.data])
            block_fraction = self.block_layers / self.total_layers
            return 1.0 + (avg_iters - 1) * block_fraction
        return 1.0

    def pareto_data(self):
        """Compute AUC and cost for each method (for Pareto frontier plot).

        Returns:
            List of {"method": str, "auc": float, "cost": float}.
        """
        aucs = self.roc_auc_all()
        result = []
        for method in METHOD_CONFIG:
            result.append({
                "method": method,
                "auc": aucs[method]["auc"],
                "cost": self._compute_cost(method),
            })
        return result

    def disagreement_analysis(self, method_a, method_b,
                               threshold_a=0.5, threshold_b=0.5):
        """Analyze cases where two methods disagree on confidence.

        Normalizes scores to [0,1] and splits at the given thresholds.

        Returns:
            Dict with keys: a_confident_b_not, b_confident_a_not,
            both_confident, neither_confident. Each has count and accuracy.
        """
        scores_a = self._get_scores(method_a)
        scores_b = self._get_scores(method_b)

        def normalize(s):
            lo, hi = s.min(), s.max()
            return (s - lo) / (hi - lo) if hi > lo else np.full_like(s, 0.5)

        norm_a = normalize(scores_a)
        norm_b = normalize(scores_b)

        conf_a = norm_a >= threshold_a
        conf_b = norm_b >= threshold_b

        categories = {
            "both_confident": conf_a & conf_b,
            "a_confident_b_not": conf_a & ~conf_b,
            "b_confident_a_not": ~conf_a & conf_b,
            "neither_confident": ~conf_a & ~conf_b,
        }

        result = {}
        for name, mask in categories.items():
            count = int(mask.sum())
            acc = float(self._correct[mask].mean()) if count > 0 else 0.0
            result[name] = {"count": count, "accuracy": acc}
        return result

    def summary_table(self):
        """Produce a complete comparison summary."""
        aucs = self.roc_auc_all()
        methods = []
        for method in METHOD_CONFIG:
            methods.append({
                "method": method,
                "auc": aucs[method]["auc"],
                "cost": self._compute_cost(method),
            })
        methods.sort(key=lambda x: x["auc"], reverse=True)
        return {
            "n": len(self.data),
            "accuracy": float(self._correct.mean()),
            "methods": methods,
        }
