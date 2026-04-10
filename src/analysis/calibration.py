"""Calibration analysis for convergence-based uncertainty signals.

Bins traces by a convergence metric and measures whether the metric value
is calibrated — i.e., whether higher metric values correspond to higher
accuracy. Computes Expected Calibration Error (ECE) and reliability
diagram data.
"""
import numpy as np


class CalibrationAnalyzer:
    """Analyze calibration of a convergence metric as a confidence signal.

    Args:
        traces: List of trace dicts with "correct" and "summary" fields.
        metric: Which convergence metric to use as the confidence signal.
            One of: "avg_final_similarity", "avg_iterations",
            "avg_convergence_speed", "pct_early_halt".
        higher_is_confident: If True (default), higher metric values mean
            higher confidence. Set False for "avg_iterations" where lower
            means more confident.
    """

    def __init__(self, traces, metric="avg_final_similarity",
                 higher_is_confident=None):
        self.traces = traces
        self.metric = metric
        self._correct = np.array([t["correct"] for t in traces], dtype=float)
        self._values = np.array([
            t["summary"].get(metric, 0) for t in traces
        ], dtype=float)

        if higher_is_confident is None:
            higher_is_confident = metric not in ("avg_iterations",)
        self.higher_is_confident = higher_is_confident

        vmin, vmax = self._values.min(), self._values.max()
        if vmax > vmin:
            self._confidence = (self._values - vmin) / (vmax - vmin)
        else:
            self._confidence = np.full_like(self._values, 0.5)

        if not self.higher_is_confident:
            self._confidence = 1.0 - self._confidence

    def reliability_bins(self, n_bins=10):
        """Bin traces by confidence and compute accuracy per bin.

        Returns:
            List of dicts with bin_center, accuracy, confidence, count.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bins = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (self._confidence >= lo) & (self._confidence < hi)
            if i == n_bins - 1:
                mask = mask | (self._confidence == hi)
            count = mask.sum()
            if count == 0:
                continue
            accuracy = self._correct[mask].mean()
            confidence = self._confidence[mask].mean()
            bins.append({
                "bin_center": float((lo + hi) / 2),
                "accuracy": float(accuracy),
                "confidence": float(confidence),
                "count": int(count),
            })
        return bins

    def expected_calibration_error(self, n_bins=10):
        """Compute Expected Calibration Error (ECE).

        ECE = sum over bins of (count/total) * |accuracy - confidence|
        """
        bins = self.reliability_bins(n_bins)
        total = sum(b["count"] for b in bins)
        if total == 0:
            return 0.0
        ece = sum(
            (b["count"] / total) * abs(b["accuracy"] - b["confidence"])
            for b in bins
        )
        return float(ece)

    def summary(self, n_bins=10):
        """Produce a complete calibration summary."""
        return {
            "metric": self.metric,
            "n": len(self.traces),
            "ece": self.expected_calibration_error(n_bins),
            "bins": self.reliability_bins(n_bins),
        }
