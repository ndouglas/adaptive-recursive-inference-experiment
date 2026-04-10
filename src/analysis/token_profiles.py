"""Analyze convergence profiles grouped by token role.

Aggregates per-token convergence metrics (iterations, similarity, speed)
by their JSON role (structural/reasoning/answer) and compares profiles
between correct and incorrect outputs.
"""
import numpy as np

from src.analysis.token_roles import TokenRoleClassifier, TokenRole


class TokenProfileAnalyzer:
    """Aggregate convergence behavior by token role and correctness.

    Args:
        model_name: HuggingFace model name for tokenizer (e.g., "Qwen/Qwen2.5-1.5B").
    """

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B"):
        self.classifier = TokenRoleClassifier(model_name)

    def _classify_trace(self, trace):
        """Return list of (TokenRole, token_trace_dict) pairs for a trace."""
        generated = trace["generated"]
        token_traces = trace["token_traces"]
        roles = self.classifier.classify(generated)
        n = min(len(roles), len(token_traces))
        return [(roles[i], token_traces[i]) for i in range(n)]

    def aggregate_by_role(self, traces):
        """Compute mean convergence metrics per token role across all traces.

        Args:
            traces: List of trace dicts with "generated" and "token_traces".

        Returns:
            Dict mapping role name -> {mean_iterations, mean_final_similarity,
            mean_convergence_speed, count}.
        """
        role_data = {r.value: [] for r in TokenRole}

        for trace in traces:
            classified = self._classify_trace(trace)
            for role, tt in classified:
                role_data[role.value].append(tt)

        result = {}
        for role_name, tts in role_data.items():
            if not tts:
                continue
            result[role_name] = {
                "mean_iterations": float(np.mean([t["iterations"] for t in tts])),
                "mean_final_similarity": float(np.mean(
                    [t["final_similarity"] for t in tts if t["final_similarity"] is not None]
                )),
                "mean_convergence_speed": float(np.mean(
                    [t["convergence_speed"] for t in tts if t["convergence_speed"] is not None]
                )),
                "count": len(tts),
            }
        return result

    def compare_correct_vs_incorrect(self, traces):
        """Aggregate by role, split by correctness.

        Returns:
            {"correct": {role: metrics}, "incorrect": {role: metrics}}
        """
        correct = [t for t in traces if t["correct"]]
        incorrect = [t for t in traces if not t["correct"]]
        result = {}
        if correct:
            result["correct"] = self.aggregate_by_role(correct)
        if incorrect:
            result["incorrect"] = self.aggregate_by_role(incorrect)
        return result

    def positional_profile(self, traces):
        """Compute convergence metrics at each token position, averaged across traces.

        Pads shorter traces with NaN so all traces contribute up to their length.

        Returns:
            {"positions": list[int], "mean_iterations": list[float],
             "mean_final_similarity": list[float]}
        """
        max_len = max(len(t["token_traces"]) for t in traces)
        all_iters = np.full((len(traces), max_len), np.nan)
        all_sims = np.full((len(traces), max_len), np.nan)

        for i, trace in enumerate(traces):
            tts = trace["token_traces"]
            for j, tt in enumerate(tts):
                all_iters[i, j] = tt["iterations"]
                if tt["final_similarity"] is not None:
                    all_sims[i, j] = tt["final_similarity"]

        return {
            "positions": list(range(max_len)),
            "mean_iterations": [float(x) for x in np.nanmean(all_iters, axis=0)],
            "mean_final_similarity": [float(x) for x in np.nanmean(all_sims, axis=0)],
        }
