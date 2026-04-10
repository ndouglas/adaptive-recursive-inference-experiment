"""Structured convergence trace data for statistical analysis.

TokenTrace captures per-token convergence dynamics (iterations, similarity
trajectory, L2 norms). ConvergenceTrace wraps a full generation with all
token traces plus metadata.

Both classes serialize to/from plain dicts for JSON storage.
"""
from dataclasses import dataclass, field


@dataclass
class TokenTrace:
    """Convergence data for a single generated token.

    Attributes:
        token_id: The generated token's vocabulary ID.
        iterations: Number of block passes used for this token.
        similarities: Cosine similarity at each iteration.
        l2_norms: L2 norm of hidden state after each iteration.
        halted_early: Whether convergence threshold was reached before max_iterations.
        elapsed_s: Wall-clock time for this token's generation.
    """
    token_id: int
    iterations: int
    similarities: list[float]
    l2_norms: list[float]
    halted_early: bool
    elapsed_s: float

    @property
    def final_similarity(self) -> float | None:
        return self.similarities[-1] if self.similarities else None

    @property
    def convergence_speed(self) -> float | None:
        """Similarity gain per iteration: (last - first) / (n - 1)."""
        if len(self.similarities) < 2:
            return None
        return (self.similarities[-1] - self.similarities[0]) / (len(self.similarities) - 1)

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "iterations": self.iterations,
            "similarities": self.similarities,
            "l2_norms": self.l2_norms,
            "halted_early": self.halted_early,
            "elapsed_s": self.elapsed_s,
            "final_similarity": self.final_similarity,
            "convergence_speed": self.convergence_speed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TokenTrace":
        return cls(
            token_id=d["token_id"],
            iterations=d["iterations"],
            similarities=d["similarities"],
            l2_norms=d["l2_norms"],
            halted_early=d["halted_early"],
            elapsed_s=d["elapsed_s"],
        )


@dataclass
class ConvergenceTrace:
    """Full convergence trace for one generation (prompt -> output).

    Attributes:
        prompt: The input prompt text.
        generated: The generated output text.
        score: Evaluation score (0-1) for the generated answer.
        threshold: Cosine similarity halting threshold used.
        max_iterations: Max iteration cap used.
        token_traces: Per-token convergence data.
    """
    prompt: str
    generated: str
    score: float
    threshold: float
    max_iterations: int
    token_traces: list[TokenTrace] = field(default_factory=list)

    def summary(self) -> dict:
        """Aggregate statistics across all tokens."""
        if not self.token_traces:
            return {}
        iters = [t.iterations for t in self.token_traces]
        finals = [t.final_similarity for t in self.token_traces if t.final_similarity is not None]
        speeds = [t.convergence_speed for t in self.token_traces if t.convergence_speed is not None]
        return {
            "total_tokens": len(self.token_traces),
            "avg_iterations": sum(iters) / len(iters),
            "min_iterations": min(iters),
            "max_iterations_used": max(iters),
            "pct_early_halt": sum(1 for t in self.token_traces if t.halted_early) / len(self.token_traces),
            "avg_final_similarity": sum(finals) / len(finals) if finals else None,
            "avg_convergence_speed": sum(speeds) / len(speeds) if speeds else None,
            "total_elapsed_s": sum(t.elapsed_s for t in self.token_traces),
        }

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "generated": self.generated,
            "score": self.score,
            "threshold": self.threshold,
            "max_iterations": self.max_iterations,
            "token_traces": [t.to_dict() for t in self.token_traces],
            "summary": self.summary(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConvergenceTrace":
        return cls(
            prompt=d["prompt"],
            generated=d["generated"],
            score=d["score"],
            threshold=d["threshold"],
            max_iterations=d["max_iterations"],
            token_traces=[TokenTrace.from_dict(t) for t in d["token_traces"]],
        )
