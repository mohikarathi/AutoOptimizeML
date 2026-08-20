"""Constraint evaluation and multi-objective scoring system."""

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from autoopt.benchmark.metrics import BenchmarkMetrics
from autoopt.optimizer.space import OptimizationCandidate


@dataclass
class OptimizationConstraints:
    min_accuracy: Optional[float] = None
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    min_throughput_samples_per_sec: Optional[float] = None
    objective: str = "maximize_throughput"  # 'maximize_throughput', 'minimize_latency', 'minimize_memory', 'balanced'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    candidate: OptimizationCandidate
    metrics: Optional[BenchmarkMetrics]
    status: str  # 'ACCEPTED', 'REJECTED', 'FAILED'
    rejection_reasons: List[str]
    score: float
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "status": self.status,
            "rejection_reasons": self.rejection_reasons,
            "score": round(self.score, 4),
            "error_message": self.error_message
        }


class ConstraintEvaluator:
    """Evaluates candidate benchmark metrics against hard constraints and computes objective scores."""

    @staticmethod
    def evaluate(
        candidate: OptimizationCandidate,
        metrics: BenchmarkMetrics,
        constraints: OptimizationConstraints,
        baseline_metrics: Optional[BenchmarkMetrics] = None
    ) -> EvaluationResult:
        reasons = []

        # 1. Check Accuracy Constraint
        if constraints.min_accuracy is not None:
            if metrics.accuracy < constraints.min_accuracy:
                reasons.append(
                    f"Accuracy ({metrics.accuracy * 100:.2f}%) < min_accuracy ({constraints.min_accuracy * 100:.2f}%)"
                )

        # 2. Check Latency Constraint
        if constraints.max_latency_ms is not None:
            if metrics.mean_latency_ms > constraints.max_latency_ms:
                reasons.append(
                    f"Latency ({metrics.mean_latency_ms:.2f} ms) > max_latency ({constraints.max_latency_ms:.2f} ms)"
                )

        # 3. Check Memory Constraint
        if constraints.max_memory_mb is not None:
            if metrics.memory_allocated_mb > constraints.max_memory_mb:
                reasons.append(
                    f"Memory ({metrics.memory_allocated_mb:.1f} MB) > max_memory ({constraints.max_memory_mb:.1f} MB)"
                )

        # 4. Check Throughput Constraint
        if constraints.min_throughput_samples_per_sec is not None:
            if metrics.throughput_samples_per_sec < constraints.min_throughput_samples_per_sec:
                reasons.append(
                    f"Throughput ({metrics.throughput_samples_per_sec:.1f} samp/s) < min_throughput ({constraints.min_throughput_samples_per_sec:.1f} samp/s)"
                )

        status = "ACCEPTED" if not reasons else "REJECTED"

        # Compute Objective Score
        score = ConstraintEvaluator._compute_score(metrics, constraints.objective, baseline_metrics)

        # Penalize rejected candidates
        if status == "REJECTED":
            score = -1e6

        return EvaluationResult(
            candidate=candidate,
            metrics=metrics,
            status=status,
            rejection_reasons=reasons,
            score=score
        )

    @staticmethod
    def _compute_score(
        metrics: BenchmarkMetrics,
        objective: str,
        baseline: Optional[BenchmarkMetrics] = None
    ) -> float:
        obj = objective.lower().strip()

        if obj == "maximize_throughput":
            # Primary: throughput in samples/sec
            return metrics.throughput_samples_per_sec

        elif obj == "minimize_latency":
            # Higher score for lower latency
            return 1000.0 / max(1e-4, metrics.mean_latency_ms)

        elif obj == "minimize_memory":
            # Higher score for lower memory
            return 10000.0 / max(1.0, metrics.memory_allocated_mb)

        elif obj == "maximize_accuracy":
            return metrics.accuracy * 100.0

        elif obj == "balanced":
            # Normalized Pareto utility: Throughput / Latency * Accuracy
            lat_factor = 1000.0 / max(1e-4, metrics.mean_latency_ms)
            return (metrics.throughput_samples_per_sec * 0.6) + (lat_factor * 0.4) * max(0.1, metrics.accuracy)

        return metrics.throughput_samples_per_sec
