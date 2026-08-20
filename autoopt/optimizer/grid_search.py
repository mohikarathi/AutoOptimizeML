"""Grid Search Optimization Strategy."""

import time
from typing import List, Optional, Callable
from autoopt.models.base import ModelAdapter
from autoopt.benchmark.runner import BenchmarkRunner
from autoopt.benchmark.metrics import BenchmarkMetrics
from autoopt.optimizer.space import OptimizationCandidate
from autoopt.optimizer.constraints import OptimizationConstraints, ConstraintEvaluator, EvaluationResult


class GridSearchOptimizer:
    """Exhaustive search across all valid optimization candidate configurations."""

    @staticmethod
    def search(
        adapter: ModelAdapter,
        candidates: List[OptimizationCandidate],
        constraints: OptimizationConstraints,
        baseline_metrics: Optional[BenchmarkMetrics] = None,
        progress_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None
    ) -> List[EvaluationResult]:
        results: List[EvaluationResult] = []
        total = len(candidates)

        for idx, candidate in enumerate(candidates):
            try:
                metrics = BenchmarkRunner.benchmark(
                    adapter=adapter,
                    device=candidate.device,
                    precision=candidate.precision,
                    batch_size=candidate.batch_size,
                    workers=candidate.workers,
                    compile_graph=candidate.compile_graph,
                    native_preprocessing=candidate.native_preprocessing,
                    warmup_runs=3,
                    measured_runs=15
                )
                eval_res = ConstraintEvaluator.evaluate(
                    candidate=candidate,
                    metrics=metrics,
                    constraints=constraints,
                    baseline_metrics=baseline_metrics
                )
            except Exception as e:
                eval_res = EvaluationResult(
                    candidate=candidate,
                    metrics=None,
                    status="FAILED",
                    rejection_reasons=[],
                    score=-1e9,
                    error_message=str(e)
                )

            results.append(eval_res)
            if progress_callback:
                progress_callback(idx + 1, total, eval_res)

        return results
