"""Bayesian Optimization Search Strategy using Gaussian Process Surrogates."""

import warnings
import numpy as np
from typing import List, Optional, Callable, Dict, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.stats import norm

from autoopt.models.base import ModelAdapter
from autoopt.benchmark.runner import BenchmarkRunner
from autoopt.benchmark.metrics import BenchmarkMetrics
from autoopt.optimizer.space import OptimizationCandidate
from autoopt.optimizer.constraints import OptimizationConstraints, ConstraintEvaluator, EvaluationResult


class BayesianOptimizer:
    """Intelligent Bayesian search using Gaussian Process surrogate modeling and Expected Improvement."""

    @staticmethod
    def _encode_candidate(candidate: OptimizationCandidate) -> np.ndarray:
        """Encode candidate configuration as a continuous numerical feature vector."""
        feat = [
            1.0 if candidate.device == "cuda" else 0.0,
            1.0 if candidate.precision == "fp16" else (0.5 if candidate.precision == "int8" else 0.0),
            float(np.log2(max(1, candidate.batch_size))),
            float(candidate.workers),
            1.0 if candidate.compile_graph else 0.0,
            1.0 if candidate.native_preprocessing else 0.0
        ]
        return np.array(feat, dtype=np.float64)

    @staticmethod
    def _expected_improvement(X: np.ndarray, model: GaussianProcessRegressor, y_best: float, xi: float = 0.01) -> np.ndarray:
        """Calculate Expected Improvement (EI) acquisition values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mu, sigma = model.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-6)

        improvement = mu - y_best - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    @classmethod
    def search(
        cls,
        adapter: ModelAdapter,
        candidates: List[OptimizationCandidate],
        constraints: OptimizationConstraints,
        budget: int = 15,
        baseline_metrics: Optional[BenchmarkMetrics] = None,
        progress_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None
    ) -> List[EvaluationResult]:
        if not candidates:
            return []

        budget = min(budget, len(candidates))
        candidate_pool = list(candidates)
        X_all = np.array([cls._encode_candidate(c) for c in candidate_pool])

        evaluated_indices = []
        results: List[EvaluationResult] = []

        # 1. Warm-up with diverse initial candidates (e.g. min, median, max batch size)
        init_sample_count = min(4, budget)
        step_stride = max(1, len(candidate_pool) // init_sample_count)
        initial_indices = [i * step_stride for i in range(init_sample_count) if i * step_stride < len(candidate_pool)]
        if 0 not in initial_indices:
            initial_indices.insert(0, 0)
        if (len(candidate_pool) - 1) not in initial_indices and len(initial_indices) < init_sample_count:
            initial_indices.append(len(candidate_pool) - 1)

        for idx in initial_indices[:init_sample_count]:
            if idx in evaluated_indices:
                continue
            evaluated_indices.append(idx)
            cand = candidate_pool[idx]
            try:
                metrics = BenchmarkRunner.benchmark(
                    adapter=adapter,
                    device=cand.device,
                    precision=cand.precision,
                    batch_size=cand.batch_size,
                    workers=cand.workers,
                    compile_graph=cand.compile_graph,
                    native_preprocessing=cand.native_preprocessing,
                    warmup_runs=3,
                    measured_runs=15
                )
                eval_res = ConstraintEvaluator.evaluate(cand, metrics, constraints, baseline_metrics)
            except Exception as e:
                eval_res = EvaluationResult(cand, None, "FAILED", [], -1e9, str(e))
            
            results.append(eval_res)
            if progress_callback:
                progress_callback(len(results), budget, eval_res)

        # 2. Bayesian Optimization Iterations with Gaussian Process
        kernel = Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=2.5) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=1, random_state=42)

        while len(results) < budget:
            X_train = X_all[evaluated_indices]
            y_train = np.array([r.score for r in results])
            y_best = np.max(y_train)

            # Fit GP surrogate model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    gp.fit(X_train, y_train)
                except Exception:
                    pass

            # Compute Expected Improvement over remaining unvisited candidate pool
            unvisited_indices = [i for i in range(len(candidate_pool)) if i not in evaluated_indices]
            if not unvisited_indices:
                break

            X_unvisited = X_all[unvisited_indices]
            ei_scores = cls._expected_improvement(X_unvisited, gp, y_best)
            best_unvisited_idx = unvisited_indices[int(np.argmax(ei_scores))]

            evaluated_indices.append(best_unvisited_idx)
            cand = candidate_pool[best_unvisited_idx]

            try:
                metrics = BenchmarkRunner.benchmark(
                    adapter=adapter,
                    device=cand.device,
                    precision=cand.precision,
                    batch_size=cand.batch_size,
                    workers=cand.workers,
                    compile_graph=cand.compile_graph,
                    native_preprocessing=cand.native_preprocessing,
                    warmup_runs=3,
                    measured_runs=15
                )
                eval_res = ConstraintEvaluator.evaluate(cand, metrics, constraints, baseline_metrics)
            except Exception as e:
                eval_res = EvaluationResult(cand, None, "FAILED", [], -1e9, str(e))

            results.append(eval_res)
            if progress_callback:
                progress_callback(len(results), budget, eval_res)

        return results
