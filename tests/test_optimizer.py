"""Tests for SearchSpace, Constraints, and OptimizationEngine."""

import pytest
from autoopt.models import get_adapter
from autoopt.hardware import HardwareProfiler
from autoopt.analyzer import ModelAnalyzer
from autoopt.optimizer import (
    SearchSpace,
    OptimizationConstraints,
    ConstraintEvaluator,
    OptimizationEngine
)
from autoopt.benchmark import BenchmarkMetrics
import examples.sklearn_random_forest as ex_sklearn


def test_search_space_generation():
    hw = HardwareProfiler.profile()
    model = ex_sklearn.get_model()
    adapter = get_adapter(model, sample_input=ex_sklearn.get_sample_input(1))
    profile = ModelAnalyzer.analyze(adapter)

    candidates = SearchSpace.generate_candidates(hw, profile)
    assert len(candidates) > 0
    for c in candidates:
        assert c.device in ("cpu", "cuda")
        assert c.batch_size in (1, 2, 4, 8, 16, 32)
        assert c.workers >= 1


def test_constraint_evaluator():
    hw = HardwareProfiler.profile()
    model = ex_sklearn.get_model()
    adapter = get_adapter(model, sample_input=ex_sklearn.get_sample_input(1))
    candidates = SearchSpace.generate_candidates(hw, ModelAnalyzer.analyze(adapter))
    cand = candidates[0]

    # Metrics that meet constraints
    mock_metrics = BenchmarkMetrics(
        mean_latency_ms=10.0,
        std_latency_ms=0.5,
        p50_latency_ms=10.0,
        p90_latency_ms=11.0,
        p95_latency_ms=11.5,
        p99_latency_ms=12.0,
        min_latency_ms=9.0,
        max_latency_ms=13.0,
        throughput_req_per_sec=100.0,
        throughput_samples_per_sec=100.0,
        memory_allocated_mb=500.0,
        peak_memory_mb=500.0,
        accuracy=0.95,
        warmup_runs=1,
        measured_runs=5
    )

    # 1. Accepted evaluation
    constraints_ok = OptimizationConstraints(min_accuracy=0.90, max_latency_ms=15.0)
    res_ok = ConstraintEvaluator.evaluate(cand, mock_metrics, constraints_ok)
    assert res_ok.status == "ACCEPTED"
    assert len(res_ok.rejection_reasons) == 0

    # 2. Rejected evaluation
    constraints_fail = OptimizationConstraints(min_accuracy=0.99, max_latency_ms=5.0)
    res_fail = ConstraintEvaluator.evaluate(cand, mock_metrics, constraints_fail)
    assert res_fail.status == "REJECTED"
    assert len(res_fail.rejection_reasons) == 2


def test_optimization_engine_end_to_end():
    model = ex_sklearn.get_model(n_estimators=10)
    sample = ex_sklearn.get_sample_input(1)
    test_data = ex_sklearn.get_test_data()

    constraints = OptimizationConstraints(objective="maximize_throughput")
    report = OptimizationEngine.run(
        model_or_adapter=model,
        sample_input=sample,
        test_data=test_data,
        constraints=constraints,
        strategy="grid"
    )

    assert report.total_candidates_explored > 0
    assert report.best_configuration is not None
    assert "AutoOptimizeML Report" in report.format_cli()
