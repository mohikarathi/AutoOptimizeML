"""Tests for Baseline Profiler and Bottleneck Analyzer."""

import pytest
from autoopt.models import get_adapter
from autoopt.analyzer import ModelAnalyzer
from autoopt.profiling import BaselineProfiler, BottleneckAnalyzer
import examples.pytorch_cnn as ex_pytorch


def test_baseline_profiler_and_bottleneck():
    model = ex_pytorch.get_model()
    sample = ex_pytorch.get_sample_input(1)
    test_data = ex_pytorch.get_test_data(20)

    adapter = get_adapter(model, sample_input=sample, test_data=test_data)
    profile = ModelAnalyzer.analyze(adapter)

    baseline = BaselineProfiler.profile(
        adapter=adapter,
        device="cpu",
        precision="fp32",
        batch_size=1,
        warmup_runs=2,
        measured_runs=5
    )

    assert baseline.total_latency_ms > 0
    assert baseline.throughput_samples_per_sec > 0
    assert "Baseline Profile" in baseline.format_cli()

    report = BottleneckAnalyzer.analyze(baseline, profile)
    assert report.total_latency_ms == baseline.total_latency_ms
    assert len(report.high_priority_optimizations) > 0
    assert "Bottleneck Analysis" in report.format_cli()
