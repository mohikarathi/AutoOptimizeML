"""Repeatable benchmark metrics and percentile calculation."""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List


@dataclass
class BenchmarkMetrics:
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_req_per_sec: float
    throughput_samples_per_sec: float
    memory_allocated_mb: float
    peak_memory_mb: float
    accuracy: float
    warmup_runs: int
    measured_runs: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def compute(
        cls,
        latencies_ms: List[float],
        batch_size: int,
        memory_mb: float,
        peak_memory_mb: float,
        accuracy: float,
        warmup_runs: int,
        measured_runs: int
    ) -> "BenchmarkMetrics":
        arr = np.asarray(latencies_ms, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        p50 = float(np.percentile(arr, 50))
        p90 = float(np.percentile(arr, 90))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        min_l = float(np.min(arr))
        max_l = float(np.max(arr))

        req_per_sec = (1000.0 / mean) if mean > 0 else 0.0
        samp_per_sec = req_per_sec * batch_size

        return cls(
            mean_latency_ms=round(mean, 3),
            std_latency_ms=round(std, 3),
            p50_latency_ms=round(p50, 3),
            p90_latency_ms=round(p90, 3),
            p95_latency_ms=round(p95, 3),
            p99_latency_ms=round(p99, 3),
            min_latency_ms=round(min_l, 3),
            max_latency_ms=round(max_l, 3),
            throughput_req_per_sec=round(req_per_sec, 2),
            throughput_samples_per_sec=round(samp_per_sec, 2),
            memory_allocated_mb=round(memory_mb, 2),
            peak_memory_mb=round(peak_memory_mb, 2),
            accuracy=round(accuracy, 4),
            warmup_runs=warmup_runs,
            measured_runs=measured_runs
        )
