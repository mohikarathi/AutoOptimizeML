"""Baseline execution profiler with stage-by-stage timing breakdowns."""

import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from autoopt.models.base import ModelAdapter
from autoopt.backends import get_backend, ExecutionBackend


@dataclass
class BaselineResult:
    device: str
    precision: str
    batch_size: int
    warmup_runs: int
    measured_runs: int
    preprocessing_ms: float
    h2d_transfer_ms: float
    inference_ms: float
    d2h_transfer_ms: float
    postprocessing_ms: float
    total_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_req_per_sec: float
    throughput_samples_per_sec: float
    memory_mb: float
    peak_memory_mb: float
    accuracy: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_cli(self) -> str:
        lines = [
            "=" * 50,
            "               Baseline Profile",
            "=" * 50,
            f"Device:                 {self.device.upper()}",
            f"Precision:              {self.precision.upper()}",
            f"Batch Size:             {self.batch_size}",
            f"Baseline Accuracy:      {self.accuracy * 100:.2f}%",
            "-" * 50,
            "Latency & Throughput:",
            f"  Mean Total Latency:   {self.total_latency_ms:.2f} ms",
            f"  P50 Latency:          {self.p50_latency_ms:.2f} ms",
            f"  P95 Latency:          {self.p95_latency_ms:.2f} ms",
            f"  P99 Latency:          {self.p99_latency_ms:.2f} ms",
            f"  Throughput (req/s):   {self.throughput_req_per_sec:.2f} req/s",
            f"  Throughput (samp/s):  {self.throughput_samples_per_sec:.2f} samples/s",
            "-" * 50,
            "Stage Timing Breakdown:",
            f"  Preprocessing:        {self.preprocessing_ms:.2f} ms",
            f"  Host→Device Transfer: {self.h2d_transfer_ms:.2f} ms",
            f"  Inference Compute:    {self.inference_ms:.2f} ms",
            f"  Device→Host Transfer: {self.d2h_transfer_ms:.2f} ms",
            f"  Postprocessing:       {self.postprocessing_ms:.2f} ms",
            "-" * 50,
            f"Memory Allocation:      {self.memory_mb:.2f} MB (Peak: {self.peak_memory_mb:.2f} MB)",
            "=" * 50
        ]
        return "\n".join(lines)


class BaselineProfiler:
    """Measures precise multi-stage baseline metrics before optimization."""

    @staticmethod
    def profile(
        adapter: ModelAdapter,
        device: str = "cpu",
        precision: str = "fp32",
        batch_size: int = 1,
        warmup_runs: int = 10,
        measured_runs: int = 50,
        backend: Optional[ExecutionBackend] = None
    ) -> BaselineResult:
        if backend is None:
            backend = get_backend(device)

        backend.reset_peak_memory()
        prepared_model = adapter.prepare_for_inference(device=device, precision=precision)
        raw_sample = adapter.sample_input

        # Warmup iterations
        for _ in range(warmup_runs):
            prep = adapter.preprocess(raw_sample)
            if device == "cuda" and hasattr(prep, "to"):
                prep = prep.to("cuda")
            backend.synchronize()
            out = adapter.run_inference(prepared_model, prep)
            backend.synchronize()
            _ = adapter.postprocess(out)

        prep_times = []
        h2d_times = []
        infer_times = []
        d2h_times = []
        post_times = []
        total_times = []

        for _ in range(measured_runs):
            t0 = time.perf_counter()
            
            # Preprocessing
            t_p0 = time.perf_counter()
            prep_input = adapter.preprocess(raw_sample)
            t_p1 = time.perf_counter()

            # Host-to-Device Transfer (if tensor on device)
            t_h0 = time.perf_counter()
            if device == "cuda" and hasattr(prep_input, "to"):
                prep_input = prep_input.to("cuda")
                backend.synchronize()
            t_h1 = time.perf_counter()

            # Inference Compute
            t_i0 = time.perf_counter()
            raw_output = adapter.run_inference(prepared_model, prep_input)
            backend.synchronize()
            t_i1 = time.perf_counter()

            # Device-to-Host Transfer
            t_d0 = time.perf_counter()
            if device == "cuda" and hasattr(raw_output, "cpu"):
                raw_output = raw_output.cpu()
                backend.synchronize()
            t_d1 = time.perf_counter()

            # Postprocessing
            t_post0 = time.perf_counter()
            _ = adapter.postprocess(raw_output)
            t_post1 = time.perf_counter()

            t1 = time.perf_counter()

            prep_times.append((t_p1 - t_p0) * 1000.0)
            h2d_times.append((t_h1 - t_h0) * 1000.0)
            infer_times.append((t_i1 - t_i0) * 1000.0)
            d2h_times.append((t_d1 - t_d0) * 1000.0)
            post_times.append((t_post1 - t_post0) * 1000.0)
            total_times.append((t1 - t0) * 1000.0)

        mean_total = float(np.mean(total_times))
        p50 = float(np.percentile(total_times, 50))
        p95 = float(np.percentile(total_times, 95))
        p99 = float(np.percentile(total_times, 99))

        req_per_sec = (1000.0 / mean_total) if mean_total > 0 else 0.0
        samp_per_sec = req_per_sec * batch_size

        accuracy = adapter.evaluate_accuracy(prepared_model)

        return BaselineResult(
            device=device,
            precision=precision,
            batch_size=batch_size,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            preprocessing_ms=round(float(np.mean(prep_times)), 3),
            h2d_transfer_ms=round(float(np.mean(h2d_times)), 3),
            inference_ms=round(float(np.mean(infer_times)), 3),
            d2h_transfer_ms=round(float(np.mean(d2h_times)), 3),
            postprocessing_ms=round(float(np.mean(post_times)), 3),
            total_latency_ms=round(mean_total, 3),
            p50_latency_ms=round(p50, 3),
            p95_latency_ms=round(p95, 3),
            p99_latency_ms=round(p99, 3),
            throughput_req_per_sec=round(req_per_sec, 2),
            throughput_samples_per_sec=round(samp_per_sec, 2),
            memory_mb=backend.get_memory_allocated_mb(),
            peak_memory_mb=backend.get_peak_memory_mb(),
            accuracy=accuracy
        )
