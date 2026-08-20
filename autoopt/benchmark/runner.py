"""Repeatable benchmark execution harness."""

import time
import numpy as np
from typing import Dict, Any, Optional
from autoopt.models.base import ModelAdapter
from autoopt.backends import get_backend, ExecutionBackend
from autoopt.benchmark.metrics import BenchmarkMetrics


class BenchmarkRunner:
    """Executes repeatable warmup and multi-run latency/throughput measurements."""

    @staticmethod
    def benchmark(
        adapter: ModelAdapter,
        device: str = "cpu",
        precision: str = "fp32",
        batch_size: int = 1,
        workers: int = 1,
        compile_graph: bool = False,
        native_preprocessing: bool = False,
        warmup_runs: int = 5,
        measured_runs: int = 30,
        backend: Optional[ExecutionBackend] = None
    ) -> BenchmarkMetrics:
        if backend is None:
            backend = get_backend(device)

        backend.set_num_threads(workers)
        backend.reset_peak_memory()

        prepared_model = adapter.prepare_for_inference(
            device=device,
            precision=precision,
            compile_graph=compile_graph,
            num_threads=workers
        )

        # Scale sample input to match batch size
        raw_sample = adapter.sample_input
        if raw_sample is not None:
            if hasattr(raw_sample, "shape") and raw_sample.shape[0] != batch_size:
                if isinstance(raw_sample, np.ndarray):
                    # Tile or slice to batch_size
                    tiles = int(np.ceil(batch_size / raw_sample.shape[0]))
                    tiled = np.tile(raw_sample, (tiles, *([1] * (raw_sample.ndim - 1))))
                    raw_sample = tiled[:batch_size]
                elif hasattr(raw_sample, "repeat"):
                    import torch
                    tiles = int(np.ceil(batch_size / raw_sample.shape[0]))
                    tiled = raw_sample.repeat(tiles, *([1] * (raw_sample.ndim - 1)))
                    raw_sample = tiled[:batch_size]

        # Warmup runs
        for _ in range(warmup_runs):
            prep = adapter.preprocess(raw_sample)
            if device == "cuda" and hasattr(prep, "to"):
                prep = prep.to("cuda")
            backend.synchronize()
            out = adapter.run_inference(prepared_model, prep)
            backend.synchronize()
            _ = adapter.postprocess(out)

        latencies_ms = []

        # Measured runs
        for _ in range(measured_runs):
            t0 = time.perf_counter()

            # Preprocessing
            prep = adapter.preprocess(raw_sample)
            if device == "cuda" and hasattr(prep, "to"):
                prep = prep.to("cuda")

            backend.synchronize()
            t_start_infer = time.perf_counter()

            # Inference
            out = adapter.run_inference(prepared_model, prep)
            backend.synchronize()

            # Postprocessing
            _ = adapter.postprocess(out)

            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        # Evaluate accuracy on test dataset
        accuracy = adapter.evaluate_accuracy(prepared_model)

        mem_mb = backend.get_memory_allocated_mb()
        peak_mem_mb = backend.get_peak_memory_mb()

        return BenchmarkMetrics.compute(
            latencies_ms=latencies_ms,
            batch_size=batch_size,
            memory_mb=mem_mb,
            peak_memory_mb=peak_mem_mb,
            accuracy=accuracy,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs
        )
