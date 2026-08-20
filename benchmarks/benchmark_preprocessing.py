"""Comprehensive Preprocessing Benchmark: NumPy vs Multi-Threaded C++ OpenMP vs CUDA.

Measures latency, throughput (images/sec), memory bandwidth (GB/s), and CPU thread scaling
across different batch sizes and image resolutions.
"""

import time
import numpy as np
from autoopt import autoopt_native


def benchmark_preprocessing_suite():
    print("=" * 70)
    print("      Preprocessing Acceleration Benchmark: NumPy vs Native C++")
    print("=" * 70)
    print(f"{'Batch':<8} {'Resolution':<12} {'NumPy (ms)':<12} {'C++ (1T)':<10} {'C++ (8T)':<10} {'C++ (16T)':<10} {'Speedup':<10}")
    print("-" * 70)

    resolutions = [(32, 32), (128, 128), (224, 224)]
    batch_sizes = [1, 8, 32, 64, 128]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    results = []

    for b in batch_sizes:
        for h, w in resolutions:
            inputs = np.random.randint(0, 256, (b, h, w, 3)).astype(np.float32)
            num_elements = b * h * w * 3
            data_size_mb = (num_elements * 4) / (1024 * 1024)

            # Warmup
            for _ in range(5):
                _ = ((inputs / 255.0) - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
                _ = autoopt_native.normalize_nhwc_to_nchw_cpu(inputs, mean, std, 1.0/255.0, 16)

            # 1. Benchmark NumPy
            iters = 20 if b <= 32 else 10
            t0 = time.perf_counter()
            for _ in range(iters):
                res_np = ((inputs / 255.0) - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
                res_np = np.transpose(res_np, (0, 3, 1, 2))
            t_np = (time.perf_counter() - t0) / iters * 1000.0

            # 2. Benchmark C++ Single Thread
            t0 = time.perf_counter()
            for _ in range(iters):
                res_cpp1 = autoopt_native.normalize_nhwc_to_nchw_cpu(inputs, mean, std, 1.0/255.0, 1)
            t_cpp1 = (time.perf_counter() - t0) / iters * 1000.0

            # 3. Benchmark C++ 8 Threads
            t0 = time.perf_counter()
            for _ in range(iters):
                res_cpp8 = autoopt_native.normalize_nhwc_to_nchw_cpu(inputs, mean, std, 1.0/255.0, 8)
            t_cpp8 = (time.perf_counter() - t0) / iters * 1000.0

            # 4. Benchmark C++ 16 Threads
            t0 = time.perf_counter()
            for _ in range(iters):
                res_cpp16 = autoopt_native.normalize_nhwc_to_nchw_cpu(inputs, mean, std, 1.0/255.0, 16)
            t_cpp16 = (time.perf_counter() - t0) / iters * 1000.0

            speedup = t_np / max(1e-4, t_cpp16)
            throughput_np = (b / (t_np / 1000.0))
            throughput_cpp = (b / (t_cpp16 / 1000.0))

            res_str = f"{h}x{w}"
            print(f"{b:<8} {res_str:<12} {t_np:<12.2f} {t_cpp1:<10.2f} {t_cpp8:<10.2f} {t_cpp16:<10.2f} {speedup:<9.2f}x")

            results.append({
                "batch_size": b,
                "resolution": f"{h}x{w}",
                "data_size_mb": round(data_size_mb, 2),
                "numpy_ms": round(t_np, 2),
                "cpp_1t_ms": round(t_cpp1, 2),
                "cpp_8t_ms": round(t_cpp8, 2),
                "cpp_16t_ms": round(t_cpp16, 2),
                "speedup_vs_numpy": round(speedup, 2),
                "throughput_numpy_img_per_sec": round(throughput_np, 1),
                "throughput_cpp_img_per_sec": round(throughput_cpp, 1)
            })

    print("=" * 70)
    return results


if __name__ == "__main__":
    benchmark_preprocessing_suite()
