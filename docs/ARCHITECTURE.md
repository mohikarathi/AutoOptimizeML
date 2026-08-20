# AutoOptimizeML Architecture Guide

AutoOptimizeML is a hardware-agnostic platform engineered to profile, benchmark, optimize, and deploy machine learning and deep learning workloads across heterogeneous compute architectures.

---

## 1. High-Level Pipeline

```text
               +-------------------------------------------------------+
               |                  ML / DL WORKLOAD                     |
               |  (PyTorch nn.Module, TF/Keras, Scikit-Learn Pipeline) |
               +---------------------------+---------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |                    Model Analyzer                     |
               |   - Graph & layer introspection                       |
               |   - Parameter count & memory size calculation         |
               |   - Analytical FLOPs / compute complexity             |
               +---------------------------+---------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |                   Hardware Profiler                   |
               |   - CPU topology, cores & AVX/SIMD extensions         |
               |   - GPU device capabilities & VRAM                    |
               |   - Runtimes (CUDA, PyTorch, TensorFlow)              |
               +---------------------------+---------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |                   Baseline Profiler                   |
               |   - Warmup & multi-iteration measurement              |
               |   - Stage timing: Prep, H2D, Inference, D2H, Post     |
               |   - Latency percentiles & baseline throughput         |
               +---------------------------+---------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |                  Bottleneck Analyzer                  |
               |   - Pipeline percentage distribution                  |
               |   - Compute-bound vs Prep-bound vs Transfer-bound     |
               |   - Targeted optimization directives                  |
               +---------------------------+---------------------------+
                                           |
                                           v
+--------------------------------------------------------------------------------------+
|                                 OPTIMIZATION ENGINE                                  |
|                                                                                      |
|   +--------------------------+  +--------------------------+  +-------------------+  |
|   |   Runtime Optimization   |  |   Model Optimization     |  | Backend Opt       |  |
|   |   - Batch Size (1..64)   |  |   - Precision (FP32/16)  |  | - CPU / CUDA      |  |
|   |   - Worker Threads (1..8)|  |   - Dynamic INT8 Quant   |  | - Native C++/CUDA |  |
|   |   - Concurrency Level    |  |   - TorchScript JIT/XLA  |  | - Memory Pinning  |  |
|   +--------------------------+  +--------------------------+  +-------------------+  |
|                                                                                      |
|   Search Strategies:                                                                 |
|     1. Grid Search (exhaustive valid candidate exploration)                          |
|     2. Bayesian Optimization (Gaussian Process surrogate + Expected Improvement)     |
|                                                                                      |
|   Constraint Verification:                                                           |
|     - Min Accuracy, Max Latency, Max Memory, Min Throughput                          |
+------------------------------------------+-------------------------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |                   Benchmark Engine                    |
               |   - Repeatable timing harness                         |
               |   - P50, P90, P95, P99 latency percentiles            |
               |   - Memory allocation & peak tracking                 |
               +---------------------------+---------------------------+
                                           |
                                           v
               +-------------------------------------------------------+
               |                  Deployment Engine                    |
               |   - High-throughput FastAPI REST API                  |
               |   - Asynchronous dynamic request batching queue       |
               |   - Real-time telemetry: queue wait vs inference      |
               +-------------------------------------------------------+
```

---

## 2. Component Design Principles

### Model Adapters (`autoopt/models/`)
The optimization engine never interacts directly with underlying framework specifics. Instead, framework behaviors are encapsulated behind the `ModelAdapter` interface:
* `SklearnAdapter`: Classical ML models, parameter sizing, multi-core `n_jobs`.
* `PyTorchAdapter`: Deep learning graphs, FP16/AMP, dynamic INT8 quantization, TorchScript JIT.
* `TensorFlowAdapter`: Keras models, mixed-precision policies, XLA compilation.

### Execution Backends (`autoopt/backends/`)
Hardware execution is abstracted via `ExecutionBackend`:
* `CPUBackend`: Multi-threaded CPU execution with OpenMP, MKL, and intra-op thread allocation.
* `CUDABackend`: CUDA stream synchronization, VRAM queries, and GPU memory tracking.
* `ROCmBackend`: Interface and architectural specification for AMD Instinct/Radeon GPUs.

### Native Acceleration (`native/`)
Built with CMake and PyBind11, the native C++/CUDA layer provides fused tensor transformations and image normalization:
* **CPU Backend**: Multithreaded OpenMP C++ with SIMD vectorization.
* **CUDA Backend**: Custom CUDA kernels fusing normalization and NHWC -> NCHW channel transpositions.
