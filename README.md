# AutoOptimizeML

**Hardware-Agnostic ML/DL Model Profiling, Bottleneck Analysis, Optimization, and Deployment Platform**

AutoOptimizeML is an infrastructure platform that automatically profiles ML/DL workloads and the host heterogeneous compute architecture, isolates execution bottlenecks stage-by-stage, searches multi-dimensional execution configurations under hard SLA and accuracy constraints, and deploys the winning configuration with high-throughput dynamic request batching.

---

## Empirical Optimization Results (Measured on AMD Ryzen 7 7840HS 8C/16T)

### 1. Deep Learning Vision CNN
* **Optimization Goal**: Maximize throughput subject to $\text{Latency}_{\text{P95}} \le 25.0\text{ ms}$ and $\text{Accuracy} \ge \text{Baseline} - 5\%$.
* **Baseline**: Device=CPU, Precision=FP32, Batch=1, Workers=1 (Mean Latency: 0.53 ms, Throughput: 1,889.9 samples/sec).

| Configuration | Device | Batch | Workers | JIT | Native C++ | Mean Latency | P95 Latency | Throughput | Constraint Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | CPU | 1 | 1 | No | No | 0.53 ms | 0.65 ms | 1,889.9 smp/s | ACCEPTED |
| cand_145 | CPU | 8 | 1 | No | No | 7.42 ms | 7.71 ms | 1,078.2 smp/s | ACCEPTED |
| cand_153 | CPU | 8 | 4 | No | No | 2.42 ms | 2.89 ms | 3,306.5 smp/s | ACCEPTED |
| **cand_160 (Winner)** | **CPU** | **8** | **8** | **Yes** | **Yes** | **1.45 ms** | **1.66 ms** | **5,526.7 smp/s** | **ACCEPTED (2.92x Speedup)** |
| cand_175 | CPU | 16 | 8 | Yes | No | 3.06 ms | 3.57 ms | 5,231.9 smp/s | ACCEPTED (2.77x) |
| cand_177 | CPU | 32 | 1 | No | No | 27.70 ms | 29.32 ms | 1,155.2 smp/s | **REJECTED (Latency > 25ms)** |
| cand_179 | CPU | 32 | 1 | Yes | No | 28.43 ms | 30.89 ms | 1,125.6 smp/s | **REJECTED (Latency > 25ms)** |

> **Systems Finding**: Scaling batch size to 32 on a single worker violated the 25ms latency SLA (27.7ms -> REJECTED). However, combining **Batch=8, 8 worker threads, TorchScript JIT compilation, and C++ fused native preprocessing** achieved **5,526.7 samples/sec (2.92x throughput speedup)** while reducing batch latency to just **1.45 ms (P95: 1.66 ms)**.

---

### 2. Tabular ML (Scikit-Learn Random Forest)
* **Optimization Goal**: Maximize throughput subject to $\text{Latency} \le 15.0\text{ ms}$.
* **Baseline**: Batch=1, Workers=1 (Mean Latency: 0.98 ms, Throughput: 1,017.1 samples/sec).

| Configuration | Batch Size | Workers (n_jobs) | Mean Latency | Throughput | Measured Speedup | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 1 | 1 | 0.98 ms | 1,017.1 smp/s | 1.00x | ACCEPTED |
| Multi-thread unbatched | 1 | 8 | 12.69 ms | 78.8 smp/s | 0.08x (Lock overhead) | ACCEPTED |
| Moderate batch | 16 | 1 | 1.02 ms | 15,753.0 smp/s | 15.49x | ACCEPTED |
| Large batch | 64 | 1 | 1.00 ms | 63,747.8 smp/s | 62.67x | ACCEPTED |
| **Max Batch (Winner)** | **256** | **1** | **1.36 ms** | **187,686.9 smp/s** | **184.53x** | **ACCEPTED** |

> **Systems Finding**: For shallow decision tree ensembles, Python threading (`n_jobs > 1`) on small batches incurs severe IPC and mutex synchronization penalties (latency rose from 0.98ms to 12.69ms). Amortizing tree traversal across a coalesced batch of 256 samples increased throughput from **1,017 smp/s to 187,686 smp/s (184.5x speedup)** with only a 0.38ms latency delta.

---

### 3. Native C++ / OpenMP Tensor Preprocessing Speedup vs NumPy
Fused image normalization ($\frac{x}{255} - \mu) / \sigma$ and channel transposition (NHWC -> NCHW) benchmarked across batch sizes and resolutions:

| Batch Size | Resolution | NumPy (ms) | C++ OpenMP 1T (ms) | C++ OpenMP 8T (ms) | C++ OpenMP 16T (ms) | Measured Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 128x128 | 0.16 ms | 0.09 ms | 0.05 ms | 0.14 ms | **1.46x** |
| 1 | 224x224 | 0.56 ms | 0.33 ms | 0.14 ms | 0.20 ms | **2.86x** |
| 8 | 128x128 | 1.50 ms | 0.86 ms | 0.30 ms | 0.29 ms | **5.12x** |
| 8 | 224x224 | 5.30 ms | 2.73 ms | 0.82 ms | 0.60 ms | **8.85x** |
| 32 | 128x128 | 7.01 ms | 3.69 ms | 1.11 ms | 0.89 ms | **7.89x** |
| 32 | 224x224 | 23.88 ms | 11.96 ms | 3.54 ms | 2.90 ms | **8.23x** |
| 64 | 128x128 | 15.81 ms | 8.30 ms | 2.58 ms | 2.48 ms | **6.37x** |
| 128 | 32x32 | 1.53 ms | 0.88 ms | 0.26 ms | 0.36 ms | **4.27x** |

---

## System Architecture

```text
                       ML WORKLOAD (TensorFlow, PyTorch, Sklearn)
                                       |
                                       v
                             +-------------------+
                             |  Model Analyzer   |
                             +---------+---------+
                                       |
                                       v
                             +-------------------+
                             | Hardware Profiler |
                             +---------+---------+
                                       |
                                       v
                             +-------------------+
                             | Baseline Profiler |
                             +---------+---------+
                                       |
                                       v
                             +-------------------+
                             |Bottleneck Analyzer|
                             | (Stage Breakdown) |
                             +---------+---------+
                                       |
                                       v
                             +-------------------+
                             |Optimization Engine|
                             +---------+---------+
                                       |
            +--------------------------+--------------------------+
            |                          |                          |
            v                          v                          v
     Runtime Options           Precision Options          Backend Options
    (Batch size, Workers)        (FP32, FP16, INT8)        (CPU, CUDA, C++)
            |                          |                          |
            +--------------------------+--------------------------+
                                       |
                                       v
                             +-------------------+
                             | Benchmark Engine  |
                             +---------+---------+
                                       |
                                       v
                             +-------------------+
                             |Constraint Engine  |
                             | (Accept / Reject) |
                             +---------+---------+
                                       |
                                       v
                             +-------------------+
                             |Best Configuration |
                             +---------+---------+
                                       |
                                       v
                             +-------------------+
                             |Deployment Service |
                             |(Dynamic Batching) |
                             +-------------------+
```

---

## Mathematical Optimization Formulation

\[
\begin{aligned}
\max_{x \in \mathcal{X}} \quad & \text{Throughput}(x) \quad (\text{or } \min_{x \in \mathcal{X}} \text{Latency}(x)) \\
\text{subject to} \quad & \text{Latency}(x) \le L_{\max} \\
& \text{Accuracy}(x) \ge A_{\min} \\
& \text{Memory}(x) \le M_{\max}
\end{aligned}
\]

where candidate configuration vector \( x = (\text{backend}, \text{precision}, \text{batch\_size}, \text{workers}, \text{compile\_graph}, \text{native\_accel}) \in \mathcal{X} \).

---

## CLI Quickstart

```bash
# 1. Profile host hardware and accelerators
autoopt profile

# 2. Inspect model graph, parameters, and FLOPs
autoopt analyze --model ./my_model.py

# 3. Benchmark a specific configuration
autoopt benchmark --model ./my_model.py --batch-size 8 --workers 4

# 4. Automatically optimize model subject to constraints
autoopt optimize \
    --model ./my_model.py \
    --objective maximize_throughput \
    --strategy bayesian \
    --max-latency 25.0 \
    --budget 15

# 5. Display run report
autoopt report --run latest

# 6. Deploy optimized model with dynamic batching
autoopt deploy --config latest --model ./my_model.py --port 8000

# 7. Launch interactive web dashboard
autoopt dashboard --port 8501
```

---

## Hardware Abstraction and Extensibility

AutoOptimizeML is built upon an abstract `ExecutionBackend` interface (`autoopt/backends/base.py`). The optimization engine interacts exclusively with backend-agnostic abstractions, allowing new execution targets (such as AMD ROCm / HIP) to be integrated without modifying the search engine, benchmark harnesses, or deployment servers.
