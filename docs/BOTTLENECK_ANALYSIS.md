# Automatic Bottleneck Analysis

Rather than blindly evaluating random configuration permutations, AutoOptimizeML performs **data-driven bottleneck diagnosis** before initiating the optimization search.

---

## 1. Stage-by-Stage Latency Decomposition

Total end-to-end inference latency is measured as the sum of distinct pipeline stages:

\[
T_{\text{total}} = T_{\text{prep}} + T_{\text{H2D}} + T_{\text{infer}} + T_{\text{D2H}} + T_{\text{post}}
\]

where:
* \( T_{\text{prep}} \): Data loading, tensor formatting, image normalization, type casting.
* \( T_{\text{H2D}} \): Host-to-Device transfer (system RAM $\rightarrow$ GPU VRAM via PCIe bus).
* \( T_{\text{infer}} \): Core forward pass compute through model weights.
* \( T_{\text{D2H}} \): Device-to-Host transfer (GPU VRAM $\rightarrow$ system RAM).
* \( T_{\text{post}} \): Argmax/softmax decoding, thresholding, and response packing.

---

## 2. Bottleneck Classification & Optimization Directives

| Diagnosed Condition | Pipeline Metric Threshold | Targeted Optimization Directives |
| :--- | :--- | :--- |
| **Compute-Bound** | \( \frac{T_{\text{infer}}}{T_{\text{total}}} \ge 50\% \) | • Batch size scaling (maximize compute saturation)<br>• Precision reduction (FP16 / Dynamic INT8)<br>• TorchScript JIT / XLA graph compilation |
| **Preprocessing-Bound** | \( \frac{T_{\text{prep}}}{T_{\text{total}}} \ge 30\% \) | • Multi-threaded C++ OpenMP / CUDA native kernels<br>• Fused normalization + channel transposition<br>• Asynchronous pipeline concurrency |
| **Transfer-Bound** | \( \frac{T_{\text{H2D}} + T_{\text{D2H}}}{T_{\text{total}}} \ge 25\% \) | • Pinned memory allocation (`pin_memory=True`)<br>• Non-blocking asynchronous CUDA streams<br>• Larger batch sizes to amortize PCIe dispatch |
| **Balanced** | Distributed evenly | • Joint batch size and worker thread concurrency tuning |

---

## 3. Real-World Output Example

```text
==================================================
            Bottleneck Analysis & Insights
==================================================
Total Latency:          38.40 ms
--------------------------------------------------
Execution Stage Latency Distribution:
  Preprocessing:        6.1%
  Host→Device Transfer: 4.2%
  Model Inference:      85.4%
  Device→Host Transfer: 2.1%
  Postprocessing:       2.2%
--------------------------------------------------
Primary Bottleneck:     Compute-Bound (Model Inference)
Diagnosis:              Model forward pass consumes 85.4% of execution time.
--------------------------------------------------
Optimization Engine Directives:
  High Priority Focus:
    [✓] Batch size exploration (maximize GPU/CPU compute saturation)
    [✓] Precision optimization (FP16 / dynamic INT8 quantization)
    [✓] TorchScript JIT graph compilation
  Moderate Priority:
    [•] Multi-threading / Worker tuning
  Low Priority / Skip:
    [○] Native preprocessing acceleration (minor impact on compute-bound workload)
==================================================
```
