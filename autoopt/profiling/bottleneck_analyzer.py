"""Automatic Bottleneck Analyzer and Optimization Recommender."""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
from autoopt.profiling.baseline_profiler import BaselineResult
from autoopt.analyzer.model_analyzer import ModelProfile


@dataclass
class BottleneckReport:
    total_latency_ms: float
    stage_percentages: Dict[str, float]
    primary_bottleneck: str
    primary_bottleneck_description: str
    high_priority_optimizations: List[str]
    moderate_priority_optimizations: List[str]
    low_priority_optimizations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_cli(self) -> str:
        lines = [
            "=" * 50,
            "            Bottleneck Analysis & Insights",
            "=" * 50,
            f"Total Latency:          {self.total_latency_ms:.2f} ms",
            "-" * 50,
            "Execution Stage Latency Distribution:",
            f"  Preprocessing:        {self.stage_percentages['preprocessing']:.1f}%",
            f"  Host→Device Transfer: {self.stage_percentages['h2d_transfer']:.1f}%",
            f"  Model Inference:      {self.stage_percentages['inference']:.1f}%",
            f"  Device→Host Transfer: {self.stage_percentages['d2h_transfer']:.1f}%",
            f"  Postprocessing:       {self.stage_percentages['postprocessing']:.1f}%",
            "-" * 50,
            f"Primary Bottleneck:     {self.primary_bottleneck}",
            f"Diagnosis:              {self.primary_bottleneck_description}",
            "-" * 50,
            "Optimization Engine Directives:",
            "  High Priority Focus:"
        ]
        for opt in self.high_priority_optimizations:
            lines.append(f"    [✓] {opt}")

        if self.moderate_priority_optimizations:
            lines.append("  Moderate Priority:")
            for opt in self.moderate_priority_optimizations:
                lines.append(f"    [•] {opt}")

        if self.low_priority_optimizations:
            lines.append("  Low Priority / Skip:")
            for opt in self.low_priority_optimizations:
                lines.append(f"    [○] {opt}")

        lines.append("=" * 50)
        return "\n".join(lines)


class BottleneckAnalyzer:
    """Analyzes execution stage breakdowns to identify system bottlenecks and guide search."""

    @staticmethod
    def analyze(baseline: BaselineResult, model_profile: ModelProfile) -> BottleneckReport:
        total = max(1e-6, baseline.total_latency_ms)
        prep_pct = (baseline.preprocessing_ms / total) * 100.0
        h2d_pct = (baseline.h2d_transfer_ms / total) * 100.0
        infer_pct = (baseline.inference_ms / total) * 100.0
        d2h_pct = (baseline.d2h_transfer_ms / total) * 100.0
        post_pct = (baseline.postprocessing_ms / total) * 100.0

        percentages = {
            "preprocessing": round(prep_pct, 1),
            "h2d_transfer": round(h2d_pct, 1),
            "inference": round(infer_pct, 1),
            "d2h_transfer": round(d2h_pct, 1),
            "postprocessing": round(post_pct, 1),
        }

        high_priority = []
        moderate_priority = []
        low_priority = []

        # 1. Inference Compute Bound
        if infer_pct >= 50.0:
            primary = "Compute-Bound (Model Inference)"
            desc = f"Model forward pass consumes {infer_pct:.1f}% of total execution time. Optimization should focus on execution parallelism, batch scaling, and precision reduction."
            high_priority.append("Batch size exploration (maximize GPU/CPU compute saturation)")
            if "fp16" in model_profile.supported_precisions or "int8" in model_profile.supported_precisions:
                high_priority.append("Precision optimization (FP16 / dynamic INT8 quantization)")
            if model_profile.framework == "pytorch":
                high_priority.append("TorchScript JIT graph compilation")
            moderate_priority.append("Multi-threading / Worker tuning")
            low_priority.append("Native preprocessing acceleration (minor impact on compute-bound workload)")

        # 2. Preprocessing Bound
        elif prep_pct >= 30.0:
            primary = "Preprocessing-Bound (Data Ingestion & Transform)"
            desc = f"Data preprocessing and array formatting dominates at {prep_pct:.1f}% of execution time. Model inference is under-fed."
            high_priority.append("C++ OpenMP / CUDA native batched preprocessing kernels")
            high_priority.append("Fused normalization and channel-transpose operations")
            moderate_priority.append("Async pipeline concurrency")
            low_priority.append("Aggressive model pruning / quantization (inference is already fast)")

        # 3. Host-Device Transfer Bound
        elif (h2d_pct + d2h_pct) >= 25.0:
            primary = "Memory Transfer-Bound (PCIe Host ↔ Device Overhead)"
            desc = f"Host-Device data transfers account for {(h2d_pct + d2h_pct):.1f}% of latency."
            high_priority.append("Pinned memory allocation & non-blocking asynchronous CUDA streams")
            high_priority.append("Larger batch sizes (amortize PCIe dispatch overhead)")
            moderate_priority.append("Fused on-device preprocessing")
            low_priority.append("Thread concurrency tuning")

        else:
            primary = "Balanced Multi-Stage Workload"
            desc = "Latency is distributed across preprocessing and model execution."
            high_priority.append("Batch size exploration")
            high_priority.append("Worker thread concurrency")
            moderate_priority.append("Precision scaling")
            moderate_priority.append("Native C++ preprocessing acceleration")

        return BottleneckReport(
            total_latency_ms=baseline.total_latency_ms,
            stage_percentages=percentages,
            primary_bottleneck=primary,
            primary_bottleneck_description=desc,
            high_priority_optimizations=high_priority,
            moderate_priority_optimizations=moderate_priority,
            low_priority_optimizations=low_priority
        )
