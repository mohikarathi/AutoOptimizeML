"""Master Optimization Engine orchestrating profiling, search, constraint checking, and deployment readiness."""

import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Callable

from autoopt.hardware import HardwareProfiler, HardwareProfile
from autoopt.models import get_adapter, ModelAdapter
from autoopt.analyzer import ModelAnalyzer, ModelProfile
from autoopt.profiling import BaselineProfiler, BaselineResult, BottleneckAnalyzer, BottleneckReport
from autoopt.optimizer.space import SearchSpace, OptimizationCandidate
from autoopt.optimizer.constraints import OptimizationConstraints, EvaluationResult
from autoopt.optimizer.grid_search import GridSearchOptimizer
from autoopt.optimizer.bayesian_search import BayesianOptimizer
from autoopt.utils.storage import save_run_artifact


@dataclass
class OptimizationRunReport:
    run_id: str
    timestamp: str
    hardware: Dict[str, Any]
    model: Dict[str, Any]
    baseline: Dict[str, Any]
    bottleneck: Dict[str, Any]
    constraints: Dict[str, Any]
    strategy: str
    total_candidates_explored: int
    evaluations: List[Dict[str, Any]]
    best_configuration: Optional[Dict[str, Any]]
    best_metrics: Optional[Dict[str, Any]]
    improvement: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_cli(self) -> str:
        lines = [
            "=" * 60,
            "                   AutoOptimizeML Report",
            "=" * 60,
            f"Run ID:                 {self.run_id}",
            f"Timestamp:              {self.timestamp}",
            f"Search Strategy:        {self.strategy.upper()}",
            "-" * 60,
            "Model Overview:",
            f"  Framework:            {self.model.get('framework', '').upper()}",
            f"  Model Type:           {self.model.get('model_type')}",
            f"  Parameters:           {self.model.get('parameters', 0):,}",
            f"  Model Size:           {self.model.get('model_size_mb', 0)} MB",
            "-" * 60,
            "Hardware Environment:",
            f"  CPU:                  {self.hardware.get('cpu', {}).get('model_name')}",
            f"  Cores:                {self.hardware.get('cpu', {}).get('physical_cores')} Physical / {self.hardware.get('cpu', {}).get('logical_cores')} Logical",
            f"  Available Memory:     {self.hardware.get('cpu', {}).get('available_ram_gb')} GB",
            "-" * 60,
            "Baseline Performance:",
            f"  Device:               {self.baseline.get('device', '').upper()}",
            f"  Precision:            {self.baseline.get('precision', '').upper()}",
            f"  Batch Size:           {self.baseline.get('batch_size')}",
            f"  Latency:              {self.baseline.get('total_latency_ms', 0):.2f} ms",
            f"  Throughput:           {self.baseline.get('throughput_samples_per_sec', 0):.2f} samples/s",
            f"  Accuracy:             {self.baseline.get('accuracy', 0) * 100:.2f}%",
            "-" * 60,
            "Primary Bottleneck:",
            f"  {self.bottleneck.get('primary_bottleneck')}",
            f"  {self.bottleneck.get('primary_bottleneck_description')}",
            "-" * 60,
        ]

        if self.best_configuration and self.best_metrics:
            bc = self.best_configuration
            bm = self.best_metrics
            imp = self.improvement

            jit_str = " (TorchScript JIT)" if bc.get("compile_graph") else ""
            native_str = " (Native C++/CUDA)" if bc.get("native_preprocessing") else ""

            lines.extend([
                "Best Configuration Found:",
                f"  Device:               {bc.get('device', '').upper()}",
                f"  Precision:            {bc.get('precision', '').upper()}",
                f"  Batch Size:           {bc.get('batch_size')}",
                f"  Worker Threads:       {bc.get('workers')}",
                f"  Graph Acceleration:   {bc.get('compile_graph')}{jit_str}",
                f"  Native Preprocessing: {bc.get('native_preprocessing')}{native_str}",
                "",
                "Optimized Performance:",
                f"  Latency (Mean):       {bm.get('mean_latency_ms', 0):.2f} ms (P95: {bm.get('p95_latency_ms', 0):.2f} ms)",
                f"  Throughput:           {bm.get('throughput_samples_per_sec', 0):.2f} samples/s",
                f"  Accuracy:             {bm.get('accuracy', 0) * 100:.2f}%",
                f"  Memory Footprint:     {bm.get('memory_allocated_mb', 0):.2f} MB",
                "",
                "Measured Speedup & Improvement:",
                f"  Latency Reduction:    {imp.get('latency_reduction_pct', 0):.1f}% faster",
                f"  Throughput Boost:     {imp.get('throughput_gain_pct', 0):.1f}% higher",
                f"  Accuracy Delta:       {imp.get('accuracy_delta_pct', 0):+.2f}%",
                "-" * 60,
                "Constraints Evaluation:"
            ])

            # Constraints verification checklist
            acc_con = self.constraints.get("min_accuracy")
            if acc_con is not None:
                passed = bm.get("accuracy", 0) >= acc_con
                lines.append(f"  [{'✓' if passed else '✗'}] Accuracy >= {acc_con*100:.1f}% (Actual: {bm.get('accuracy', 0)*100:.2f}%)")

            lat_con = self.constraints.get("max_latency_ms")
            if lat_con is not None:
                passed = bm.get("mean_latency_ms", 0) <= lat_con
                lines.append(f"  [{'✓' if passed else '✗'}] Latency <= {lat_con:.1f}ms (Actual: {bm.get('mean_latency_ms', 0):.2f}ms)")

            mem_con = self.constraints.get("max_memory_mb")
            if mem_con is not None:
                passed = bm.get("memory_allocated_mb", 0) <= mem_con
                lines.append(f"  [{'✓' if passed else '✗'}] Memory <= {mem_con:.0f}MB (Actual: {bm.get('memory_allocated_mb', 0):.1f}MB)")

        else:
            lines.append("No configuration satisfied all specified hard constraints.")

        lines.append("=" * 60)
        return "\n".join(lines)


class OptimizationEngine:
    """Master controller that searches, benchmarks, filters, and selects optimal configurations."""

    @staticmethod
    def run(
        model_or_adapter: Any,
        sample_input: Optional[Any] = None,
        test_data: Optional[Any] = None,
        constraints: Optional[OptimizationConstraints] = None,
        strategy: str = "grid",
        budget: int = 15,
        progress_callback: Optional[Callable[[int, int, EvaluationResult], None]] = None
    ) -> OptimizationRunReport:
        run_id = f"opt_run_{int(time.time())}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if constraints is None:
            constraints = OptimizationConstraints()

        # 1. Hardware Profiling
        hardware_profile = HardwareProfiler.profile()

        # 2. Model Adapter & Analysis
        adapter = model_or_adapter if isinstance(model_or_adapter, ModelAdapter) else get_adapter(
            model_or_adapter, sample_input=sample_input, test_data=test_data
        )
        model_profile = ModelAnalyzer.analyze(adapter)

        # 3. Baseline Profiling
        base_device = "cuda" if hardware_profile.supported_backends.get("cuda") and "cuda" in model_profile.supported_devices else "cpu"
        baseline_result = BaselineProfiler.profile(
            adapter=adapter,
            device=base_device,
            precision="fp32",
            batch_size=1,
            warmup_runs=5,
            measured_runs=20
        )

        # 4. Bottleneck Analysis
        bottleneck_report = BottleneckAnalyzer.analyze(baseline_result, model_profile)

        # 5. Candidate Generation
        candidates = SearchSpace.generate_candidates(
            hardware=hardware_profile,
            model=model_profile,
            bottleneck=bottleneck_report
        )

        # 6. Execute Search Strategy
        strat = strategy.lower().strip()
        baseline_metrics = baseline_result

        if strat == "bayesian":
            evaluations = BayesianOptimizer.search(
                adapter=adapter,
                candidates=candidates,
                constraints=constraints,
                budget=budget,
                baseline_metrics=baseline_metrics,
                progress_callback=progress_callback
            )
        else:
            evaluations = GridSearchOptimizer.search(
                adapter=adapter,
                candidates=candidates,
                constraints=constraints,
                baseline_metrics=baseline_metrics,
                progress_callback=progress_callback
            )

        # 7. Select Best Configuration
        accepted_results = [r for r in evaluations if r.status == "ACCEPTED" and r.metrics is not None]
        best_eval = max(accepted_results, key=lambda r: r.score) if accepted_results else None

        best_config = best_eval.candidate.to_dict() if best_eval else None
        best_metrics = best_eval.metrics.to_dict() if best_eval else None

        # 8. Compute Improvement
        improvement = {}
        if best_eval and best_eval.metrics:
            b_lat = baseline_result.total_latency_ms
            o_lat = best_eval.metrics.mean_latency_ms
            lat_red = ((b_lat - o_lat) / max(1e-6, b_lat)) * 100.0

            b_th = baseline_result.throughput_samples_per_sec
            o_th = best_eval.metrics.throughput_samples_per_sec
            th_gain = ((o_th - b_th) / max(1e-6, b_th)) * 100.0

            acc_delta = (best_eval.metrics.accuracy - baseline_result.accuracy) * 100.0

            improvement = {
                "latency_reduction_pct": round(lat_red, 2),
                "throughput_gain_pct": round(th_gain, 2),
                "accuracy_delta_pct": round(acc_delta, 2)
            }

        report = OptimizationRunReport(
            run_id=run_id,
            timestamp=timestamp,
            hardware=hardware_profile.to_dict(),
            model=model_profile.to_dict(),
            baseline=baseline_result.to_dict(),
            bottleneck=bottleneck_report.to_dict(),
            constraints=constraints.to_dict(),
            strategy=strat,
            total_candidates_explored=len(evaluations),
            evaluations=[e.to_dict() for e in evaluations],
            best_configuration=best_config,
            best_metrics=best_metrics,
            improvement=improvement
        )

        # Save run artifact
        save_run_artifact(report.to_dict())

        return report
