"""Empirical Experiment Harness: Runs comprehensive benchmark matrix and constraint optimization."""

import time
import json
import numpy as np
from autoopt.models import get_adapter
from autoopt.analyzer import ModelAnalyzer
from autoopt.hardware import HardwareProfiler
from autoopt.profiling import BaselineProfiler, BottleneckAnalyzer
from autoopt.optimizer import OptimizationEngine, OptimizationConstraints, SearchSpace
from autoopt.benchmark import BenchmarkRunner
from autoopt.optimizer.constraints import ConstraintEvaluator

import examples.pytorch_cnn as ex_vision
import examples.sklearn_random_forest as ex_tabular
import examples.emotion_analyzer as ex_nlp


def run_full_experimental_matrix():
    print("=" * 80)
    print("      AutoOptimizeML Empirical Experiment Matrix & Optimization Study")
    print("=" * 80)

    hardware = HardwareProfiler.profile()
    print(f"Detected Hardware: {hardware.cpu['model_name']} ({hardware.cpu['physical_cores']}C/{hardware.cpu['logical_cores']}T, {hardware.cpu['total_ram_gb']} GB RAM)")
    print(f"Active Backends:   CPU: {hardware.supported_backends.get('cpu')}, CUDA: {hardware.supported_backends.get('cuda')}")
    print("=" * 80)

    # ---------------------------------------------------------
    # WORKLOAD 1: PyTorch Vision CNN (CIFAR-style 3x32x32)
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print(" WORKLOAD 1: PyTorch Vision CNN (Deep Learning Computer Vision)")
    print("#" * 80)

    v_model = ex_vision.get_model()
    v_sample = ex_vision.get_sample_input(1)
    v_test = ex_vision.get_test_data(100)
    v_adapter = get_adapter(v_model, sample_input=v_sample, test_data=v_test)
    v_profile = ModelAnalyzer.analyze(v_adapter)

    # Baseline Profiling
    v_baseline = BaselineProfiler.profile(
        adapter=v_adapter,
        device="cpu",
        precision="fp32",
        batch_size=1,
        warmup_runs=10,
        measured_runs=50
    )
    v_bottleneck = BottleneckAnalyzer.analyze(v_baseline, v_profile)

    print("\n--- BASELINE METRICS ---")
    print(f"Latency (Mean):      {v_baseline.total_latency_ms:.2f} ms (P95: {v_baseline.p95_latency_ms:.2f} ms, P99: {v_baseline.p99_latency_ms:.2f} ms)")
    print(f"Throughput:          {v_baseline.throughput_samples_per_sec:.1f} samples/sec")
    print(f"Memory RSS:          {v_baseline.memory_mb:.1f} MB")
    print(f"Accuracy:            {v_baseline.accuracy * 100:.1f}%")
    print(f"Primary Bottleneck:  {v_bottleneck.primary_bottleneck} ({v_bottleneck.stage_percentages['inference']:.1f}% inference compute)")

    # Define Optimization Problem
    v_constraints = OptimizationConstraints(
        min_accuracy=max(0.0, v_baseline.accuracy - 0.05),
        max_latency_ms=25.0,  # Strict SLA: <= 25ms
        max_memory_mb=2048.0,
        objective="maximize_throughput"
    )

    print(f"\n--- OPTIMIZATION CONSTRAINTS ---")
    print(f"Objective:   Maximize Throughput")
    print(f"Subject to:  Accuracy >= {v_constraints.min_accuracy * 100:.1f}%, Latency <= {v_constraints.max_latency_ms:.1f} ms, Memory <= {v_constraints.max_memory_mb:.0f} MB")

    # Generate Candidate Search Space
    v_candidates = SearchSpace.generate_candidates(
        hardware=hardware,
        model=v_profile,
        bottleneck=v_bottleneck,
        custom_batch_sizes=[1, 2, 4, 8, 16, 32],
        custom_workers=[1, 2, 4, 8]
    )
    print(f"Generated {len(v_candidates)} candidate execution configurations.")

    print("\n--- CANDIDATE BENCHMARK MATRIX ---")
    print(f"{'ID':<6} {'Batch':<6} {'Workers':<8} {'JIT':<6} {'Native':<8} {'Latency':<12} {'P95 (ms)':<10} {'Throughput':<16} {'Accuracy':<10} {'Status':<10}")
    print("-" * 92)

    v_results = []
    for c in v_candidates:
        metrics = BenchmarkRunner.benchmark(
            adapter=v_adapter,
            device=c.device,
            precision=c.precision,
            batch_size=c.batch_size,
            workers=c.workers,
            compile_graph=c.compile_graph,
            native_preprocessing=c.native_preprocessing,
            warmup_runs=5,
            measured_runs=25
        )
        eval_res = ConstraintEvaluator.evaluate(c, metrics, v_constraints, v_baseline)
        v_results.append(eval_res)

        jit_str = "Yes" if c.compile_graph else "No"
        nat_str = "Yes" if c.native_preprocessing else "No"
        lat_str = f"{metrics.mean_latency_ms:.2f} ms"
        p95_str = f"{metrics.p95_latency_ms:.2f}"
        tp_str = f"{metrics.throughput_samples_per_sec:.1f} smp/s"
        acc_str = f"{metrics.accuracy * 100:.1f}%"

        print(f"{c.candidate_id:<6} {c.batch_size:<6} {c.workers:<8} {jit_str:<6} {nat_str:<8} {lat_str:<12} {p95_str:<10} {tp_str:<16} {acc_str:<10} {eval_res.status:<10}")

    # Select Best Configuration
    accepted = [r for r in v_results if r.status == "ACCEPTED"]
    best_vision = max(accepted, key=lambda r: r.score) if accepted else None

    if best_vision:
        bv_cand = best_vision.candidate
        bv_met = best_vision.metrics
        th_gain = ((bv_met.throughput_samples_per_sec - v_baseline.throughput_samples_per_sec) / v_baseline.throughput_samples_per_sec) * 100.0
        lat_red = ((v_baseline.total_latency_ms - bv_met.mean_latency_ms) / v_baseline.total_latency_ms) * 100.0
        print("\n" + "=" * 80)
        print(" 🏆 WINNING CONFIGURATION (PyTorch CNN)")
        print("=" * 80)
        print(f"Batch Size:           {bv_cand.batch_size}")
        print(f"Worker Threads:       {bv_cand.workers}")
        print(f"TorchScript JIT:      {bv_cand.compile_graph}")
        print(f"Native Acceleration:  {bv_cand.native_preprocessing}")
        print(f"Baseline Throughput:  {v_baseline.throughput_samples_per_sec:.1f} samples/sec (Latency: {v_baseline.total_latency_ms:.2f} ms)")
        print(f"Optimized Throughput: {bv_met.throughput_samples_per_sec:.1f} samples/sec (Latency: {bv_met.mean_latency_ms:.2f} ms, P95: {bv_met.p95_latency_ms:.2f} ms)")
        print(f"Throughput Speedup:   {bv_met.throughput_samples_per_sec / v_baseline.throughput_samples_per_sec:.2f}x (+{th_gain:.1f}%)")
        print(f"Latency SLA:          {bv_met.mean_latency_ms:.2f} ms <= {v_constraints.max_latency_ms} ms (SLA Satisfied ✓)")
        print("=" * 80)

    # ---------------------------------------------------------
    # WORKLOAD 2: Scikit-Learn Random Forest (Tabular ML)
    # ---------------------------------------------------------
    print("\n" + "#" * 80)
    print(" WORKLOAD 2: Scikit-Learn Random Forest (Tabular Machine Learning)")
    print("#" * 80)

    t_model = ex_tabular.get_model(n_estimators=50)
    t_sample = ex_tabular.get_sample_input(1)
    t_test = ex_tabular.get_test_data()
    t_adapter = get_adapter(t_model, sample_input=t_sample, test_data=t_test)
    t_profile = ModelAnalyzer.analyze(t_adapter)

    t_baseline = BaselineProfiler.profile(
        adapter=t_adapter,
        device="cpu",
        precision="fp32",
        batch_size=1,
        warmup_runs=10,
        measured_runs=50
    )
    print("\n--- BASELINE METRICS ---")
    print(f"Latency (Mean):      {t_baseline.total_latency_ms:.2f} ms")
    print(f"Throughput:          {t_baseline.throughput_samples_per_sec:.1f} samples/sec")
    print(f"Accuracy:            {t_baseline.accuracy * 100:.1f}%")

    t_constraints = OptimizationConstraints(
        min_accuracy=t_baseline.accuracy - 0.01,
        max_latency_ms=15.0,
        objective="maximize_throughput"
    )

    t_candidates = SearchSpace.generate_candidates(
        hardware=hardware,
        model=t_profile,
        custom_batch_sizes=[1, 4, 16, 64, 256],
        custom_workers=[1, 2, 4, 8]
    )

    print("\n--- CANDIDATE BENCHMARK MATRIX ---")
    print(f"{'ID':<6} {'Batch':<8} {'Workers':<8} {'Latency':<14} {'Throughput':<18} {'Accuracy':<10} {'Status':<10}")
    print("-" * 74)

    t_results = []
    for c in t_candidates:
        metrics = BenchmarkRunner.benchmark(
            adapter=t_adapter,
            device=c.device,
            precision=c.precision,
            batch_size=c.batch_size,
            workers=c.workers,
            warmup_runs=5,
            measured_runs=25
        )
        eval_res = ConstraintEvaluator.evaluate(c, metrics, t_constraints, t_baseline)
        t_results.append(eval_res)

        lat_str = f"{metrics.mean_latency_ms:.2f} ms"
        tp_str = f"{metrics.throughput_samples_per_sec:.1f} smp/s"
        acc_str = f"{metrics.accuracy * 100:.1f}%"
        print(f"{c.candidate_id:<6} {c.batch_size:<8} {c.workers:<8} {lat_str:<14} {tp_str:<18} {acc_str:<10} {eval_res.status:<10}")

    best_tab = max([r for r in t_results if r.status == "ACCEPTED"], key=lambda r: r.score, default=None)
    if best_tab:
        bt_cand = best_tab.candidate
        bt_met = best_tab.metrics
        th_gain_t = ((bt_met.throughput_samples_per_sec - t_baseline.throughput_samples_per_sec) / t_baseline.throughput_samples_per_sec) * 100.0
        print("\n" + "=" * 80)
        print(" 🏆 WINNING CONFIGURATION (Scikit-Learn Random Forest)")
        print("=" * 80)
        print(f"Batch Size:           {bt_cand.batch_size}")
        print(f"Worker Threads:       {bt_cand.workers}")
        print(f"Baseline Throughput:  {t_baseline.throughput_samples_per_sec:.1f} samples/sec (Latency: {t_baseline.total_latency_ms:.2f} ms)")
        print(f"Optimized Throughput: {bt_met.throughput_samples_per_sec:.1f} samples/sec (Latency: {bt_met.mean_latency_ms:.2f} ms)")
        print(f"Throughput Speedup:   {bt_met.throughput_samples_per_sec / t_baseline.throughput_samples_per_sec:.2f}x (+{th_gain_t:.1f}%)")
        print("=" * 80)


if __name__ == "__main__":
    run_full_experimental_matrix()
