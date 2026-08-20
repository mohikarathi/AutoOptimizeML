"""AutoOptimizeML Command Line Interface."""

import os
import sys
import importlib.util
from typing import Optional, List, Dict, Any
import click
import uvicorn

from autoopt.hardware import HardwareProfiler
from autoopt.models import get_adapter
from autoopt.analyzer import ModelAnalyzer
from autoopt.benchmark import BenchmarkRunner
from autoopt.optimizer import OptimizationEngine, OptimizationConstraints, OptimizationRunReport
from autoopt.utils.storage import load_run_artifact, list_runs
from autoopt.deployment import create_inference_app
from autoopt.dashboard import create_dashboard_app


def _load_model_from_target(target: str):
    """Load model from built-in example keywords or python file path."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    target_clean = target.lower().strip()
    if target_clean in ("sklearn", "sklearn_random_forest", "random_forest"):
        import examples.sklearn_random_forest as ex
        return ex.get_model(), ex.get_sample_input(1), ex.get_test_data()
    elif target_clean in ("pytorch", "pytorch_cnn", "cnn"):
        import examples.pytorch_cnn as ex
        return ex.get_model(), ex.get_sample_input(1), ex.get_test_data()
    elif target_clean in ("tensorflow", "tf", "tf_cnn"):
        import examples.tf_cnn as ex
        return ex.get_model(), ex.get_sample_input(1), ex.get_test_data()
    elif target_clean in ("emotion", "emotion_analyzer", "nlp"):
        import examples.emotion_analyzer as ex
        return ex.get_model(), ex.get_sample_input(1), ex.get_test_data()

    # Load from external Python script
    if os.path.exists(target):
        spec = importlib.util.spec_from_file_location("custom_model_module", target)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["custom_model_module"] = mod
        spec.loader.exec_module(mod)

        if hasattr(mod, "get_model"):
            model = mod.get_model()
            sample = mod.get_sample_input() if hasattr(mod, "get_sample_input") else None
            test_data = mod.get_test_data() if hasattr(mod, "get_test_data") else None
            return model, sample, test_data
        elif hasattr(mod, "model"):
            return mod.model, getattr(mod, "sample_input", None), getattr(mod, "test_data", None)

    raise click.BadParameter(f"Could not load model from '{target}'. Provide a path to a python file with get_model() or use an example name (pytorch_cnn, sklearn, tf_cnn, emotion_analyzer).")


@click.group()
@click.version_option(version="1.0.0", message="AutoOptimizeML %(version)s")
def cli():
    """AutoOptimizeML — Hardware-Agnostic ML/DL Optimization & Deployment Platform."""
    pass


@cli.command("profile")
@click.option("--json-out", is_flag=True, help="Output hardware profile in JSON format")
def profile(json_out: bool):
    """Detect and profile host CPU, GPU, memory, and execution runtime capabilities."""
    p = HardwareProfiler.profile()
    if json_out:
        click.echo(p.to_json())
    else:
        click.echo(p.format_cli())


@cli.command("analyze")
@click.option("--model", "-m", required=True, help="Model path or built-in example (pytorch_cnn, sklearn, tf_cnn, emotion_analyzer)")
@click.option("--json-out", is_flag=True, help="Output model profile in JSON format")
def analyze(model: str, json_out: bool):
    """Inspect model graph, parameters, input/output tensors, and estimated compute complexity."""
    m, sample, test_data = _load_model_from_target(model)
    adapter = get_adapter(m, sample_input=sample, test_data=test_data)
    profile = ModelAnalyzer.analyze(adapter)
    if json_out:
        click.echo(profile.to_json())
    else:
        click.echo(profile.format_cli())


@cli.command("benchmark")
@click.option("--model", "-m", required=True, help="Model path or example name")
@click.option("--device", "-d", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Execution target device")
@click.option("--precision", "-p", default="fp32", type=click.Choice(["fp32", "fp16", "int8"]), help="Inference precision")
@click.option("--batch-size", "-b", default=1, type=int, help="Inference batch size")
@click.option("--workers", "-w", default=1, type=int, help="Thread concurrency worker count")
@click.option("--jit/--no-jit", default=False, help="Enable graph compilation (TorchScript/XLA)")
def benchmark(model: str, device: str, precision: str, batch_size: int, workers: int, jit: bool):
    """Run repeatable latency and throughput benchmarking on a specific configuration."""
    m, sample, test_data = _load_model_from_target(model)
    adapter = get_adapter(m, sample_input=sample, test_data=test_data)

    click.echo(f"Benchmarking {adapter.framework.upper()} model on {device.upper()} (Precision={precision.upper()}, Batch={batch_size}, Workers={workers}, JIT={jit})...")
    metrics = BenchmarkRunner.benchmark(
        adapter=adapter,
        device=device,
        precision=precision,
        batch_size=batch_size,
        workers=workers,
        compile_graph=jit,
        warmup_runs=5,
        measured_runs=30
    )

    click.echo("\n" + "=" * 50)
    click.echo("             Benchmark Results")
    click.echo("=" * 50)
    click.echo(f"Mean Latency:       {metrics.mean_latency_ms:.2f} ms")
    click.echo(f"P50 Latency:        {metrics.p50_latency_ms:.2f} ms")
    click.echo(f"P95 Latency:        {metrics.p95_latency_ms:.2f} ms")
    click.echo(f"P99 Latency:        {metrics.p99_latency_ms:.2f} ms")
    click.echo(f"Throughput:         {metrics.throughput_samples_per_sec:.2f} samples/s ({metrics.throughput_req_per_sec:.2f} req/s)")
    click.echo(f"Accuracy:           {metrics.accuracy * 100:.2f}%")
    click.echo(f"Memory Footprint:   {metrics.memory_allocated_mb:.2f} MB (Peak: {metrics.peak_memory_mb:.2f} MB)")
    click.echo("=" * 50)


@cli.command("optimize")
@click.option("--model", "-m", required=True, help="Model path or example name")
@click.option("--objective", "-o", default="maximize_throughput", type=click.Choice(["maximize_throughput", "minimize_latency", "minimize_memory", "balanced"]), help="Optimization goal")
@click.option("--strategy", "-s", default="bayesian", type=click.Choice(["bayesian", "grid"]), help="Search exploration algorithm")
@click.option("--min-accuracy", default=None, type=float, help="Minimum accuracy constraint (e.g. 0.90)")
@click.option("--max-latency", default=None, type=float, help="Maximum allowed latency in ms (e.g. 25.0)")
@click.option("--max-memory", default=None, type=float, help="Maximum memory budget in MB")
@click.option("--budget", default=15, type=int, help="Maximum candidate evaluation budget")
def optimize(model: str, objective: str, strategy: str, min_accuracy: Optional[float], max_latency: Optional[float], max_memory: Optional[float], budget: int):
    """Automatically explore configurations, benchmark bottlenecks, enforce constraints, and select the best setup."""
    m, sample, test_data = _load_model_from_target(model)
    adapter = get_adapter(m, sample_input=sample, test_data=test_data)

    constraints = OptimizationConstraints(
        min_accuracy=min_accuracy,
        max_latency_ms=max_latency,
        max_memory_mb=max_memory,
        objective=objective
    )

    click.echo("=" * 60)
    click.echo(" 🚀 Initializing AutoOptimizeML Automated Search")
    click.echo("=" * 60)
    click.echo(f"Target Workload:    {adapter.framework.upper()} ({adapter.model.__class__.__name__})")
    click.echo(f"Strategy:           {strategy.upper()}")
    click.echo(f"Objective:          {objective}")
    if min_accuracy: click.echo(f"Constraint:         Accuracy >= {min_accuracy*100:.1f}%")
    if max_latency: click.echo(f"Constraint:         Latency <= {max_latency:.1f} ms")
    if max_memory: click.echo(f"Constraint:         Memory <= {max_memory:.0f} MB")
    click.echo("-" * 60)

    def on_progress(current: int, total: int, eval_res):
        c = eval_res.candidate
        status_symbol = "✓" if eval_res.status == "ACCEPTED" else ("✗" if eval_res.status == "REJECTED" else "!")
        lat = f"{eval_res.metrics.mean_latency_ms:.2f}ms" if eval_res.metrics else "-"
        tp = f"{eval_res.metrics.throughput_samples_per_sec:.1f}smp/s" if eval_res.metrics else "-"
        click.echo(f"[{current:02d}/{total:02d}] {status_symbol} {c.summary()} -> Latency: {lat}, Throughput: {tp} ({eval_res.status})")

    report = OptimizationEngine.run(
        model_or_adapter=adapter,
        sample_input=sample,
        test_data=test_data,
        constraints=constraints,
        strategy=strategy,
        budget=budget,
        progress_callback=on_progress
    )

    click.echo("\n" + report.format_cli())


@cli.command("report")
@click.option("--run", "-r", default="latest", help="Run ID to display (e.g. 'latest' or 'opt_run_123')")
def report(run: str):
    """Display detailed results and constraint verification for an optimization run."""
    data = load_run_artifact(run)
    if data is None:
        click.echo(f"Error: Run artifact '{run}' not found.", err=True)
        sys.exit(1)

    rep = OptimizationRunReport(**data)
    click.echo(rep.format_cli())


@cli.command("deploy")
@click.option("--config", "-c", "config_target", default="latest", help="Run ID or JSON config path")
@click.option("--model", "-m", default="pytorch_cnn", help="Model path or example name to serve")
@click.option("--port", "-p", default=8000, type=int, help="REST API server port")
@click.option("--dynamic-batching/--no-dynamic-batching", default=True, help="Enable dynamic batching queue")
def deploy(config_target: str, model: str, port: int, dynamic_batching: bool):
    """Deploy the optimized model behind a high-throughput FastAPI inference server."""
    data = load_run_artifact(config_target)
    if data and "best_configuration" in data:
        cfg = data["best_configuration"]
        click.echo(f"Loaded best configuration from run artifact '{config_target}'.")
    elif os.path.exists(config_target):
        import json
        with open(config_target, "r") as f:
            cfg = json.load(f)
    else:
        click.echo(f"Warning: No valid run artifact found for '{config_target}'. Using default configuration.")
        cfg = {"device": "cpu", "precision": "fp32", "batch_size": 4, "workers": 2, "compile_graph": False}

    m, sample, test_data = _load_model_from_target(model)
    adapter = get_adapter(m, sample_input=sample, test_data=test_data)

    click.echo(f"Starting AutoOptimizeML Inference Server on http://0.0.0.0:{port}...")
    click.echo(f"Active Backend: {cfg.get('device', 'cpu').upper()} | Precision: {cfg.get('precision', 'fp32').upper()} | Dynamic Batching: {dynamic_batching}")

    app = create_inference_app(adapter, cfg, enable_dynamic_batching=dynamic_batching)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


@cli.command("dashboard")
@click.option("--port", "-p", default=8501, type=int, help="Dashboard port")
def dashboard(port: int):
    """Launch the interactive web dashboard."""
    click.echo(f"Starting AutoOptimizeML Interactive Dashboard on http://localhost:{port}...")
    app = create_dashboard_app()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main():
    cli()


if __name__ == "__main__":
    main()
