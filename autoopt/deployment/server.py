"""FastAPI Inference Service with Dynamic Batching and Telemetry."""

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from autoopt.models.base import ModelAdapter
from autoopt.backends import get_backend
from autoopt.deployment.dynamic_batcher import DynamicBatcher


class PredictRequest(BaseModel):
    input: Optional[Any] = None
    inputs: Optional[List[Any]] = None


class PredictResponse(BaseModel):
    predictions: Any
    latency_ms: float
    batched: bool


def create_inference_app(
    adapter: ModelAdapter,
    config: Dict[str, Any],
    enable_dynamic_batching: bool = True
) -> FastAPI:
    """Create and configure FastAPI application with the optimized model deployment."""
    device = config.get("device", "cpu")
    precision = config.get("precision", "fp32")
    batch_size = config.get("batch_size", 1)
    workers = config.get("workers", 1)
    compile_graph = config.get("compile_graph", False)

    backend = get_backend(device)
    backend.set_num_threads(workers)

    prepared_model = adapter.prepare_for_inference(
        device=device,
        precision=precision,
        compile_graph=compile_graph,
        num_threads=workers
    )

    batcher: Optional[DynamicBatcher] = None
    if enable_dynamic_batching and batch_size > 1:
        batcher = DynamicBatcher(
            adapter=adapter,
            prepared_model=prepared_model,
            max_batch_size=batch_size,
            max_batch_wait_ms=config.get("max_batch_wait_ms", 10.0)
        )

    # Telemetry storage
    metrics_data = {
        "start_time": time.time(),
        "total_requests": 0,
        "total_latency_ms": 0.0,
        "latencies": []
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if batcher:
            await batcher.start()
        yield
        if batcher:
            await batcher.stop()

    app = FastAPI(
        title="AutoOptimizeML Deployment Service",
        description="Hardware-optimized high-throughput ML/DL inference server",
        version="1.0.0",
        lifespan=lifespan
    )

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "uptime_seconds": round(time.time() - metrics_data["start_time"], 1),
            "backend": backend.name,
            "device": backend.device_name(),
            "dynamic_batching_enabled": batcher is not None
        }

    @app.get("/config")
    async def get_active_config():
        return {
            "deployment_config": config,
            "backend_properties": backend.get_device_properties(),
            "dynamic_batching": {
                "enabled": batcher is not None,
                "max_batch_size": batch_size,
                "max_batch_wait_ms": config.get("max_batch_wait_ms", 10.0)
            }
        }

    @app.get("/metrics")
    async def metrics():
        lats = metrics_data["latencies"][-500:]  # Keep sliding window
        p50 = float(np.percentile(lats, 50)) if lats else 0.0
        p95 = float(np.percentile(lats, 95)) if lats else 0.0
        p99 = float(np.percentile(lats, 99)) if lats else 0.0
        avg_lat = (metrics_data["total_latency_ms"] / max(1, metrics_data["total_requests"]))

        batcher_telemetry = batcher.get_telemetry() if batcher else {}

        return {
            "service": {
                "total_requests_processed": metrics_data["total_requests"],
                "average_latency_ms": round(avg_lat, 2),
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "memory_allocated_mb": backend.get_memory_allocated_mb(),
                "peak_memory_mb": backend.get_peak_memory_mb()
            },
            "dynamic_batcher": batcher_telemetry
        }

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest):
        t0 = time.perf_counter()

        if req.input is None and req.inputs is None:
            raise HTTPException(status_code=400, detail="Must provide either 'input' or 'inputs'")

        # 1. Dynamic batching route for single requests
        if batcher is not None and req.input is not None:
            try:
                pred = await batcher.submit(req.input)
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000.0

                metrics_data["total_requests"] += 1
                metrics_data["total_latency_ms"] += lat_ms
                metrics_data["latencies"].append(lat_ms)

                return PredictResponse(
                    predictions=pred,
                    latency_ms=round(lat_ms, 2),
                    batched=True
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # 2. Direct synchronous / batch route
        try:
            raw = req.inputs if req.inputs is not None else [req.input]
            prep = adapter.preprocess(raw)
            if device == "cuda" and hasattr(prep, "to"):
                prep = prep.to("cuda")
            backend.synchronize()
            out = adapter.run_inference(prepared_model, prep)
            backend.synchronize()
            preds = adapter.postprocess(out)

            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0

            metrics_data["total_requests"] += len(raw) if req.inputs else 1
            metrics_data["total_latency_ms"] += lat_ms
            metrics_data["latencies"].append(lat_ms)

            final_preds = preds if req.inputs is not None else preds[0]
            return PredictResponse(
                predictions=final_preds,
                latency_ms=round(lat_ms, 2),
                batched=req.inputs is not None
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
