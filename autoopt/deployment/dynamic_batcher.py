"""Dynamic Request Batching Queue for High-Throughput Online Inference."""

import time
import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from autoopt.models.base import ModelAdapter


@dataclass
class BatchRequestItem:
    raw_input: Any
    future: asyncio.Future
    enqueue_time: float


class DynamicBatcher:
    """Coalesces concurrent incoming inference requests into optimal batches."""

    def __init__(
        self,
        adapter: ModelAdapter,
        prepared_model: Any,
        max_batch_size: int = 16,
        max_batch_wait_ms: float = 10.0
    ):
        self.adapter = adapter
        self.prepared_model = prepared_model
        self.max_batch_size = max(1, max_batch_size)
        self.max_batch_wait_seconds = max_batch_wait_ms / 1000.0

        self._queue: asyncio.Queue[BatchRequestItem] = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Telemetry metrics
        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_sizes = 0
        self.total_queue_wait_ms = 0.0
        self.total_infer_latency_ms = 0.0

    async def start(self):
        """Start the background batch coalescing worker."""
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self):
        """Stop the dynamic batcher worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def submit(self, raw_input: Any) -> Any:
        """Submit an individual input to the batcher and await prediction."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        item = BatchRequestItem(
            raw_input=raw_input,
            future=fut,
            enqueue_time=time.perf_counter()
        )
        self.total_requests += 1
        await self._queue.put(item)
        return await fut

    async def _batch_worker(self):
        """Asynchronous worker that forms and executes batches based on size and timeout."""
        while self._running:
            try:
                # Wait for the first item
                first_item = await self._queue.get()
                items: List[BatchRequestItem] = [first_item]
                deadline = time.perf_counter() + self.max_batch_wait_seconds

                # Greedily pull more items until max_batch_size or deadline
                while len(items) < self.max_batch_size:
                    time_left = deadline - time.perf_counter()
                    if time_left <= 0:
                        break
                    try:
                        next_item = await asyncio.wait_for(self._queue.get(), timeout=time_left)
                        items.append(next_item)
                    except asyncio.TimeoutError:
                        break

                now = time.perf_counter()
                batch_size = len(items)
                self.total_batches += 1
                self.total_batch_sizes += batch_size

                # Compute queue wait times
                for item in items:
                    wait_ms = (now - item.enqueue_time) * 1000.0
                    self.total_queue_wait_ms += wait_ms

                # Form batched input
                raw_inputs = [item.raw_input for item in items]
                try:
                    # Ingest and infer
                    t0_infer = time.perf_counter()
                    
                    # Stack raw inputs
                    if isinstance(raw_inputs[0], np.ndarray):
                        batched_raw = np.stack(raw_inputs, axis=0)
                    else:
                        batched_raw = np.asarray(raw_inputs)

                    prep = self.adapter.preprocess(batched_raw)
                    out = self.adapter.run_inference(self.prepared_model, prep)
                    predictions = self.adapter.postprocess(out)
                    t1_infer = time.perf_counter()

                    infer_ms = (t1_infer - t0_infer) * 1000.0
                    self.total_infer_latency_ms += infer_ms

                    # Resolve individual futures
                    for item, pred in zip(items, predictions):
                        if not item.future.done():
                            item.future.set_result(pred)

                except Exception as e:
                    for item in items:
                        if not item.future.done():
                            item.future.set_exception(e)

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.001)

    def get_telemetry(self) -> dict:
        """Return operational metrics."""
        avg_batch = (self.total_batch_sizes / max(1, self.total_batches))
        avg_wait = (self.total_queue_wait_ms / max(1, self.total_requests))
        avg_infer = (self.total_infer_latency_ms / max(1, self.total_batches))
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "average_batch_size": round(avg_batch, 2),
            "average_queue_wait_ms": round(avg_wait, 2),
            "average_batch_infer_ms": round(avg_infer, 2),
            "current_queue_depth": self._queue.qsize()
        }
