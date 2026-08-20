"""Tests for Dynamic Request Batching."""

import asyncio
import numpy as np
import pytest
from autoopt.models import get_adapter
from autoopt.deployment import DynamicBatcher
import examples.sklearn_random_forest as ex_sklearn


def test_dynamic_batcher_coalescing():
    async def _run():
        model = ex_sklearn.get_model(n_estimators=10)
        sample = ex_sklearn.get_sample_input(1)
        adapter = get_adapter(model, sample_input=sample)
        prepared_model = adapter.prepare_for_inference(device="cpu", precision="fp32")

        batcher = DynamicBatcher(
            adapter=adapter,
            prepared_model=prepared_model,
            max_batch_size=4,
            max_batch_wait_ms=50.0
        )

        await batcher.start()

        # Submit 4 concurrent items
        raw_items = [np.random.randn(20).astype(np.float32) for _ in range(4)]
        tasks = [batcher.submit(item) for item in raw_items]
        results = await asyncio.gather(*tasks)

        assert len(results) == 4
        for r in results:
            assert isinstance(r, (int, np.integer))

        telemetry = batcher.get_telemetry()
        assert telemetry["total_requests"] == 4
        assert telemetry["total_batches"] >= 1

        await batcher.stop()

    asyncio.run(_run())
