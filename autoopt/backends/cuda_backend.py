"""NVIDIA CUDA GPU Execution Backend."""

import os
from typing import Dict, Any
from autoopt.backends.base import ExecutionBackend


class CUDABackend(ExecutionBackend):
    """Execution backend for NVIDIA CUDA GPUs."""

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._available = False
        self._device_name = "NVIDIA CUDA GPU"
        self._peak_memory_mb = 0.0

        try:
            import torch
            if torch.cuda.is_available():
                self._available = True
                self._device_name = torch.cuda.get_device_name(device_index)
        except Exception:
            self._available = False

    @property
    def name(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        return self._available

    def device_name(self) -> str:
        return self._device_name

    def synchronize(self) -> None:
        """Synchronize CUDA compute streams."""
        if self._available:
            try:
                import torch
                torch.cuda.synchronize(self._device_index)
            except Exception:
                pass

    def set_num_threads(self, n: int) -> None:
        """CUDA kernel concurrency is determined by kernel launch grid/block dimensions."""
        pass

    def get_memory_allocated_mb(self) -> float:
        if not self._available:
            return 0.0
        try:
            import torch
            mem = torch.cuda.memory_allocated(self._device_index) / (1024 * 1024)
            if mem > self._peak_memory_mb:
                self._peak_memory_mb = mem
            return round(mem, 2)
        except Exception:
            return 0.0

    def get_peak_memory_mb(self) -> float:
        if not self._available:
            return 0.0
        try:
            import torch
            mem = torch.cuda.max_memory_allocated(self._device_index) / (1024 * 1024)
            return round(max(mem, self._peak_memory_mb), 2)
        except Exception:
            return round(self._peak_memory_mb, 2)

    def reset_peak_memory(self) -> None:
        if self._available:
            try:
                import torch
                torch.cuda.reset_peak_memory_stats(self._device_index)
                self._peak_memory_mb = 0.0
            except Exception:
                pass

    def get_device_properties(self) -> Dict[str, Any]:
        if not self._available:
            return {"backend": "cuda", "available": False, "status": "CUDA runtime unavailable"}
        try:
            import torch
            props = torch.cuda.get_device_properties(self._device_index)
            return {
                "backend": "cuda",
                "available": True,
                "device": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count
            }
        except Exception as e:
            return {"backend": "cuda", "available": False, "error": str(e)}
