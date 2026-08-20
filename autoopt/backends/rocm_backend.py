"""AMD ROCm / HIP Execution Backend Interface.

This backend demonstrates how AMD ROCm/HIP execution targets can be plugged into
AutoOptimizeML without modifying the optimization engine, benchmark harnesses, or schedulers.
"""

from typing import Dict, Any
from autoopt.backends.base import ExecutionBackend


class ROCmBackend(ExecutionBackend):
    """Execution backend interface for AMD Instinct and Radeon GPUs via ROCm/HIP."""

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._available = False
        self._device_name = "AMD ROCm GPU"
        self._peak_memory_mb = 0.0

        # Check for PyTorch built with ROCm support (torch.version.hip)
        try:
            import torch
            if hasattr(torch.version, 'hip') and torch.version.hip is not None and torch.cuda.is_available():
                self._available = True
                self._device_name = torch.cuda.get_device_name(device_index)
        except Exception:
            self._available = False

    @property
    def name(self) -> str:
        return "rocm"

    def is_available(self) -> bool:
        return self._available

    def device_name(self) -> str:
        return self._device_name

    def synchronize(self) -> None:
        """Synchronize HIP compute streams."""
        if self._available:
            try:
                import torch
                torch.cuda.synchronize(self._device_index)
            except Exception:
                pass

    def set_num_threads(self, n: int) -> None:
        pass

    def get_memory_allocated_mb(self) -> float:
        if not self._available:
            return 0.0
        try:
            import torch
            return round(torch.cuda.memory_allocated(self._device_index) / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def get_peak_memory_mb(self) -> float:
        if not self._available:
            return 0.0
        try:
            import torch
            return round(torch.cuda.max_memory_allocated(self._device_index) / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def reset_peak_memory(self) -> None:
        if self._available:
            try:
                import torch
                torch.cuda.reset_peak_memory_stats(self._device_index)
            except Exception:
                pass

    def get_device_properties(self) -> Dict[str, Any]:
        return {
            "backend": "rocm",
            "available": self._available,
            "device": self._device_name,
            "status": "Operational" if self._available else "ROCm/HIP runtime not active in environment"
        }
