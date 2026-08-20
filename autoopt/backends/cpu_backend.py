"""Multithreaded CPU Execution Backend."""

import os
import psutil
from typing import Dict, Any
from autoopt.backends.base import ExecutionBackend
from autoopt.hardware.cpu_info import get_cpu_info


class CPUBackend(ExecutionBackend):
    """Execution backend for multicore CPU compute."""

    def __init__(self, num_threads: int = None):
        self._cpu_info = get_cpu_info()
        self._process = psutil.Process(os.getpid())
        self._peak_memory_mb = 0.0
        self._current_threads = num_threads or self._cpu_info.get("physical_cores", 1)
        self.set_num_threads(self._current_threads)

    @property
    def name(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        return True

    def device_name(self) -> str:
        return self._cpu_info.get("model_name", "Host CPU")

    def synchronize(self) -> None:
        """On CPU, execution is synchronous by default."""
        pass

    def set_num_threads(self, n: int) -> None:
        self._current_threads = max(1, n)
        os.environ["OMP_NUM_THREADS"] = str(self._current_threads)
        os.environ["MKL_NUM_THREADS"] = str(self._current_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self._current_threads)

        try:
            import torch
            torch.set_num_threads(self._current_threads)
        except (ImportError, RuntimeError):
            pass

        try:
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(self._current_threads)
        except (ImportError, RuntimeError):
            pass

    def get_memory_allocated_mb(self) -> float:
        mem = self._process.memory_info().rss / (1024 * 1024)
        if mem > self._peak_memory_mb:
            self._peak_memory_mb = mem
        return round(mem, 2)

    def get_peak_memory_mb(self) -> float:
        mem = self._process.memory_info().rss / (1024 * 1024)
        if mem > self._peak_memory_mb:
            self._peak_memory_mb = mem
        return round(self._peak_memory_mb, 2)

    def reset_peak_memory(self) -> None:
        self._peak_memory_mb = self._process.memory_info().rss / (1024 * 1024)

    def get_device_properties(self) -> Dict[str, Any]:
        return {
            "backend": "cpu",
            "device": self.device_name(),
            "physical_cores": self._cpu_info.get("physical_cores"),
            "logical_cores": self._cpu_info.get("logical_cores"),
            "active_threads": self._current_threads,
            "total_ram_gb": self._cpu_info.get("total_ram_gb"),
            "simd_extensions": self._cpu_info.get("simd_extensions", [])
        }
