"""Execution backends for AutoOptimizeML."""

from autoopt.backends.base import ExecutionBackend
from autoopt.backends.cpu_backend import CPUBackend
from autoopt.backends.cuda_backend import CUDABackend
from autoopt.backends.rocm_backend import ROCmBackend

def get_backend(name: str, **kwargs) -> ExecutionBackend:
    """Factory method to instantiate an execution backend by name."""
    name_clean = name.lower().strip()
    if name_clean == "cpu":
        return CPUBackend(**kwargs)
    elif name_clean == "cuda":
        return CUDABackend(**kwargs)
    elif name_clean == "rocm":
        return ROCmBackend(**kwargs)
    else:
        raise ValueError(f"Unsupported backend '{name}'. Supported backends: cpu, cuda, rocm")

__all__ = ["ExecutionBackend", "CPUBackend", "CUDABackend", "ROCmBackend", "get_backend"]
