"""Tests for hardware profiler and execution backends."""

import pytest
from autoopt.hardware import HardwareProfiler, get_cpu_info, get_gpu_info
from autoopt.backends import get_backend, CPUBackend, CUDABackend, ROCmBackend


def test_cpu_info_introspection():
    cpu = get_cpu_info()
    assert isinstance(cpu, dict)
    assert cpu["physical_cores"] >= 1
    assert cpu["logical_cores"] >= 1
    assert cpu["total_ram_gb"] > 0
    assert isinstance(cpu["simd_extensions"], list)
    assert "architecture" in cpu


def test_gpu_info_introspection():
    gpus = get_gpu_info()
    assert isinstance(gpus, list)
    for g in gpus:
        assert "vendor" in g
        assert "device_name" in g
        assert "driver_status" in g


def test_hardware_profiler_profile():
    profile = HardwareProfiler.profile()
    assert profile.cpu["physical_cores"] >= 1
    assert "cpu" in profile.supported_backends
    assert profile.supported_backends["cpu"] is True
    
    cli_str = profile.format_cli()
    assert "AutoOptimizeML Profile" in cli_str
    assert "CPU" in cli_str
    
    d = profile.to_dict()
    assert isinstance(d, dict)


def test_execution_backends():
    # CPU Backend
    cpu_b = get_backend("cpu")
    assert isinstance(cpu_b, CPUBackend)
    assert cpu_b.is_available() is True
    assert cpu_b.get_memory_allocated_mb() >= 0
    cpu_b.set_num_threads(2)
    cpu_b.synchronize()

    # CUDA Backend (gracefully checks availability)
    cuda_b = get_backend("cuda")
    assert isinstance(cuda_b, CUDABackend)
    assert isinstance(cuda_b.is_available(), bool)

    # ROCm Backend (interface ready)
    rocm_b = get_backend("rocm")
    assert isinstance(rocm_b, ROCmBackend)
    assert isinstance(rocm_b.is_available(), bool)
