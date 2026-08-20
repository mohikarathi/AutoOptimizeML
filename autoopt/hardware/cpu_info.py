"""CPU hardware and SIMD instruction set introspection."""

import os
import platform
import psutil
from typing import Dict, Any, List


def get_simd_flags() -> List[str]:
    """Parse supported SIMD and vector extensions from /proc/cpuinfo or platform."""
    flags: List[str] = []
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
            for line in content.splitlines():
                if line.startswith("flags") or line.startswith("Features"):
                    raw_flags = line.split(":", 1)[1].strip().split()
                    target_flags = [
                        "avx", "avx2", "avx512f", "avx512_vnni", "avx512_bf16",
                        "fma", "sse4_1", "sse4_2", "neon", "vfpv4", "sve"
                    ]
                    for tf in target_flags:
                        if tf in raw_flags:
                            flags.append(tf.upper())
                    break
        except Exception:
            pass
    return sorted(list(set(flags)))


def get_cpu_info() -> Dict[str, Any]:
    """Retrieve detailed CPU topology and specifications."""
    arch = platform.machine()
    system = platform.system()
    model_name = platform.processor() or "Unknown CPU"
    vendor = "Unknown"

    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        model_name = line.split(":", 1)[1].strip()
                    elif "vendor_id" in line:
                        vendor = line.split(":", 1)[1].strip()
        except Exception:
            pass

    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or 1
    
    mem = psutil.virtual_memory()
    total_ram_gb = round(mem.total / (1024 ** 3), 2)
    available_ram_gb = round(mem.available / (1024 ** 3), 2)

    freq = psutil.cpu_freq()
    max_mhz = round(freq.max, 1) if freq and freq.max else None

    simd_extensions = get_simd_flags()

    return {
        "architecture": arch,
        "system": system,
        "vendor": vendor,
        "model_name": model_name,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": available_ram_gb,
        "max_freq_mhz": max_mhz,
        "simd_extensions": simd_extensions
    }
