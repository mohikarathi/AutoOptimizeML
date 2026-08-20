"""Unified Hardware Profiler for AutoOptimizeML."""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

from autoopt.hardware.cpu_info import get_cpu_info
from autoopt.hardware.gpu_info import get_gpu_info


@dataclass
class HardwareProfile:
    cpu: Dict[str, Any]
    gpus: List[Dict[str, Any]]
    runtimes: Dict[str, Any]
    supported_backends: Dict[str, bool]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_cli(self) -> str:
        lines = [
            "=" * 50,
            "             AutoOptimizeML Profile",
            "=" * 50,
            "",
            "CPU",
            "-" * 50,
            f"Architecture:       {self.cpu.get('architecture', 'Unknown')}",
            f"Model:              {self.cpu.get('model_name', 'Unknown')}",
            f"Vendor:             {self.cpu.get('vendor', 'Unknown')}",
            f"Physical cores:     {self.cpu.get('physical_cores', 1)}",
            f"Logical cores:      {self.cpu.get('logical_cores', 1)}",
            f"RAM:                {self.cpu.get('total_ram_gb', 0)} GB (Available: {self.cpu.get('available_ram_gb', 0)} GB)",
            f"SIMD Extensions:    {', '.join(self.cpu.get('simd_extensions', [])) or 'Standard'}",
            "",
            "GPU",
            "-" * 50,
        ]

        if not self.gpus:
            lines.append("No dedicated GPU devices detected.")
        else:
            for idx, g in enumerate(self.gpus):
                lines.append(f"Device [{idx}]:         {g.get('device_name')}")
                lines.append(f"  Vendor:           {g.get('vendor')}")
                vram = f"{g.get('vram_total_gb')} GB" if g.get('vram_total_gb') else "System Shared / Dynamic"
                lines.append(f"  VRAM:             {vram}")
                cap = g.get('compute_capability') or "N/A"
                lines.append(f"  Compute Cap:      {cap}")
                lines.append(f"  Status:           {g.get('driver_status')}")

        lines.extend([
            "",
            "Runtimes & Framework Acceleration",
            "-" * 50,
            f"PyTorch Version:    {self.runtimes.get('pytorch_version', 'Not installed')}",
            f"TensorFlow Version: {self.runtimes.get('tensorflow_version', 'Not installed')}",
            f"CUDA Runtime:       {'Available' if self.runtimes.get('cuda_available') else 'Not Available'}",
            "",
            "Supported Backends",
            "-" * 50,
            f"[{'✓' if self.supported_backends.get('cpu') else ' '}] CPU",
            f"[{'✓' if self.supported_backends.get('cuda') else ' '}] CUDA",
            f"[{'✓' if self.supported_backends.get('rocm') else ' '}] ROCm/HIP (Interface ready)",
            "=" * 50
        ])
        return "\n".join(lines)


class HardwareProfiler:
    """Introspects host hardware, accelerators, and runtime libraries."""

    @staticmethod
    def profile() -> HardwareProfile:
        cpu = get_cpu_info()
        gpus = get_gpu_info()

        # Check PyTorch & CUDA
        torch_ver = None
        cuda_avail = False
        try:
            import torch
            torch_ver = torch.__version__
            cuda_avail = torch.cuda.is_available()
        except ImportError:
            pass

        # Check TensorFlow
        tf_ver = None
        try:
            import tensorflow as tf
            tf_ver = tf.__version__
        except ImportError:
            pass

        # Determine backend support
        backends = {
            "cpu": True,
            "cuda": cuda_avail,
            "rocm": False  # Interface supported, runtime not active
        }

        runtimes = {
            "pytorch_version": torch_ver,
            "tensorflow_version": tf_ver,
            "cuda_available": cuda_avail,
        }

        return HardwareProfile(
            cpu=cpu,
            gpus=gpus,
            runtimes=runtimes,
            supported_backends=backends
        )
