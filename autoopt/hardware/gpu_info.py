"""GPU hardware, VRAM, compute capability, and driver introspection."""

import os
import glob
import subprocess
from typing import List, Dict, Any, Optional


def _detect_pci_gpus() -> List[Dict[str, Any]]:
    """Scan Linux sysfs for PCI display/VGA controllers."""
    devices = []
    # PCI vendor IDs: 0x10de = NVIDIA, 0x1002 = AMD, 0x8086 = Intel
    vendor_map = {
        "0x10de": "NVIDIA",
        "0x1002": "AMD",
        "0x8086": "Intel"
    }
    
    pci_paths = glob.glob("/sys/bus/pci/devices/*")
    for path in pci_paths:
        class_file = os.path.join(path, "class")
        vendor_file = os.path.join(path, "vendor")
        device_file = os.path.join(path, "device")
        
        if os.path.exists(class_file) and os.path.exists(vendor_file):
            try:
                with open(class_file, "r") as f:
                    pci_class = f.read().strip()
                # 0x030000 = VGA compatible, 0x030200 = 3D controller, 0x038000 = Display controller
                if pci_class.startswith("0x03"):
                    with open(vendor_file, "r") as f:
                        vendor_id = f.read().strip().lower()
                    with open(device_file, "r") as f:
                        device_id = f.read().strip().lower()
                    
                    vendor_name = vendor_map.get(vendor_id, f"Vendor {vendor_id}")
                    devices.append({
                        "pci_slot": os.path.basename(path),
                        "vendor_id": vendor_id,
                        "vendor": vendor_name,
                        "device_id": device_id,
                    })
            except Exception:
                continue
    return devices


def get_gpu_info() -> List[Dict[str, Any]]:
    """Retrieve all detected GPU devices, VRAM, and runtime capabilities."""
    gpus: List[Dict[str, Any]] = []

    # 1. Try PyTorch CUDA if available and initialized
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                cap = f"{props.major}.{props.minor}"
                vram_gb = round(props.total_memory / (1024 ** 3), 2)
                gpus.append({
                    "vendor": "NVIDIA",
                    "device_name": props.name,
                    "device_index": i,
                    "vram_total_gb": vram_gb,
                    "compute_capability": cap,
                    "cuda_available": True,
                    "driver_status": "Operational (CUDA Runtime Active)"
                })
            if gpus:
                return gpus
    except Exception:
        pass

    # 2. Try nvidia-smi if torch cuda wasn't available
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2
        )
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            for idx, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    vram_mb = float(parts[1]) if parts[1].replace('.', '', 1).isdigit() else 0.0
                    driver = parts[2] if len(parts) > 2 else "Unknown"
                    gpus.append({
                        "vendor": "NVIDIA",
                        "device_name": name,
                        "device_index": idx,
                        "vram_total_gb": round(vram_mb / 1024, 2),
                        "compute_capability": "Available",
                        "cuda_available": True,
                        "driver_status": f"Driver {driver}"
                    })
            if gpus:
                return gpus
    except Exception:
        pass

    # 3. Fallback: inspect PCI bus directly for hardware presence
    pci_gpus = _detect_pci_gpus()
    for pci in pci_gpus:
        vendor = pci["vendor"]
        dev_id = pci["device_id"]
        
        # Check known common names or fallback
        name = f"{vendor} GPU (Device {dev_id})"
        if dev_id in ("0x25ac", "0x2582"):
            name = "NVIDIA GeForce RTX 3050 Laptop GPU"
        elif dev_id == "0x15bf":
            name = "AMD Radeon 780M Graphics"

        # Check driver state
        driver_status = "No proprietary driver active"
        cuda_avail = False
        if vendor == "NVIDIA":
            if os.path.exists("/dev/nvidia0"):
                driver_status = "NVIDIA driver present"
            else:
                driver_status = "NVIDIA hardware detected (nouveau / uninitialized)"
        elif vendor == "AMD":
            if os.path.exists("/dev/kfd"):
                driver_status = "AMD ROCm KFD kernel driver active"
            else:
                driver_status = "AMD driver active"

        gpus.append({
            "vendor": vendor,
            "device_name": name,
            "device_index": len(gpus),
            "vram_total_gb": None,
            "compute_capability": None,
            "cuda_available": cuda_avail,
            "driver_status": driver_status,
            "pci_slot": pci.get("pci_slot")
        })

    return gpus
