# Hardware Abstraction & AMD ROCm / HIP Extensibility

AutoOptimizeML is designed to eliminate vendor lock-in by separating the optimization engine, benchmarking harnesses, and deployment services from hardware-specific primitives.

---

## 1. Unified `ExecutionBackend` Interface

All execution targets implement the `ExecutionBackend` abstract interface (`autoopt/backends/base.py`):

```python
class ExecutionBackend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier ('cpu', 'cuda', 'rocm')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Runtime and hardware presence validation."""
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Block host until asynchronous GPU compute operations complete."""
        pass

    @abstractmethod
    def set_num_threads(self, n: int) -> None:
        """Configure worker/thread concurrency."""
        pass

    @abstractmethod
    def get_memory_allocated_mb(self) -> float:
        """Active device memory allocated."""
        pass

    @abstractmethod
    def get_peak_memory_mb(self) -> float:
        """Peak memory allocated during execution."""
        pass

    @abstractmethod
    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking stats."""
        pass
```

---

## 2. Implementing an AMD ROCm / HIP Backend

Because the entire optimization pipeline queries `ExecutionBackend.synchronize()`, `get_memory_allocated_mb()`, and `is_available()`, adding full AMD ROCm support requires zero changes to the optimization engine.

### Step 1: ROCm Runtime Detection (`autoopt/backends/rocm_backend.py`)
```python
import torch

class ROCmBackend(ExecutionBackend):
    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._available = False
        self._device_name = "AMD ROCm GPU"

        # Check for PyTorch built with HIP (ROCm) support
        if hasattr(torch.version, 'hip') and torch.version.hip is not None and torch.cuda.is_available():
            self._available = True
            self._device_name = torch.cuda.get_device_name(device_index)

    @property
    def name(self) -> str:
        return "rocm"

    def synchronize(self) -> None:
        if self._available:
            torch.cuda.synchronize(self._device_index)  # Under ROCm PyTorch, torch.cuda calls hipDeviceSynchronize
```

### Step 2: HIP Native Kernels (`native/src/tensor_ops_hip.cpp`)
For native C++ acceleration on AMD GPUs, HIP code can be compiled directly with `hipcc`:

```cpp
#include <hip/hip_runtime.h>

__global__ void normalize_nhwc_to_nchw_hip(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int height, int width, int channels,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    float scale
) {
    int idx = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    // Identical thread indexing logic as CUDA
}
```

### Step 3: CMake HIP Toolchain Integration
In `native/CMakeLists.txt`:
```cmake
find_package(HIP)
if(HIP_FOUND)
    set(ENABLE_ROCM ON)
    hip_add_library(autoopt_native MODULE ${SOURCES})
endif()
```

---

## 3. Benefits of This Architecture

* **Zero Optimizer Rewriting**: Adding a new hardware backend does not require changing search strategies, Bayesian surrogate models, or constraint verification.
* **Honest Introspection**: The framework detects hardware automatically without pretending unsupported accelerators are active.
* **Portable Deployment**: Deployed models run transparently behind the same FastAPI REST API across CPUs, NVIDIA GPUs, and AMD GPUs.
