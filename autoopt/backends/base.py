"""Abstract ExecutionBackend interface for hardware-agnostic execution."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ExecutionBackend(ABC):
    """Abstract interface defining execution, memory, and synchronization primitives."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the execution backend (e.g. 'cpu', 'cuda', 'rocm')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend runtime and hardware are available."""
        pass

    @abstractmethod
    def device_name(self) -> str:
        """Name/model of the compute device."""
        pass

    @abstractmethod
    def synchronize(self) -> None:
        """Block host until all asynchronous device operations have completed."""
        pass

    @abstractmethod
    def set_num_threads(self, n: int) -> None:
        """Configure intra-op / execution worker thread concurrency."""
        pass

    @abstractmethod
    def get_memory_allocated_mb(self) -> float:
        """Return currently allocated memory in megabytes."""
        pass

    @abstractmethod
    def get_peak_memory_mb(self) -> float:
        """Return peak allocated memory during execution in megabytes."""
        pass

    @abstractmethod
    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking counters."""
        pass

    @abstractmethod
    def get_device_properties(self) -> Dict[str, Any]:
        """Return detailed device specifications and capability metrics."""
        pass
