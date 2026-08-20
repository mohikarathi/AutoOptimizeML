"""Abstract ModelAdapter interface for framework-agnostic model interaction."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple


class ModelAdapter(ABC):
    """Abstract base class wrapping ML/DL models from various frameworks."""

    def __init__(self, model: Any, sample_input: Any = None, test_data: Any = None):
        self.model = model
        self.sample_input = sample_input
        self.test_data = test_data

    @property
    @abstractmethod
    def framework(self) -> str:
        """Name of the framework (e.g. 'pytorch', 'tensorflow', 'sklearn')."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Extract model architecture, parameters, input/output shapes, and size."""
        pass

    @abstractmethod
    def supported_precisions(self) -> List[str]:
        """List precision formats supported by this model (e.g. ['fp32', 'fp16'])."""
        pass

    @abstractmethod
    def supported_devices(self) -> List[str]:
        """List execution devices supported by this model (e.g. ['cpu', 'cuda'])."""
        pass

    @abstractmethod
    def prepare_for_inference(
        self,
        device: str = "cpu",
        precision: str = "fp32",
        compile_graph: bool = False,
        num_threads: Optional[int] = None
    ) -> Any:
        """Prepare and optimize model for execution on the target device/precision."""
        pass

    @abstractmethod
    def preprocess(self, raw_input: Any) -> Any:
        """Preprocess raw input into tensor or array format."""
        pass

    @abstractmethod
    def run_inference(self, prepared_model: Any, input_tensor: Any) -> Any:
        """Execute inference step."""
        pass

    @abstractmethod
    def postprocess(self, model_output: Any) -> Any:
        """Transform model output into user-facing predictions."""
        pass

    @abstractmethod
    def evaluate_accuracy(self, prepared_model: Any, test_data: Optional[Any] = None) -> float:
        """Compute evaluation metric (e.g. accuracy 0.0-1.0) on test dataset."""
        pass
