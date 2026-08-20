"""Model Adapters package."""

from autoopt.models.base import ModelAdapter
from autoopt.models.sklearn_adapter import SklearnAdapter
from autoopt.models.pytorch_adapter import PyTorchAdapter
from autoopt.models.tensorflow_adapter import TensorFlowAdapter

def get_adapter(model: any, sample_input: any = None, test_data: any = None) -> ModelAdapter:
    """Automatically detect model framework and wrap with the appropriate ModelAdapter."""
    if isinstance(model, ModelAdapter):
        return model

    # Check PyTorch
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return PyTorchAdapter(model, sample_input=sample_input, test_data=test_data)
    except ImportError:
        pass

    # Check TensorFlow
    try:
        import tensorflow as tf
        if isinstance(model, (tf.keras.Model, tf.Module)):
            return TensorFlowAdapter(model, sample_input=sample_input, test_data=test_data)
    except ImportError:
        pass

    # Check Sklearn / BaseEstimator
    try:
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            return SklearnAdapter(model, sample_input=sample_input, test_data=test_data)
    except ImportError:
        pass

    # Check if object has predict method (Duck typing for sklearn-like)
    if hasattr(model, "predict"):
        return SklearnAdapter(model, sample_input=sample_input, test_data=test_data)

    raise ValueError(f"Unable to auto-detect framework adapter for model of type {type(model)}.")

__all__ = [
    "ModelAdapter",
    "SklearnAdapter",
    "PyTorchAdapter",
    "TensorFlowAdapter",
    "get_adapter"
]
