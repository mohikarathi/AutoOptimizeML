"""Scikit-Learn Model Adapter for classical ML workloads."""

import sys
import numpy as np
from typing import Dict, Any, List, Optional
from autoopt.models.base import ModelAdapter


class SklearnAdapter(ModelAdapter):
    """Adapter for Scikit-Learn classifiers and regressors."""

    @property
    def framework(self) -> str:
        return "sklearn"

    def get_metadata(self) -> Dict[str, Any]:
        model_type = self.model.__class__.__name__
        n_features = getattr(self.model, "n_features_in_", None)
        if n_features is None and self.sample_input is not None:
            n_features = np.asarray(self.sample_input).shape[-1]

        classes = getattr(self.model, "classes_", None)
        n_classes = len(classes) if classes is not None else None

        n_estimators = getattr(self.model, "n_estimators", None)

        # Approximate model size in MB via sys.getsizeof on attributes
        size_bytes = sys.getsizeof(self.model)
        for attr in dir(self.model):
            if not attr.startswith("__"):
                try:
                    val = getattr(self.model, attr)
                    if hasattr(val, "nbytes"):
                        size_bytes += val.nbytes
                    else:
                        size_bytes += sys.getsizeof(val)
                except Exception:
                    pass

        return {
            "framework": "sklearn",
            "model_type": model_type,
            "n_features": n_features,
            "n_classes": n_classes,
            "n_estimators": n_estimators,
            "input_shape": [None, n_features] if n_features else None,
            "output_shape": [None, n_classes] if n_classes else [None, 1],
            "parameters": n_features * (n_estimators or 1),
            "trainable_parameters": 0,
            "model_size_mb": round(size_bytes / (1024 * 1024), 3),
            "layer_breakdown": [{"type": model_type, "params": n_features or 0}]
        }

    def supported_precisions(self) -> List[str]:
        return ["fp32", "fp64"]

    def supported_devices(self) -> List[str]:
        return ["cpu"]

    def prepare_for_inference(
        self,
        device: str = "cpu",
        precision: str = "fp32",
        compile_graph: bool = False,
        num_threads: Optional[int] = None
    ) -> Any:
        if num_threads and hasattr(self.model, "n_jobs"):
            try:
                self.model.n_jobs = num_threads
            except Exception:
                pass
        return self.model

    def preprocess(self, raw_input: Any) -> np.ndarray:
        arr = np.asarray(raw_input)
        if arr.dtype != np.float32 and arr.dtype != np.float64:
            arr = arr.astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def run_inference(self, prepared_model: Any, input_tensor: np.ndarray) -> np.ndarray:
        if hasattr(prepared_model, "predict_proba"):
            return prepared_model.predict_proba(input_tensor)
        return prepared_model.predict(input_tensor)

    def postprocess(self, model_output: np.ndarray) -> List[Any]:
        if model_output.ndim > 1 and model_output.shape[1] > 1:
            return np.argmax(model_output, axis=1).tolist()
        return model_output.tolist()

    def evaluate_accuracy(self, prepared_model: Any, test_data: Optional[Any] = None) -> float:
        data = test_data if test_data is not None else self.test_data
        if data is None:
            return 1.0
        X_test, y_test = data
        X_test = self.preprocess(X_test)
        preds = self.postprocess(self.run_inference(prepared_model, X_test))
        y_true = np.asarray(y_test).tolist()
        correct = sum(1 for p, y in zip(preds, y_true) if p == y)
        return round(correct / max(1, len(y_true)), 4)
