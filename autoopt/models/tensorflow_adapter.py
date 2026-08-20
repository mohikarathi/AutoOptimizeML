"""TensorFlow / Keras Model Adapter."""

import numpy as np
from typing import Dict, Any, List, Optional
import tensorflow as tf
from autoopt.models.base import ModelAdapter


class TensorFlowAdapter(ModelAdapter):
    """Adapter for TensorFlow / tf.keras models."""

    @property
    def framework(self) -> str:
        return "tensorflow"

    def get_metadata(self) -> Dict[str, Any]:
        total_params = self.model.count_params() if hasattr(self.model, "count_params") else 0
        trainable_params = int(sum(np.prod(w.shape) for w in self.model.trainable_weights)) if hasattr(self.model, "trainable_weights") else 0

        input_shape = None
        if hasattr(self.model, "input_shape"):
            input_shape = list(self.model.input_shape)
        elif self.sample_input is not None:
            input_shape = list(np.asarray(self.sample_input).shape)

        layer_breakdown = []
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                layer_breakdown.append({
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "params": layer.count_params() if hasattr(layer, "count_params") else 0
                })

        # Approximate model size in MB (float32 = 4 bytes per param)
        model_size_mb = round((total_params * 4) / (1024 * 1024), 3)

        return {
            "framework": "tensorflow",
            "model_type": self.model.__class__.__name__,
            "input_shape": input_shape,
            "output_shape": list(self.model.output_shape) if hasattr(self.model, "output_shape") else None,
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "layer_count": len(layer_breakdown),
            "layer_breakdown": layer_breakdown
        }

    def supported_precisions(self) -> List[str]:
        # TensorFlow supports float32 and mixed_float16 if GPU is active
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return ["fp32", "fp16"]
        return ["fp32"]

    def supported_devices(self) -> List[str]:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            return ["cpu", "cuda"]
        return ["cpu"]

    def prepare_for_inference(
        self,
        device: str = "cpu",
        precision: str = "fp32",
        compile_graph: bool = False,
        num_threads: Optional[int] = None
    ) -> Any:
        if num_threads:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(num_threads)
            except Exception:
                pass

        if compile_graph:
            @tf.function(jit_compile=True)
            def compiled_fn(x):
                return self.model(x, training=False)
            return compiled_fn

        return self.model

    def preprocess(self, raw_input: Any) -> tf.Tensor:
        arr = np.asarray(raw_input, dtype=np.float32)
        return tf.convert_to_tensor(arr)

    def run_inference(self, prepared_model: Any, input_tensor: tf.Tensor) -> tf.Tensor:
        if callable(prepared_model) and not hasattr(prepared_model, "predict"):
            return prepared_model(input_tensor)
        return prepared_model(input_tensor, training=False)

    def postprocess(self, model_output: Any) -> List[Any]:
        if isinstance(model_output, tf.Tensor):
            out = model_output.numpy()
        else:
            out = np.asarray(model_output)

        if out.ndim > 1 and out.shape[1] > 1:
            return np.argmax(out, axis=1).tolist()
        return out.tolist()

    def evaluate_accuracy(self, prepared_model: Any, test_data: Optional[Any] = None) -> float:
        data = test_data if test_data is not None else self.test_data
        if data is None:
            return 1.0
        X_test, y_test = data
        X_tensor = self.preprocess(X_test)
        out = self.run_inference(prepared_model, X_tensor)
        preds = self.postprocess(out)
        y_true = np.asarray(y_test).tolist()
        correct = sum(1 for p, y in zip(preds, y_true) if p == y)
        return round(correct / max(1, len(y_true)), 4)
