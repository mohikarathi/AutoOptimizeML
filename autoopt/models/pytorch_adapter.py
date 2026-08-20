"""PyTorch Model Adapter for deep learning workloads."""

import copy
import numpy as np
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from autoopt.models.base import ModelAdapter


class PyTorchAdapter(ModelAdapter):
    """Adapter for PyTorch nn.Module models."""

    def __init__(self, model: nn.Module, sample_input: Any = None, test_data: Any = None):
        super().__init__(model, sample_input, test_data)
        self.model.eval()

    @property
    def framework(self) -> str:
        return "pytorch"

    def get_metadata(self) -> Dict[str, Any]:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Estimate size in MB
        param_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size_bytes = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = round((param_size_bytes + buffer_size_bytes) / (1024 * 1024), 3)

        layer_breakdown = []
        for name, module in self.model.named_modules():
            if name != "" and not list(module.children()):
                layer_breakdown.append({
                    "name": name,
                    "type": module.__class__.__name__,
                    "params": sum(p.numel() for p in module.parameters())
                })

        input_shape = None
        if self.sample_input is not None:
            if isinstance(self.sample_input, torch.Tensor):
                input_shape = list(self.sample_input.shape)
            elif isinstance(self.sample_input, np.ndarray):
                input_shape = list(self.sample_input.shape)

        model_type = self.model.__class__.__name__

        return {
            "framework": "pytorch",
            "model_type": model_type,
            "input_shape": input_shape,
            "output_shape": None,
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb,
            "layer_count": len(layer_breakdown),
            "layer_breakdown": layer_breakdown
        }

    def supported_precisions(self) -> List[str]:
        precisions = ["fp32"]
        if torch.cuda.is_available():
            precisions.append("fp16")
        # Dynamic quantization is supported on CPU for Linear layers
        has_linear = any(isinstance(m, nn.Linear) for m in self.model.modules())
        if has_linear:
            precisions.append("int8")
        return precisions

    def supported_devices(self) -> List[str]:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        return devices

    def prepare_for_inference(
        self,
        device: str = "cpu",
        precision: str = "fp32",
        compile_graph: bool = False,
        num_threads: Optional[int] = None
    ) -> Any:
        if num_threads:
            torch.set_num_threads(num_threads)

        model = copy.deepcopy(self.model)
        model.eval()

        # Handle Precision
        if precision == "fp16":
            if device == "cuda" and torch.cuda.is_available():
                model = model.half()
            else:
                # If on CPU, bfloat16 or half if supported
                try:
                    model = model.half()
                except Exception:
                    pass
        elif precision == "int8":
            try:
                # Dynamic quantization for Linear layers
                model = torch.ao.quantization.quantize_dynamic(
                    model, {nn.Linear}, dtype=torch.qint8
                )
            except Exception:
                pass

        # Handle Device
        if device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        else:
            model = model.to("cpu")

        # Handle Graph Compilation (TorchScript JIT)
        if compile_graph and self.sample_input is not None:
            try:
                sample = self.preprocess(self.sample_input)
                if device == "cuda" and torch.cuda.is_available():
                    sample = sample.to("cuda")
                if precision == "fp16" and sample.dtype == torch.float32:
                    sample = sample.half()
                model = torch.jit.trace(model, sample)
            except Exception:
                pass

        return model

    def preprocess(self, raw_input: Any) -> torch.Tensor:
        if isinstance(raw_input, torch.Tensor):
            return raw_input
        if isinstance(raw_input, np.ndarray):
            return torch.from_numpy(raw_input).float()
        return torch.tensor(raw_input, dtype=torch.float32)

    def run_inference(self, prepared_model: Any, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            try:
                first_param = next(prepared_model.parameters(), None)
                if first_param is not None and first_param.dtype == torch.float16:
                    if input_tensor.dtype != torch.float16:
                        input_tensor = input_tensor.half()
            except Exception:
                pass
            return prepared_model(input_tensor)

    def postprocess(self, model_output: torch.Tensor) -> List[Any]:
        if isinstance(model_output, torch.Tensor):
            out = model_output.detach().cpu().numpy()
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
        
        # Match device of prepared model
        try:
            device = next(prepared_model.parameters()).device
            X_tensor = X_tensor.to(device)
            if next(prepared_model.parameters()).dtype == torch.float16:
                X_tensor = X_tensor.half()
        except Exception:
            pass

        with torch.no_grad():
            out = self.run_inference(prepared_model, X_tensor)
            preds = self.postprocess(out)
        
        if isinstance(y_test, torch.Tensor):
            y_true = y_test.cpu().numpy().tolist()
        else:
            y_true = np.asarray(y_test).tolist()

        correct = sum(1 for p, y in zip(preds, y_true) if p == y)
        return round(correct / max(1, len(y_true)), 4)
