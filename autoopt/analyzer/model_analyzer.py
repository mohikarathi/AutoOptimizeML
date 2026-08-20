"""Model Architecture and Compute Analyzer."""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from autoopt.models import get_adapter, ModelAdapter


@dataclass
class ModelProfile:
    framework: str
    model_type: str
    input_shape: Optional[List[Any]]
    output_shape: Optional[List[Any]]
    parameters: int
    trainable_parameters: int
    model_size_mb: float
    estimated_mflops: Optional[float]
    layer_count: int
    layer_breakdown: List[Dict[str, Any]]
    supported_precisions: List[str]
    supported_devices: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def format_cli(self) -> str:
        lines = [
            "=" * 50,
            "             Model Architecture Profile",
            "=" * 50,
            f"Framework:           {self.framework.upper()}",
            f"Model Type:          {self.model_type}",
            f"Input Shape:         {self.input_shape}",
            f"Output Shape:        {self.output_shape}",
            f"Total Parameters:    {self.parameters:,}",
            f"Trainable Params:    {self.trainable_parameters:,}",
            f"Model Size (MB):     {self.model_size_mb} MB",
            f"Estimated Compute:   {f'{self.estimated_mflops:.2f} MFLOPs' if self.estimated_mflops else 'N/A'}",
            f"Layer Count:         {self.layer_count}",
            f"Supported Precision: {', '.join(self.supported_precisions)}",
            f"Supported Devices:   {', '.join(self.supported_devices)}",
            "-" * 50,
            "Layer Breakdown Summary:"
        ]
        if self.layer_breakdown:
            for l in self.layer_breakdown[:10]:
                name = l.get('name', l.get('type', 'Layer'))
                l_type = l.get('type', '')
                params = l.get('params', 0)
                lines.append(f"  • {name} ({l_type}): {params:,} params")
            if len(self.layer_breakdown) > 10:
                lines.append(f"  ... and {len(self.layer_breakdown) - 10} more layers")
        lines.append("=" * 50)
        return "\n".join(lines)


class ModelAnalyzer:
    """Introspects model graphs, parameters, FLOPs, and compute characteristics."""

    @staticmethod
    def analyze(model_or_adapter: Any, sample_input: Any = None) -> ModelProfile:
        adapter = model_or_adapter if isinstance(model_or_adapter, ModelAdapter) else get_adapter(model_or_adapter, sample_input=sample_input)
        meta = adapter.get_metadata()

        # Estimate FLOPs where practical
        mflops = None
        if adapter.framework == "pytorch":
            mflops = ModelAnalyzer._estimate_pytorch_mflops(adapter.model, adapter.sample_input)
        elif adapter.framework == "tensorflow":
            mflops = ModelAnalyzer._estimate_tf_mflops(adapter.model, adapter.sample_input)

        return ModelProfile(
            framework=meta["framework"],
            model_type=meta["model_type"],
            input_shape=meta.get("input_shape"),
            output_shape=meta.get("output_shape"),
            parameters=meta.get("parameters", 0),
            trainable_parameters=meta.get("trainable_parameters", 0),
            model_size_mb=meta.get("model_size_mb", 0.0),
            estimated_mflops=mflops,
            layer_count=meta.get("layer_count", len(meta.get("layer_breakdown", []))),
            layer_breakdown=meta.get("layer_breakdown", []),
            supported_precisions=adapter.supported_precisions(),
            supported_devices=adapter.supported_devices()
        )

    @staticmethod
    def _estimate_pytorch_mflops(model: Any, sample_input: Any) -> Optional[float]:
        try:
            import torch
            import torch.nn as nn
            total_flops = 0
            # Rough analytical estimation based on layer types
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    # 2 * in_features * out_features
                    total_flops += 2 * m.in_features * m.out_features
                elif isinstance(m, nn.Conv2d):
                    # 2 * Cin * Kh * Kw * Cout * (Hout * Wout)
                    # Assume approximate feature map size from kernel or default
                    h_out = 32
                    w_out = 32
                    if sample_input is not None and hasattr(sample_input, 'shape') and len(sample_input.shape) == 4:
                        h_out = sample_input.shape[2]
                        w_out = sample_input.shape[3]
                    kh, kw = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
                    total_flops += 2 * m.in_channels * kh * kw * m.out_channels * (h_out * w_out)
            return round(total_flops / 1e6, 2) if total_flops > 0 else None
        except Exception:
            return None

    @staticmethod
    def _estimate_tf_mflops(model: Any, sample_input: Any) -> Optional[float]:
        try:
            total_flops = 0
            if hasattr(model, "layers"):
                for layer in model.layers:
                    name = layer.__class__.__name__.lower()
                    if "dense" in name:
                        units = getattr(layer, "units", 0)
                        in_dim = getattr(layer, "input_shape", (None, 0))[-1] or 0
                        total_flops += 2 * in_dim * units
                    elif "conv" in name:
                        filters = getattr(layer, "filters", 0)
                        k_size = getattr(layer, "kernel_size", (3, 3))
                        in_dim = getattr(layer, "input_shape", (None, 32, 32, 3))[-1] or 3
                        total_flops += 2 * in_dim * k_size[0] * k_size[1] * filters * (32 * 32)
            return round(total_flops / 1e6, 2) if total_flops > 0 else None
        except Exception:
            return None
