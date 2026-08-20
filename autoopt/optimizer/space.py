"""Optimization search space and candidate generation."""

import itertools
import uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from autoopt.hardware.profiler import HardwareProfile
from autoopt.analyzer.model_analyzer import ModelProfile
from autoopt.profiling.bottleneck_analyzer import BottleneckReport


@dataclass
class OptimizationCandidate:
    candidate_id: str
    device: str
    precision: str
    batch_size: int
    workers: int
    compile_graph: bool
    native_preprocessing: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        jit = " + JIT" if self.compile_graph else ""
        native = " + Native" if self.native_preprocessing else ""
        return (
            f"Device={self.device.upper()}, "
            f"Precision={self.precision.upper()}, "
            f"Batch={self.batch_size}, "
            f"Workers={self.workers}{jit}{native}"
        )


class SearchSpace:
    """Constructs feasible and targeted candidate configurations."""

    @staticmethod
    def generate_candidates(
        hardware: HardwareProfile,
        model: ModelProfile,
        bottleneck: Optional[BottleneckReport] = None,
        custom_batch_sizes: Optional[List[int]] = None,
        custom_workers: Optional[List[int]] = None
    ) -> List[OptimizationCandidate]:
        # 1. Determine valid devices
        devices = []
        if hardware.supported_backends.get("cpu", True) and "cpu" in model.supported_devices:
            devices.append("cpu")
        if hardware.supported_backends.get("cuda", False) and "cuda" in model.supported_devices:
            devices.append("cuda")
        if not devices:
            devices = ["cpu"]

        # 2. Determine valid precisions
        precisions = [p for p in model.supported_precisions if p in ("fp32", "fp16", "int8")]
        if not precisions:
            precisions = ["fp32"]

        # 3. Determine batch sizes
        if custom_batch_sizes:
            batch_sizes = custom_batch_sizes
        else:
            # Standard power-of-two batch sizes up to 32
            batch_sizes = [1, 2, 4, 8, 16, 32]

        # 4. Determine worker thread counts
        max_physical = hardware.cpu.get("physical_cores", 4)
        if custom_workers:
            workers_list = custom_workers
        else:
            workers_list = [1, 2, 4, min(8, max_physical)]
            workers_list = sorted(list(set(w for w in workers_list if w <= max_physical or w == 1)))

        # 5. Graph compilation options
        compile_options = [False]
        if model.framework in ("pytorch", "tensorflow"):
            compile_options.append(True)

        # 6. Native preprocessing options
        native_options = [False]
        # If input is 4D image tensor, native preprocessing can be explored
        if model.input_shape and len(model.input_shape) == 4:
            native_options.append(True)

        candidates: List[OptimizationCandidate] = []
        seen = set()

        for dev, prec, b, w, cg, np_opt in itertools.product(
            devices, precisions, batch_sizes, workers_list, compile_options, native_options
        ):
            # Prune invalid combinations:
            # - FP16 on CPU if unsupported
            if dev == "cpu" and prec == "fp16":
                continue
            # - INT8 dynamic quantization usually applies to CPU
            if dev == "cuda" and prec == "int8":
                continue

            key = (dev, prec, b, w, cg, np_opt)
            if key not in seen:
                seen.add(key)
                cid = f"cand_{len(candidates)+1:03d}"
                candidates.append(OptimizationCandidate(
                    candidate_id=cid,
                    device=dev,
                    precision=prec,
                    batch_size=b,
                    workers=w,
                    compile_graph=cg,
                    native_preprocessing=np_opt
                ))

        return candidates
