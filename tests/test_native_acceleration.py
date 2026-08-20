"""Tests for Native C++ and CUDA acceleration engine."""

import numpy as np
import pytest


def test_native_module_import_and_execution():
    try:
        from autoopt import autoopt_native
    except ImportError:
        pytest.skip("autoopt_native shared library not compiled")

    info = autoopt_native.get_native_backend_info()
    assert hasattr(info, "cuda_enabled")
    assert hasattr(info, "openmp_enabled")
    assert hasattr(info, "thread_count")
    assert info.thread_count >= 1

    # Verify numerical correctness against numpy
    batch_size = 4
    h, w, c = 32, 32, 3
    inputs = np.random.randint(0, 256, (batch_size, h, w, c)).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    out_native = autoopt_native.normalize_nhwc_to_nchw_cpu(inputs, mean, std, 1.0 / 255.0, 4)
    assert out_native.shape == (batch_size, c, h, w)

    expected = ((inputs / 255.0) - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    expected = np.transpose(expected, (0, 3, 1, 2))

    max_diff = np.max(np.abs(out_native - expected))
    assert max_diff < 1e-5
