"""Tests for Model Adapters and Model Analyzer."""

import numpy as np
import pytest
from autoopt.models import get_adapter, SklearnAdapter, PyTorchAdapter, TensorFlowAdapter
from autoopt.analyzer import ModelAnalyzer
import examples.sklearn_random_forest as ex_sklearn
import examples.pytorch_cnn as ex_pytorch
import examples.tf_cnn as ex_tf
import examples.emotion_analyzer as ex_emotion


def test_sklearn_adapter():
    model = ex_sklearn.get_model()
    sample = ex_sklearn.get_sample_input(2)
    test_data = ex_sklearn.get_test_data()

    adapter = get_adapter(model, sample_input=sample, test_data=test_data)
    assert isinstance(adapter, SklearnAdapter)
    assert adapter.framework == "sklearn"

    meta = adapter.get_metadata()
    assert meta["n_features"] == 20
    assert meta["model_type"] == "RandomForestClassifier"

    prep = adapter.preprocess(sample)
    out = adapter.run_inference(model, prep)
    preds = adapter.postprocess(out)
    assert len(preds) == 2

    acc = adapter.evaluate_accuracy(model)
    assert 0.0 <= acc <= 1.0


def test_pytorch_adapter():
    model = ex_pytorch.get_model()
    sample = ex_pytorch.get_sample_input(2)
    test_data = ex_pytorch.get_test_data(20)

    adapter = get_adapter(model, sample_input=sample, test_data=test_data)
    assert isinstance(adapter, PyTorchAdapter)
    assert adapter.framework == "pytorch"

    meta = adapter.get_metadata()
    assert meta["parameters"] > 100000
    assert meta["model_type"] == "SimpleVisionCNN"

    # Test JIT preparation
    jit_model = adapter.prepare_for_inference(device="cpu", precision="fp32", compile_graph=True)
    prep = adapter.preprocess(sample)
    out = adapter.run_inference(jit_model, prep)
    preds = adapter.postprocess(out)
    assert len(preds) == 2


def test_tensorflow_adapter():
    model = ex_tf.get_model()
    sample = ex_tf.get_sample_input(2)
    test_data = ex_tf.get_test_data(20)

    adapter = get_adapter(model, sample_input=sample, test_data=test_data)
    assert isinstance(adapter, TensorFlowAdapter)
    assert adapter.framework == "tensorflow"

    meta = adapter.get_metadata()
    assert meta["parameters"] > 100000

    prep = adapter.preprocess(sample)
    out = adapter.run_inference(model, prep)
    preds = adapter.postprocess(out)
    assert len(preds) == 2


def test_model_analyzer():
    model = ex_emotion.get_model()
    sample = ex_emotion.get_sample_input(1)
    profile = ModelAnalyzer.analyze(model, sample_input=sample)

    assert profile.framework == "pytorch"
    assert profile.parameters > 500000
    assert profile.layer_count > 3
    assert "Model Architecture Profile" in profile.format_cli()
