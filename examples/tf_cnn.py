"""Example Workload 3: TensorFlow/Keras Vision CNN."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def get_model(num_classes: int = 10) -> tf.keras.Model:
    """Build and initialize a TensorFlow/Keras CNN."""
    tf.random.set_seed(42)
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def get_sample_input(batch_size: int = 1) -> np.ndarray:
    """Generate synthetic image batch (B, H, W, C)."""
    np.random.seed(42)
    return np.random.randn(batch_size, 32, 32, 3).astype(np.float32)


def get_test_data(n_samples: int = 100, num_classes: int = 10):
    """Generate synthetic test dataset."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(n_samples,)).astype(np.int32)
    return X, y


if __name__ == "__main__":
    model = get_model()
    x = get_sample_input(2)
    out = model(x, training=False)
    print("TensorFlow CNN loaded. Output shape:", out.shape)
