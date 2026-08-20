"""Example Workload 1: Scikit-Learn Random Forest Classifier (Tabular ML)."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


def get_model(n_estimators: int = 50, random_state: int = 42):
    """Generate and fit a lightweight Random Forest model for benchmarking."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=random_state
    )
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=8)
    clf.fit(X[:800], y[:800])
    return clf


def get_sample_input(batch_size: int = 1):
    """Return synthetic sample tabular vector."""
    np.random.seed(42)
    return np.random.randn(batch_size, 20).astype(np.float32)


def get_test_data():
    """Return test evaluation dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_classes=3,
        random_state=42
    )
    return X[800:], y[800:]


if __name__ == "__main__":
    model = get_model()
    sample = get_sample_input(4)
    preds = model.predict(sample)
    print("Sklearn RandomForest loaded. Sample predictions:", preds)
