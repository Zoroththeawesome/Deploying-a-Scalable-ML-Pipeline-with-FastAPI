import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, compute_model_metrics
from ml.data import process_data


def test_train_model_returns_random_forest():
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_values():
    """
    Test that compute_model_metrics returns correct precision, recall, and fbeta.
    """
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == 1.0
    assert recall == 0.5
    assert round(fbeta, 2) == 0.67


def test_process_data_shapes():
    """
    Test that process_data returns X and y with expected shapes.
    """
    df = pd.DataFrame({
        "age": [25, 32, 40],
        "workclass": ["Private", "Self-emp", "Private"],
        "salary": ["<=50K", ">50K", "<=50K"]
    })

    cat_features = ["workclass"]

    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == 3
    assert y.shape[0] == 3
    assert encoder is not None
    assert lb is not None
