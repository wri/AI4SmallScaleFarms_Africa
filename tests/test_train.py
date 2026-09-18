from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from models.train import (
    _cv_splits,
    _test_scores,
    fit_best,
    load_model,
    model_class_names,
    save_model,
    split_data,
    train,
)


def _xy(n_per_class: int = 12):
    rows = []
    for label in (0, 1):
        for i in range(n_per_class):
            rows.append(
                {
                    "precipitation": 10.0 + label + i * 0.01,
                    "elevation": 1800.0 + label * 50 + i,
                    "slope": 5.0 + label,
                    "y": label,
                }
            )
    df = pd.DataFrame(rows)
    return df[["precipitation", "elevation", "slope"]], df["y"]


def test_cv_splits_caps_to_smallest_class(tmp_config):
    tmp_config.cv_folds = 10
    y = pd.Series([0] * 9 + [1] * 9)
    assert _cv_splits(y, tmp_config) == 9


def test_cv_splits_raises_when_class_too_small(tmp_config):
    tmp_config.cv_folds = 5
    with pytest.raises(ValueError, match="at least 2 samples"):
        _cv_splits(pd.Series([0, 1, 1, 1]), tmp_config)


def test_split_data_is_stratified(tmp_config):
    X, y = _xy(20)
    tmp_config.test_size = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, tmp_config)
    assert set(y_train.unique()) == {0, 1}
    assert set(y_test.unique()) == {0, 1}
    assert len(X_train) + len(X_test) == len(X)


def test_binary_and_multiclass_test_scores():
    binary = _test_scores([0, 1, 1, 0], [0, 1, 0, 0], binary=True, labels=[0, 1])
    assert "accuracy" in binary
    assert binary["confusion_matrix"] == [[2, 0], [1, 1]]
    multi = _test_scores(
        [1, 2, 1, 2], [1, 2, 2, 2],
        binary=False, labels=[1, 2], class_names=["maize", "wheat"],
    )
    assert "per_class" in multi
    assert "maize" in multi["per_class"]
    assert "precision_weighted" in multi


def test_model_class_names_from_attribute_and_metrics(tmp_config):
    class Fake:
        classes_ = np.array([1, 2])
        class_names_ = ["maize", "wheat"]

    assert model_class_names(Fake(), tmp_config) == ["maize", "wheat"]

    class FakeNoNames:
        classes_ = np.array([0, 1])

    tmp_config.label_mode = "binary"
    tmp_config.target_crop_names = ["Maize"]
    assert model_class_names(FakeNoNames(), tmp_config) == ["other", "Maize"]

    tmp_config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_config.metrics_path.write_text(
        '{"class_codes": [1, 2], "class_names": ["maize", "wheat"]}'
    )

    class FakeCodes:
        classes_ = np.array([1, 2])

    tmp_config.label_mode = "multiclass"
    assert model_class_names(FakeCodes(), tmp_config) == ["maize", "wheat"]


def test_save_and_load_model_roundtrip(tmp_config):
    X, y = _xy()
    rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
    path = save_model(rf, tmp_config)
    loaded = load_model(tmp_config, path=path)
    np.testing.assert_array_equal(loaded.predict(X), rf.predict(X))


def test_load_model_missing_raises(tmp_config):
    with pytest.raises(FileNotFoundError, match="No trained model"):
        load_model(tmp_config)


def test_train_binary_writes_metrics(tmp_config):
    X, y = _xy(16)
    tmp_config.label_mode = "binary"
    tmp_config.target_column = "maize_pos"
    tmp_config.feature_columns = list(X.columns)
    model, metrics = train(X, y, tmp_config, save=True, class_names=["other", "Maize"], class_codes=[0, 1])
    assert tmp_config.model_path.exists()
    assert tmp_config.metrics_path.exists()
    assert tmp_config.metrics_index_path.exists()
    assert metrics["label_mode"] == "binary"
    assert metrics["n_classes"] == 2
    assert model.class_names_ == ["other", "Maize"]
    table = pd.read_csv(tmp_config.metrics_index_path)
    assert "UnitTest" in set(table["area_name"])


def test_fit_best_passes_class_weight(tmp_config):
    X, y = _xy()
    tmp_config.class_weight = "balanced"
    rf = fit_best(X, y, {"rf__n_estimators": 8, "rf__random_state": 0}, tmp_config)
    assert rf.class_weight == "balanced"
    assert rf.n_estimators == 8
