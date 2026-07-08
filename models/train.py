"""Model training: split, grid-search, fit best estimator, persist.

Generalises notebook cells 83-91. The classifier and hyper-parameter grid are
driven by the config so the same routine works for any area / target crop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.config import PipelineConfig


def split_data(X: pd.DataFrame, y: pd.Series, config: PipelineConfig):
    """Stratified train/test split using config-driven settings."""
    return train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )


def grid_search(X_train, y_train, config: PipelineConfig) -> GridSearchCV:
    """Run a cross-validated Random Forest hyper-parameter search."""
    pipeline = Pipeline([("rf", RandomForestClassifier())])
    model = GridSearchCV(
        estimator=pipeline,
        param_grid=config.rf_param_grid,
        cv=config.cv_folds,
        scoring=["accuracy", "precision", "recall"],
        refit="accuracy",
    )
    model.fit(X_train, y_train)
    return model


def fit_best(X_train, y_train, best_params: dict) -> RandomForestClassifier:
    """Fit a fresh RandomForest using the best params from the grid search."""
    clean = {k.replace("rf__", ""): v for k, v in best_params.items()}
    clean.setdefault("random_state", 10)
    rf = RandomForestClassifier(**clean)
    rf.fit(X_train, y_train)
    return rf


def save_model(model, config: PipelineConfig, path: Optional[str | Path] = None) -> Path:
    """Persist a fitted model to disk with joblib."""
    path = Path(path or config.model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(config: PipelineConfig, path: Optional[str | Path] = None):
    """Load a persisted model from disk."""
    return joblib.load(Path(path or config.model_path))


def train(
    X: pd.DataFrame, y: pd.Series, config: PipelineConfig, save: bool = True
) -> Tuple[RandomForestClassifier, dict]:
    """End-to-end training: split -> search -> refit -> evaluate -> save.

    Returns the fitted best model and a metrics dict.
    """
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    search = grid_search(X_train, y_train, config)
    best = fit_best(X_train, y_train, search.best_params_)

    y_pred = best.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "best_params": search.best_params_,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    if save:
        metrics["model_path"] = str(save_model(best, config))
    return best, metrics
