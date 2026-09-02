"""Model training: split, grid-search, fit best estimator, persist.

Generalises notebook cells 83-91. The classifier and hyper-parameter grid are
driven by the config so the same routine works for any area / target crop.
Metrics are written next to the pickle so areas can be compared.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
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


def _cv_splits(y_train, config: PipelineConfig) -> int:
    """Cap folds so every class can appear in each StratifiedKFold split."""
    counts = Counter(y_train)
    min_per_class = min(counts.values()) if counts else config.cv_folds
    n_splits = min(int(config.cv_folds), int(min_per_class))
    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 samples in the smallest training class for CV "
            f"(got {min_per_class}). Raise min_class_count or lower test_size."
        )
    if n_splits < config.cv_folds:
        print(
            f"cv_folds reduced from {config.cv_folds} to {n_splits} "
            f"(smallest train class has {min_per_class} samples)"
        )
    return n_splits


def _scoring(config: PipelineConfig):
    if config.is_binary:
        return ["accuracy", "precision", "recall"]
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="macro", zero_division=0),
        "recall": make_scorer(recall_score, average="macro", zero_division=0),
    }


def grid_search(X_train, y_train, config: PipelineConfig) -> GridSearchCV:
    """Run a cross-validated Random Forest hyper-parameter search."""
    rf_kwargs = {}
    if config.class_weight:
        rf_kwargs["class_weight"] = config.class_weight
    pipeline = Pipeline([("rf", RandomForestClassifier(**rf_kwargs))])
    model = GridSearchCV(
        estimator=pipeline,
        param_grid=config.rf_param_grid,
        cv=_cv_splits(y_train, config),
        scoring=_scoring(config),
        refit="accuracy",
    )
    model.fit(X_train, y_train)
    return model


def fit_best(X_train, y_train, best_params: dict, config: PipelineConfig) -> RandomForestClassifier:
    """Fit a fresh RandomForest using the best params from the grid search."""
    clean = {k.replace("rf__", ""): v for k, v in best_params.items()}
    clean.setdefault("random_state", 10)
    if config.class_weight:
        clean.setdefault("class_weight", config.class_weight)
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
    """Load a persisted model from disk.

    ``path`` overrides the config (CLI ``--model``). If the new
    ``data/models/`` location is empty, falls back to a pickle left in the
    old project-root ``models/`` folder.
    """
    candidates = []
    if path is not None:
        candidates.append(config.resolve_path(path) or Path(path))
    else:
        candidates.append(config.model_path)
        legacy = config.project_root / "models" / f"{config.slug}_rf_best_model.pkl"
        if legacy not in candidates:
            candidates.append(legacy)
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            if candidate != config.model_path:
                print(f"Loading model -> {candidate}")
            return joblib.load(candidate)
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"No trained model found. Tried: {tried}")


def model_class_names(model, config: PipelineConfig) -> List[str]:
    """Class names aligned with ``model.classes_`` (metrics JSON as fallback)."""
    attached = getattr(model, "class_names_", None)
    if attached:
        return [str(n) for n in attached]
    if config.metrics_path.exists():
        with open(config.metrics_path) as fh:
            payload = json.load(fh)
        names = payload.get("class_names")
        codes = payload.get("class_codes")
        if names and codes is not None:
            lookup = {c: n for c, n in zip(codes, names)}
            return [str(lookup.get(c, c)) for c in model.classes_]
        if names and len(names) == len(model.classes_):
            return [str(n) for n in names]
    if config.is_binary and len(model.classes_) == 2:
        crop = config.target_crop_names[0] if config.target_crop_names else "target"
        return ["other", crop]
    return [str(c) for c in model.classes_]


def _test_scores(y_test, y_pred, *, binary: bool, labels=None, class_names=None) -> dict:
    average = "binary" if binary else "macro"
    out = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
    }
    if not binary:
        out["precision_weighted"] = float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        out["recall_weighted"] = float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        out["f1_weighted"] = float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        p, r, f, s = precision_recall_fscore_support(
            y_test, y_pred, labels=labels, zero_division=0
        )
        names = list(class_names) if class_names is not None else [str(x) for x in (labels or [])]
        out["per_class"] = {
            names[i] if i < len(names) else str(labels[i]): {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in range(len(p))
        }
    return out


def _write_metrics(metrics: dict, config: PipelineConfig) -> Path:
    path = config.metrics_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(metrics, fh, indent=2)
    _update_comparison_table(metrics, config)
    print(f"Training metrics -> {path}")
    return path


def _update_comparison_table(metrics: dict, config: PipelineConfig) -> None:
    """Keep a one-row-per-area CSV so models can be compared at a glance."""
    row = {
        "area_name": metrics.get("area_name"),
        "label_mode": metrics.get("label_mode"),
        "trained_at": metrics.get("trained_at"),
        "test_accuracy": metrics.get("test", {}).get("accuracy"),
        "test_precision": metrics.get("test", {}).get("precision"),
        "test_recall": metrics.get("test", {}).get("recall"),
        "test_f1": metrics.get("test", {}).get("f1"),
        "cv_accuracy": metrics.get("cv_accuracy"),
        "n_classes": metrics.get("n_classes"),
        "n_train": metrics.get("n_train"),
        "n_test": metrics.get("n_test"),
        "n_features": metrics.get("n_features"),
        "start_date": metrics.get("start_date"),
        "end_date": metrics.get("end_date"),
        "model_path": metrics.get("model_path"),
    }
    table_path = config.metrics_index_path
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if table_path.exists():
        table = pd.read_csv(table_path)
        table = table[table["area_name"] != row["area_name"]]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table.to_csv(table_path, index=False)


def train(
    X: pd.DataFrame,
    y: pd.Series,
    config: PipelineConfig,
    save: bool = True,
    class_names: Optional[Sequence[str]] = None,
    class_codes: Optional[Sequence[int]] = None,
) -> Tuple[RandomForestClassifier, dict]:
    """End-to-end training: split -> search -> refit -> evaluate -> save.

    Returns the fitted best model and a metrics dict (also written to
    ``data/models/<area>_rf_metrics.json``).
    ``class_names`` / ``class_codes`` should be aligned with the integer
    label space (multiclass). Binary runs ignore them.
    """
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    search = grid_search(X_train, y_train, config)
    best = fit_best(X_train, y_train, search.best_params_, config)

    names_by_code = {}
    if class_codes is not None and class_names is not None:
        names_by_code = {int(c): str(n) for c, n in zip(class_codes, class_names)}
    best.class_names_ = [names_by_code.get(int(c), str(c)) for c in best.classes_]
    legend_names = list(best.class_names_)

    y_pred = best.predict(X_test)
    test = _test_scores(
        y_test, y_pred,
        binary=config.is_binary,
        labels=list(best.classes_),
        class_names=legend_names,
    )
    metrics = {
        "area_name": config.area_name,
        "slug": config.slug,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "label_mode": "binary" if config.is_binary else "multiclass",
        "target_column": config.target_column,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(best.classes_)),
        "class_codes": [int(c) for c in best.classes_],
        "class_names": legend_names,
        "test_size": config.test_size,
        "cv_folds": config.cv_folds,
        "cv_accuracy": float(search.best_score_),
        "test": test,
        "accuracy": test["accuracy"],
        "best_params": search.best_params_,
        "feature_columns": list(X.columns),
        "start_date": config.start_date,
        "end_date": config.end_date,
        "n_harmonics": config.n_harmonics,
        "scale": config.scale,
        "class_weight": config.class_weight,
    }
    if save:
        metrics["model_path"] = str(save_model(best, config))
        metrics["metrics_path"] = str(_write_metrics(metrics, config))
    return best, metrics
