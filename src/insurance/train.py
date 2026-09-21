"""Train, compare and persist models.   python -m insurance.train"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from . import config
from .data import load_data
from .features import make_pipeline_steps


def _pipeline(model) -> Pipeline:
    return Pipeline(make_pipeline_steps() + [("model", model)])


def candidates() -> dict:
    log_target = lambda m: TransformedTargetRegressor(m, func=np.log1p, inverse_func=np.expm1)
    return {
        "linear_baseline": _pipeline(log_target(LinearRegression())),
        "random_forest": _pipeline(RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=config.SEED, n_jobs=-1)),
        "gradient_boosting": _pipeline(GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=config.SEED)),
    }


def _metrics(y, pred) -> dict:
    return {"rmse": round(float(root_mean_squared_error(y, pred)), 2),
            "mae": round(float(mean_absolute_error(y, pred)), 2),
            "r2": round(float(r2_score(y, pred)), 4)}


def train(df: pd.DataFrame | None = None, save: bool = True, cv: int = 5) -> dict:
    df = load_data() if df is None else df
    X, y = df.drop(columns=[config.TARGET]), df[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=config.SEED, stratify=X["smoker"])

    kf = KFold(n_splits=cv, shuffle=True, random_state=config.SEED)
    results, fitted = {}, {}
    for name, pipe in candidates().items():
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=kf, scoring="neg_root_mean_squared_error").mean()
        pipe.fit(X_tr, y_tr)
        results[name] = {"cv_rmse": round(float(cv_rmse), 2), "test": _metrics(y_te, pipe.predict(X_te))}
        fitted[name] = pipe

    best = min(results, key=lambda n: results[n]["cv_rmse"])  # select on CV only, never on test
    best_pipe = fitted[best]

    # Prediction intervals via quantile gradient boosting (10th / 90th percentile).
    q = {}
    for label, alpha in (("lower", 0.1), ("upper", 0.9)):
        q[label] = _pipeline(GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=200, max_depth=3, learning_rate=0.05, random_state=config.SEED)).fit(X_tr, y_tr)
    lo, hi = q["lower"].predict(X_te), q["upper"].predict(X_te)
    coverage = float(np.mean((y_te.values >= lo) & (y_te.values <= hi)))

    imp = permutation_importance(best_pipe, X_te, y_te, n_repeats=10, random_state=config.SEED, scoring="neg_root_mean_squared_error")
    importance = dict(sorted(zip(X.columns, imp.importances_mean.round(1).tolist()), key=lambda kv: -kv[1]))

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_sha256": hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16],
        "n_train": len(X_tr), "n_test": len(X_te),
        "selected_model": best, "models": results,
        "interval_80_coverage": round(coverage, 3),
        "permutation_importance_rmse_increase": importance,
    }
    reference = {c: {"edges": np.quantile(X_tr[c], np.linspace(0, 1, 11)).tolist()} for c in config.NUMERIC}

    if save:
        config.ARTIFACT_DIR.mkdir(exist_ok=True)
        joblib.dump({"model": best_pipe, "lower": q["lower"], "upper": q["upper"], "meta": {k: report[k] for k in ("trained_at", "data_sha256", "selected_model")}}, config.MODEL_PATH)
        config.METRICS_PATH.write_text(json.dumps(report, indent=2))
        config.REFERENCE_PATH.write_text(json.dumps({"reference": {c: X_tr[c].tolist() for c in config.NUMERIC}}))
    return report


if __name__ == "__main__":
    rep = train()
    print(json.dumps({k: rep[k] for k in ("selected_model", "models", "interval_80_coverage")}, indent=2))
    print("top drivers:", list(rep["permutation_importance_rmse_increase"].items())[:3])
