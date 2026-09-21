import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from insurance import api, config, drift
from insurance.data import load_data, validate
from insurance.features import add_features
from insurance.train import train


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_data_validation_rejects_bad_rows(df):
    bad = df.copy()
    bad.loc[0, "bmi"] = 500
    with pytest.raises(ValueError):
        validate(bad, require_target=True)
    with pytest.raises(ValueError):
        validate(df.drop(columns=["smoker"]), require_target=True)


def test_feature_engineering(df):
    out = add_features(df.head(50))
    assert set(["obese", "smoker_flag", "smoker_obese"]) <= set(out.columns)
    assert (out["smoker_obese"] <= out["obese"]).all()


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(30, 5, 5000)
    assert drift.psi(ref, rng.normal(30, 5, 5000)) < 0.1
    assert drift.psi(ref, rng.normal(38, 5, 5000)) > 0.25


@pytest.fixture(scope="module")
def trained(tmp_path_factory, df):
    d = tmp_path_factory.mktemp("artifacts")
    config.ARTIFACT_DIR = d
    config.MODEL_PATH, config.METRICS_PATH, config.REFERENCE_PATH = d / "model.joblib", d / "metrics.json", d / "reference.json"
    report = train(df, save=True, cv=3)
    api.load_model()
    return report


def test_training_beats_baseline_and_is_accurate(trained):
    m = trained["models"]
    assert trained["selected_model"] != "linear_baseline"
    assert m[trained["selected_model"]]["test"]["r2"] > 0.8
    assert 0.6 < trained["interval_80_coverage"] < 0.95


def test_api_predict_and_validation(trained):
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok", "model_loaded": True}
    body = {"age": 40, "sex": "male", "bmi": 32.0, "children": 2, "smoker": "yes", "region": "southeast"}
    r = c.post("/predict", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["lower_80"] <= j["predicted_charges"] <= j["upper_80"]
    non_smoker = c.post("/predict", json={**body, "smoker": "no"}).json()["predicted_charges"]
    assert j["predicted_charges"] > 2 * non_smoker  # smoking is the dominant cost driver
    assert c.post("/predict", json={**body, "age": 5}).status_code == 422
    assert c.post("/predict", json={**body, "region": "mars"}).status_code == 422
