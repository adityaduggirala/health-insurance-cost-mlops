# Health Insurance Cost Prediction: ML Service

An end-to-end ML project: train and compare models on health-insurance data, serve the best one
through a validated REST API with **prediction intervals**, and monitor incoming data for **drift**.
Built to go beyond a notebook: shared feature code for training and serving, tests, Docker and CI.

## Results (held-out 20% test set, 268 rows)

| Model | CV RMSE | Test RMSE | Test MAE | Test R² |
|---|---|---|---|---|
| Log-linear baseline | $8,816 | $8,953 | $4,103 | 0.457 |
| **Random forest** (selected on CV) | **$4,596** | $4,458 | $2,573 | 0.865 |
| Gradient boosting | $4,697 | $4,376 | $2,497 | 0.870 |

- The model is **selected on cross-validation only**, never on the test set, to avoid leakage.
- The selected model cuts RMSE by about 50% versus the baseline.
- The 10th-90th percentile interval (quantile gradient boosting) has **78.4% empirical coverage** on the test set, close to the nominal 80%.
- Permutation importance: **smoking status dominates** (about $10.8k RMSE increase when shuffled), then BMI and age.

> Numbers come from `python -m insurance.train`; results vary slightly with library versions. Re-run and update this table with your own output.

## Architecture

```
data/insurance.csv ─► validate ─► features.add_features ─► sklearn Pipeline ─► artifacts/model.joblib
                                        ▲                                            │
                                        └──── same code path at serving time ────────┤
client ─► POST /predict (pydantic validation) ─► point estimate + 80% interval ◄─────┘
new batch ─► python -m insurance.drift batch.csv ─► PSI per feature (stable / moderate / significant)
```

## Quickstart

```bash
pip install -r requirements.txt && pip install -e .
make train            # trains, writes artifacts/ (model, metrics.json, drift reference)
make serve            # http://localhost:8000/docs
make test
```

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"age":40,"sex":"male","bmi":32,"children":2,"smoker":"yes","region":"southeast"}'
# {"predicted_charges": 39226.55, "lower_80": 32635.75, "upper_80": 41245.2, ...}
```

Drift check on a new batch:

```bash
python -m insurance.drift new_batch.csv
```

Docker: `make docker` (the image trains the model at build time).

## What's in here

| Path | Purpose |
|---|---|
| `src/insurance/data.py` | Schema and range validation that fails fast |
| `src/insurance/features.py` | Feature engineering shared by train and serve, which prevents train/serve skew |
| `src/insurance/train.py` | Cross-validated model comparison, quantile intervals, permutation importance, metrics report |
| `src/insurance/api.py` | FastAPI service with typed request validation, `/health`, `/model-info` |
| `src/insurance/drift.py` | Population Stability Index drift monitor |
| `tests/` | Data validation, feature, drift, training and API tests |

## Design decisions and limitations

- **Tree ensembles over linear models**: costs jump when smoking and obesity combine, which a linear model on this data cannot capture well.
- **Dataset is small (1,338 rows) and US-centric**; treat outputs as a demonstration, not actuarial pricing.
- Protected attributes such as sex appear in the data. A production system needs a fairness review; permutation importance shows sex has near-zero effect here.

## Data

`data/insurance.csv` is the public "Medical Cost Personal Datasets" (from *Machine Learning with R* by Brett Lantz, also on Kaggle). Check the source's license before redistributing.

## Ideas to extend

- Track experiments with MLflow and add a model registry.
- Add SHAP explanations to the `/predict` response.
- Deploy on Render, Fly.io or AWS ECS; add request logging and a Prometheus `/metrics` endpoint.
