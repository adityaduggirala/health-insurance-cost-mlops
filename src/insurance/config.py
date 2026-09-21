from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "insurance.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
REFERENCE_PATH = ARTIFACT_DIR / "reference.json"

TARGET = "charges"
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
REGIONS = ["northeast", "northwest", "southeast", "southwest"]
SEED = 42
