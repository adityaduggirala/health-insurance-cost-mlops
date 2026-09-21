import pandas as pd

from . import config

REQUIRED = config.NUMERIC + config.CATEGORICAL


def load_data(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or config.DATA_PATH)
    validate(df, require_target=True)
    return df


def validate(df: pd.DataFrame, require_target: bool = False) -> None:
    """Fail fast on schema or range problems."""
    cols = REQUIRED + ([config.TARGET] if require_target else [])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df[cols].isna().any().any():
        raise ValueError("null values present")
    if not df["age"].between(0, 120).all():
        raise ValueError("age out of range")
    if not df["bmi"].between(5, 80).all():
        raise ValueError("bmi out of range")
    if set(df["smoker"]) - {"yes", "no"}:
        raise ValueError("unexpected smoker values")
    if set(df["region"]) - set(config.REGIONS):
        raise ValueError("unexpected region values")
    if require_target and (df[config.TARGET] <= 0).any():
        raise ValueError("non-positive charges")
