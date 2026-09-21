"""Feature engineering shared by training and serving (single source of truth)."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from . import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features: obesity flag and the smoker x obese interaction, which is the
    dominant cost driver in this data."""
    out = df.copy()
    out["obese"] = (out["bmi"] >= 30).astype(int)
    out["smoker_flag"] = (out["smoker"] == "yes").astype(int)
    out["smoker_obese"] = out["obese"] * out["smoker_flag"]
    return out


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), config.CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def make_pipeline_steps():
    return [("features", FunctionTransformer(add_features)), ("prep", build_preprocessor())]
