"""Data-drift monitoring with Population Stability Index (PSI).
   python -m insurance.drift new_batch.csv
Rule of thumb: PSI < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import config


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(current: pd.DataFrame, reference: dict | None = None) -> dict:
    ref = reference or json.loads(config.REFERENCE_PATH.read_text())["reference"]
    out = {}
    for col in config.NUMERIC:
        score = psi(ref[col], current[col])
        out[col] = {"psi": round(score, 4), "status": "stable" if score < 0.1 else "moderate" if score < 0.25 else "significant"}
    return out


if __name__ == "__main__":
    print(json.dumps(drift_report(pd.read_csv(sys.argv[1])), indent=2))
