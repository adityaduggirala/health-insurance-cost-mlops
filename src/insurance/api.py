"""FastAPI prediction service.   uvicorn insurance.api:app --reload"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config

state: dict = {}


def load_model() -> None:
    if config.MODEL_PATH.exists():
        state["bundle"] = joblib.load(config.MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Health Insurance Cost API", version="0.1.0", lifespan=lifespan)


class Applicant(BaseModel):
    age: int = Field(ge=18, le=100, examples=[35])
    sex: Literal["male", "female"]
    bmi: float = Field(ge=10, le=70, examples=[27.5])
    children: int = Field(ge=0, le=10, examples=[1])
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


class Prediction(BaseModel):
    predicted_charges: float
    lower_80: float
    upper_80: float
    model: str
    latency_ms: float


def _require_model():
    if "bundle" not in state:
        raise HTTPException(503, "Model not loaded. Run `python -m insurance.train` first.")
    return state["bundle"]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "bundle" in state}


@app.get("/model-info")
def model_info():
    _require_model()
    return json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else state["bundle"]["meta"]


@app.post("/predict", response_model=Prediction)
def predict(a: Applicant):
    b = _require_model()
    t0 = time.perf_counter()
    X = pd.DataFrame([a.model_dump()])
    point = float(b["model"].predict(X)[0])
    lo, hi = float(b["lower"].predict(X)[0]), float(b["upper"].predict(X)[0])
    lo, hi = min(lo, point), max(hi, point)  # keep point estimate inside its own interval
    return Prediction(predicted_charges=round(point, 2), lower_80=round(lo, 2), upper_80=round(hi, 2),
                      model=b["meta"]["selected_model"], latency_ms=round((time.perf_counter() - t0) * 1000, 2))
