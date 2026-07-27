"""Prediction engine — loads pre-computed feature matrices, filters, predicts."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE    = Path(__file__).parent.parent
PRECOMP = Path(__file__).parent / "precomputed"
MODELS  = {
    "week":   BASE / "models" / "weeks.pkl",
    "month":  BASE / "models" / "month.pkl",
    "3month": BASE / "models" / "3month.pkl",
}

# object-dtype columns the model expects as categorical
_OBJ_W = [
    "Папка1", "Папка2", "Поставщик", "supplier",
    "МесяцПоследнегоПоступления", "ГодПоследнегоПоступления",
    "ТекущийМесяц", "ТекущийГод",
]
_OBJ_Q = _OBJ_W + ["ТекущийКвартал"]

_precomp: dict[str, tuple[pd.DataFrame, str]] = {}
_models:  dict[str, object] = {}
_catalog: pd.DataFrame | None = None


# ─────────────────────────── loaders ─────────────────────────────

def _get_precomp(horizon: str) -> tuple[pd.DataFrame, str]:
    if horizon not in _precomp:
        df     = pd.read_csv(PRECOMP / f"X_{horizon}.csv", sep=";",
                             dtype={"КодТовара": str, "Номенклатура": str})
        period = (PRECOMP / f"period_{horizon}.txt").read_text(encoding="utf-8")
        _precomp[horizon] = (df, period)
    return _precomp[horizon]


def _get_model(horizon: str):
    if horizon not in _models:
        _models[horizon] = joblib.load(MODELS[horizon])
    return _models[horizon]


def get_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is not None:
        return _catalog
    df, _ = _get_precomp("week")
    _catalog = df[["КодТовара", "Номенклатура", "Папка1", "Папка2"]].drop_duplicates()
    return _catalog


def warmup():
    """Load all pre-computed matrices and models into memory on server startup."""
    print("Warmup: загрузка матриц...", flush=True)
    _get_precomp("week")
    _get_precomp("month")
    _get_precomp("3month")
    get_catalog()
    print("Warmup: загрузка и прогрев моделей...", flush=True)
    for horizon, obj in [("week", _OBJ_W), ("month", _OBJ_W), ("3month", _OBJ_Q)]:
        df, _ = _get_precomp(horizon)
        X = df.iloc[:1].copy()
        X[obj] = X[obj].astype(object)
        _get_model(horizon).predict(X)
    print("Warmup завершён.", flush=True)


# ─────────────────────────── helpers ─────────────────────────────

def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if not filters:
        return df
    if filters.get("papa1"):
        df = df[df["Папка1"].isin(filters["papa1"])]
    if filters.get("papa2"):
        df = df[df["Папка2"].isin(filters["papa2"])]
    if filters.get("name_search"):
        df = df[df["Номенклатура"].str.contains(filters["name_search"], case=False, na=False)]
    if filters.get("ids"):
        df = df[df["КодТовара"].isin(filters["ids"])]
    return df


# ─────────────────────────── predict ─────────────────────────────

def predict_weekly(filters: dict) -> tuple[pd.DataFrame, str]:
    import time
    print("[week] precomp...", flush=True); t = time.time()
    X_all, period = _get_precomp("week")
    print(f"[week] precomp ok ({time.time()-t:.2f}s)", flush=True)
    X = _apply_filters(X_all, filters).copy()
    print(f"[week] filter → {len(X)} rows", flush=True)
    if X.empty:
        return pd.DataFrame(), period

    prod = X[["КодТовара", "Номенклатура", "Папка1", "Папка2"]].copy()
    X[_OBJ_W] = X[_OBJ_W].astype(object)

    print("[week] predict...", flush=True); t = time.time()
    preds = np.maximum(0, _get_model("week").predict(X))
    print(f"[week] predict ok ({time.time()-t:.2f}s)", flush=True)
    prod["Прогноз"] = np.round(preds).astype(int)
    return prod.reset_index(drop=True), period


def predict_monthly(filters: dict) -> tuple[pd.DataFrame, str]:
    X_all, period = _get_precomp("month")
    X = _apply_filters(X_all, filters).copy()
    if X.empty:
        return pd.DataFrame(), period

    prod = X[["КодТовара", "Номенклатура", "Папка1", "Папка2"]].copy()
    X[_OBJ_W] = X[_OBJ_W].astype(object)

    preds = np.maximum(0, _get_model("month").predict(X))
    prod["Прогноз"] = np.round(preds).astype(int)
    return prod.reset_index(drop=True), period


def predict_quarterly(filters: dict) -> tuple[pd.DataFrame, str]:
    X_all, period = _get_precomp("3month")
    X = _apply_filters(X_all, filters).copy()
    if X.empty:
        return pd.DataFrame(), period

    prod = X[["КодТовара", "Номенклатура", "Папка1", "Папка2"]].copy()
    X[_OBJ_Q] = X[_OBJ_Q].astype(object)

    preds = np.maximum(0, _get_model("3month").predict(X))
    prod["Прогноз"] = np.round(preds).astype(int)
    return prod.reset_index(drop=True), period


def predict(horizon: str, filters: dict) -> tuple[pd.DataFrame, str]:
    if horizon == "week":
        return predict_weekly(filters)
    if horizon == "month":
        return predict_monthly(filters)
    return predict_quarterly(filters)
