"""Cálculo da matriz de incerteza Omega (Black-Litterman)."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .config import (
        CONFIDENCE_CAP,
        CONFIDENCE_FLOOR,
        EPS,
        ERR_MIN_PERIODS,
        ERR_WINDOW,
        TAU,
        W_MODEL_CONF,
        W_NEWS_CONF,
    )
    from .xgb_views import normalize_return_columns
except ImportError:
    from config import (
        CONFIDENCE_CAP,
        CONFIDENCE_FLOOR,
        EPS,
        ERR_MIN_PERIODS,
        ERR_WINDOW,
        TAU,
        W_MODEL_CONF,
        W_NEWS_CONF,
    )
    from xgb_views import normalize_return_columns

__all__ = ["calculate_omega"]


def _safe_sample_var(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    var_value = float(clean.var(ddof=1))
    if not np.isfinite(var_value) or var_value <= 0:
        return np.nan
    return var_value


def _compute_confidence_per_ticker(group: pd.DataFrame) -> pd.DataFrame:
    out = normalize_return_columns(group).sort_values("view_date").copy()
    out["pred_abs_error"] = (out["ret_pred"] - out["ret_real"]).abs()

    out["error_scale"] = (
        out["pred_abs_error"].rolling(ERR_WINDOW, min_periods=ERR_MIN_PERIODS).median().shift(1)
    )
    fallback = float(out["pred_abs_error"].median())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 0.005
    out["error_scale"] = out["error_scale"].fillna(fallback).clip(lower=EPS)

    out["model_confidence"] = np.exp(-out["pred_abs_error"] / out["error_scale"])
    out["news_confidence"] = 1.0 - np.exp(-out["news_count"] / 3.0)
    out["view_confidence"] = (
        W_MODEL_CONF * out["model_confidence"] + W_NEWS_CONF * out["news_confidence"]
    ).clip(lower=CONFIDENCE_FLOOR, upper=CONFIDENCE_CAP)
    return out


def calculate_omega(q_long: pd.DataFrame, tau: float = TAU) -> pd.DataFrame:
    q_long = normalize_return_columns(q_long)

    sigma_by_ticker: dict[str, float] = {}
    for ticker, group in q_long.groupby("ticker", sort=True):
        sigma_ii = _safe_sample_var(group["ret_real"])
        if not np.isfinite(sigma_ii):
            sigma_ii = _safe_sample_var(group["ret_pred"])
        if not np.isfinite(sigma_ii):
            sigma_ii = 2.5e-5
        sigma_by_ticker[ticker] = sigma_ii

    parts = [_compute_confidence_per_ticker(g) for _, g in q_long.groupby("ticker", sort=True)]
    with_conf = pd.concat(parts, axis=0, ignore_index=True)

    with_conf["sigma_ii"] = with_conf["ticker"].map(sigma_by_ticker)
    with_conf["omega_base"] = tau * with_conf["sigma_ii"]
    with_conf["omega"] = with_conf["omega_base"] * (1.0 - with_conf["view_confidence"]) / with_conf["view_confidence"]
    with_conf["omega"] = with_conf["omega"].clip(lower=EPS)
    with_conf["tau"] = tau

    return with_conf[
        [
            "view_date",
            "ticker",
            "omega",
            "omega_base",
            "view_confidence",
            "model_confidence",
            "news_confidence",
            "pred_abs_error",
            "error_scale",
            "sigma_ii",
            "tau",
        ]
    ].sort_values(["view_date", "ticker"]).reset_index(drop=True)
