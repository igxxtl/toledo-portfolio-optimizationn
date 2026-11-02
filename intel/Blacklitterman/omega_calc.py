"""Cálculo da matriz de incerteza Omega (Black-Litterman)."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .config import (
        CONFIDENCE_CAP,
        CONFIDENCE_FLOOR,
        DEFAULT_VIEW_HORIZON,
        EPS,
        ERR_MIN_PERIODS,
        ERR_WINDOW,
        HARD_PRIOR_WHEN_LOW_NEWS,
        NEWS_COUNT_MIN,
        NEWS_OMEGA_BOOST,
        TAU,
        W_MODEL_CONF,
        W_NEWS_CONF,
    )
    from .xgb_views import load_ticker_horizons, normalize_return_columns, resolve_view_horizon
except ImportError:
    from config import (
        CONFIDENCE_CAP,
        CONFIDENCE_FLOOR,
        DEFAULT_VIEW_HORIZON,
        EPS,
        ERR_MIN_PERIODS,
        ERR_WINDOW,
        HARD_PRIOR_WHEN_LOW_NEWS,
        NEWS_COUNT_MIN,
        NEWS_OMEGA_BOOST,
        TAU,
        W_MODEL_CONF,
        W_NEWS_CONF,
    )
    from xgb_views import load_ticker_horizons, normalize_return_columns, resolve_view_horizon

__all__ = ["calculate_omega"]


def _safe_sample_var(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    var_value = float(clean.var(ddof=1))
    if not np.isfinite(var_value) or var_value <= 0:
        return np.nan
    return var_value


def _compute_confidence_per_ticker(
    group: pd.DataFrame,
    *,
    err_window: int = ERR_WINDOW,
    err_min_periods: int = ERR_MIN_PERIODS,
    w_model_conf: float = W_MODEL_CONF,
    w_news_conf: float = W_NEWS_CONF,
    confidence_floor: float = CONFIDENCE_FLOOR,
    confidence_cap: float = CONFIDENCE_CAP,
) -> pd.DataFrame:
    err_window = max(int(err_window), 2)
    err_min_periods = min(max(int(err_min_periods), 2), err_window)

    out = normalize_return_columns(group).sort_values("view_date").copy()
    out["pred_abs_error"] = (out["ret_pred"] - out["ret_real"]).abs()

    out["error_scale"] = (
        out["pred_abs_error"].rolling(err_window, min_periods=err_min_periods).median().shift(1)
    )
    fallback = float(out["pred_abs_error"].median())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 0.005
    out["error_scale"] = out["error_scale"].fillna(fallback).clip(lower=EPS)

    out["model_confidence"] = np.exp(-out["pred_abs_error"] / out["error_scale"])
    news_count = pd.to_numeric(out.get("news_count", 0), errors="coerce").fillna(0.0)
    out["news_count"] = news_count.astype(int)
    out["news_confidence"] = 1.0 - np.exp(-out["news_count"] / 3.0)
    out["view_confidence"] = (
        w_model_conf * out["model_confidence"] + w_news_conf * out["news_confidence"]
    ).clip(lower=confidence_floor, upper=confidence_cap)
    return out


def calculate_omega(
    q_long: pd.DataFrame,
    tau: float = TAU,
    *,
    err_window: int = ERR_WINDOW,
    err_min_periods: int = ERR_MIN_PERIODS,
    w_model_conf: float = W_MODEL_CONF,
    w_news_conf: float = W_NEWS_CONF,
    confidence_floor: float = CONFIDENCE_FLOOR,
    confidence_cap: float = CONFIDENCE_CAP,
    news_count_min: int = NEWS_COUNT_MIN,
    news_omega_boost: float = NEWS_OMEGA_BOOST,
    hard_prior_when_low_news: bool = HARD_PRIOR_WHEN_LOW_NEWS,
) -> pd.DataFrame:
    """Calcula Ω na mesma escala temporal de Q.

    ``ret_pred``/``ret_real`` permanecem no horizonte do XGB (confiança).
    ``sigma_ii`` / ``omega`` são convertidos para diário via ÷h² quando há ``horizon``.
    Cobertura baixa (``news_count < news_count_min``) multiplica Ω por ``news_omega_boost``.
    Com ``hard_prior_when_low_news``, confiança vai a EPS nesses dias (view ignorada).
    """
    q_long = normalize_return_columns(q_long).copy()
    horizons_map = load_ticker_horizons()
    if "horizon" not in q_long.columns:
        q_long["horizon"] = q_long["ticker"].map(
            lambda t: resolve_view_horizon(str(t), horizons_map)
        )
    q_long["horizon"] = (
        pd.to_numeric(q_long["horizon"], errors="coerce")
        .fillna(DEFAULT_VIEW_HORIZON)
        .clip(lower=1.0)
        .astype(int)
    )
    if "news_count" not in q_long.columns:
        q_long["news_count"] = 0
    q_long["news_count"] = (
        pd.to_numeric(q_long["news_count"], errors="coerce").fillna(0).astype(int)
    )
    k = max(int(news_count_min), 0)
    boost = max(float(news_omega_boost), 1.0)
    if "news_coverage" not in q_long.columns:
        q_long["news_coverage"] = ((q_long["news_count"] > 0).astype(float) if k == 0
                                   else (q_long["news_count"].astype(float) / float(k)).clip(upper=1.0))

    sigma_by_ticker: dict[str, float] = {}
    horizon_by_ticker: dict[str, int] = {}
    for ticker, group in q_long.groupby("ticker", sort=True):
        h = int(group["horizon"].iloc[0])
        horizon_by_ticker[str(ticker)] = h
        sigma_ii = _safe_sample_var(group["ret_real"])
        if not np.isfinite(sigma_ii):
            sigma_ii = _safe_sample_var(group["ret_pred"])
        if not np.isfinite(sigma_ii):
            sigma_ii = 2.5e-5
        # var(horizonte) → var(diária) ≈ var_h / h²
        sigma_by_ticker[str(ticker)] = float(sigma_ii) / float(h * h)

    parts = [
        _compute_confidence_per_ticker(
            g,
            err_window=err_window,
            err_min_periods=err_min_periods,
            w_model_conf=w_model_conf,
            w_news_conf=w_news_conf,
            confidence_floor=confidence_floor,
            confidence_cap=confidence_cap,
        )
        for _, g in q_long.groupby("ticker", sort=True)
    ]
    with_conf = pd.concat(parts, axis=0, ignore_index=True)
    if "horizon" not in with_conf.columns:
        with_conf["horizon"] = with_conf["ticker"].map(horizon_by_ticker)

    # Reatacha coverage / news_count do q_long (groupby pode ter reorderado)
    cov_cols = q_long[["view_date", "ticker", "news_count", "news_coverage"]].drop_duplicates(
        ["view_date", "ticker"]
    )
    with_conf = with_conf.drop(
        columns=["news_count", "news_coverage"], errors="ignore"
    ).merge(cov_cols, on=["view_date", "ticker"], how="left")
    with_conf["news_count"] = (
        pd.to_numeric(with_conf["news_count"], errors="coerce").fillna(0).astype(int)
    )
    with_conf["news_coverage"] = pd.to_numeric(
        with_conf["news_coverage"], errors="coerce"
    ).fillna(0.0)

    low_news = with_conf["news_count"] < k
    with_conf["omega_boost"] = np.where(low_news, boost, 1.0)

    if hard_prior_when_low_news:
        with_conf.loc[low_news, "view_confidence"] = EPS

    with_conf["sigma_ii"] = with_conf["ticker"].map(sigma_by_ticker)
    with_conf["omega_base"] = tau * with_conf["sigma_ii"]
    with_conf["omega"] = (
        with_conf["omega_base"]
        * (1.0 - with_conf["view_confidence"])
        / with_conf["view_confidence"]
    )
    with_conf["omega"] = with_conf["omega"] * with_conf["omega_boost"]
    with_conf["omega"] = with_conf["omega"].clip(lower=EPS)
    with_conf["tau"] = tau

    return with_conf[
        [
            "view_date",
            "ticker",
            "omega",
            "omega_base",
            "omega_boost",
            "view_confidence",
            "model_confidence",
            "news_confidence",
            "news_count",
            "news_coverage",
            "pred_abs_error",
            "error_scale",
            "sigma_ii",
            "tau",
            "horizon",
        ]
    ].sort_values(["view_date", "ticker"]).reset_index(drop=True)
