"""Avaliação de backtest do portfólio BL (retornos diários reais)."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .config import EXECUTION_LAG_DAYS, INITIAL_CAPITAL_BRL, SELIC_ANNUAL, TRADING_DAYS_YEAR
    from .io_utils import asset_columns
except ImportError:
    from config import EXECUTION_LAG_DAYS, INITIAL_CAPITAL_BRL, SELIC_ANNUAL, TRADING_DAYS_YEAR
    from io_utils import asset_columns

__all__ = ["build_gain_series", "portfolio_metrics"]


def build_gain_series(
    mu: pd.DataFrame,
    w: pd.DataFrame,
    daily_log_returns: pd.DataFrame,
    mode_label: str,
    *,
    initial_capital: float = INITIAL_CAPITAL_BRL,
    selic_annual: float = SELIC_ANNUAL,
    execution_lag_days: int = EXECUTION_LAG_DAYS,
) -> pd.DataFrame:
    assets = sorted(set(asset_columns(mu)).intersection(asset_columns(w)))
    if not assets:
        raise ValueError("Não encontrei colunas de ativos (.SA) em posterior_mu e posterior_weights.")

    merged = mu[["view_date", *assets]].merge(
        w[["view_date", *assets]],
        on="view_date",
        suffixes=("_mu", "_w"),
        how="inner",
    )

    for asset in assets:
        merged[f"{asset}_w_exec"] = merged[f"{asset}_w"].shift(execution_lag_days)
    merged = merged.dropna(subset=[f"{a}_w_exec" for a in assets]).reset_index(drop=True)

    est_parts = [merged[f"{a}_w_exec"] * np.expm1(merged[f"{a}_mu"]) for a in assets]
    merged["ret_est_portfolio"] = np.sum(np.column_stack(est_parts), axis=1)

    real_assets = [a for a in assets if a in daily_log_returns.columns]
    if real_assets:
        for asset in real_assets:
            merged[f"{asset}_real_log"] = merged["view_date"].map(daily_log_returns[asset])
        real_parts = [
            merged[f"{a}_w_exec"] * np.expm1(merged[f"{a}_real_log"]) for a in real_assets
        ]
        merged["ret_real_portfolio"] = np.sum(np.column_stack(real_parts), axis=1)
    else:
        merged["ret_real_portfolio"] = np.nan

    merged["capital_est"] = (1.0 + merged["ret_est_portfolio"]).cumprod()
    merged["ganho_est_acum_pct"] = (merged["capital_est"] - 1.0) * 100.0

    if merged["ret_real_portfolio"].notna().any():
        merged["capital_real"] = (1.0 + merged["ret_real_portfolio"].fillna(0.0)).cumprod()
        merged["ganho_real_acum_pct"] = (merged["capital_real"] - 1.0) * 100.0
    else:
        merged["capital_real"] = np.nan
        merged["ganho_real_acum_pct"] = np.nan

    selic_daily = (1.0 + selic_annual) ** (1.0 / 252.0) - 1.0
    merged["ret_selic_benchmark"] = selic_daily
    merged["capital_selic"] = (1.0 + merged["ret_selic_benchmark"]).cumprod()
    merged["ganho_selic_acum_pct"] = (merged["capital_selic"] - 1.0) * 100.0
    merged["capital_est_brl"] = initial_capital * merged["capital_est"]
    merged["capital_real_brl"] = initial_capital * merged["capital_real"]
    merged["capital_selic_brl"] = initial_capital * merged["capital_selic"]
    merged["mode"] = mode_label

    keep_cols = [
        "view_date",
        "ret_est_portfolio",
        "ret_real_portfolio",
        "ret_selic_benchmark",
        "capital_est",
        "capital_real",
        "capital_selic",
        "capital_est_brl",
        "capital_real_brl",
        "capital_selic_brl",
        "ganho_est_acum_pct",
        "ganho_real_acum_pct",
        "ganho_selic_acum_pct",
        "mode",
    ]
    return merged[keep_cols].sort_values("view_date").reset_index(drop=True)


def portfolio_metrics(
    gain: pd.DataFrame,
    *,
    eval_start: pd.Timestamp | None = None,
    eval_end: pd.Timestamp | None = None,
    trading_days_year: int = TRADING_DAYS_YEAR,
) -> dict[str, float]:
    """Métricas do portfólio realizado em um recorte opcional de datas."""
    df = gain.copy()
    df["view_date"] = pd.to_datetime(df["view_date"], errors="coerce")
    df = df.dropna(subset=["view_date"]).sort_values("view_date")
    if eval_start is not None:
        df = df[df["view_date"] >= pd.Timestamp(eval_start)]
    if eval_end is not None:
        df = df[df["view_date"] <= pd.Timestamp(eval_end)]
    if df.empty:
        return {
            "sharpe": float("nan"),
            "total_return_pct": float("nan"),
            "excess_vs_selic_pct": float("nan"),
            "max_drawdown": float("nan"),
            "final_capital_brl": float("nan"),
            "n_days": 0.0,
        }

    ret = pd.to_numeric(df["ret_real_portfolio"], errors="coerce").dropna()
    cap = pd.to_numeric(df["capital_real_brl"], errors="coerce").dropna()
    cap_selic = pd.to_numeric(df["capital_selic_brl"], errors="coerce").dropna()

    sharpe = float("nan")
    if len(ret) > 1 and float(ret.std(ddof=1)) > 0:
        sharpe = float(ret.mean() / ret.std(ddof=1) * np.sqrt(trading_days_year))

    mdd = float("nan")
    if not cap.empty:
        running_max = cap.cummax()
        mdd = float((cap / running_max - 1.0).min())

    total_return_pct = float((cap.iloc[-1] / cap.iloc[0] - 1.0) * 100.0) if len(cap) > 1 else float("nan")
    excess_vs_selic_pct = float("nan")
    if not cap.empty and not cap_selic.empty:
        excess_vs_selic_pct = float((cap.iloc[-1] / cap_selic.iloc[-1] - 1.0) * 100.0)

    return {
        "sharpe": sharpe,
        "total_return_pct": total_return_pct,
        "excess_vs_selic_pct": excess_vs_selic_pct,
        "max_drawdown": mdd,
        "final_capital_brl": float(cap.iloc[-1]) if not cap.empty else float("nan"),
        "n_days": float(len(df)),
    }
