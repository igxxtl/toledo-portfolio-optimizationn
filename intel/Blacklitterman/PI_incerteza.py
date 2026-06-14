"""
Incerteza do prior PI com retornos VWAP intradiários (7h) — script standalone.

Diferente do pipeline principal (``pipeline.py``), que usa retornos diários de
``dados_diarios/``, este script calcula Sigma a partir de ``tickers_data/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import (
        OUT_PI_STANDALONE,
        OUT_PRIOR_COV_STANDALONE,
        OUT_PRIOR_DIAG_STANDALONE,
        OUT_SIGMA_STANDALONE,
        PRIOR_END_DATE,
        PRIOR_START_DATE,
        REPO_ROOT,
        RISK_FREE_ANNUAL,
        TAU,
    )
    from .prior import compute_pi_and_prior_uncertainty, get_market_cap_weights
except ImportError:
    from config import (
        OUT_PI_STANDALONE,
        OUT_PRIOR_COV_STANDALONE,
        OUT_PRIOR_DIAG_STANDALONE,
        OUT_SIGMA_STANDALONE,
        PRIOR_END_DATE,
        PRIOR_START_DATE,
        REPO_ROOT,
        RISK_FREE_ANNUAL,
        TAU,
    )
    from prior import compute_pi_and_prior_uncertainty, get_market_cap_weights

TICKERS_DIR = REPO_ROOT / "tickers_data"
ATIVOS = ["VALE3.SA", "BBAS3.SA", "ITUB3.SA", "BBDC3.SA", "ABEV3.SA"]
VWAP_WINDOW = 7
B3_SESSION_HOURS = set(range(13, 20))


def _sa_to_base_ticker(ticker: str) -> str:
    return ticker[:-3] if ticker.endswith(".SA") else ticker


def _load_hourly_series(base_ticker: str) -> pd.DataFrame:
    file_path = TICKERS_DIR / f"BMFBOVESPA_DLY_{base_ticker}, 60.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"CSV não encontrado para {base_ticker}: {file_path}")

    df = pd.read_csv(file_path, usecols=["time", "close", "Volume"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("float32")
    return df.dropna(subset=["time", "close", "Volume"]).sort_values("time").reset_index(drop=True)


def _filter_b3_session(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weekday"] = out["time"].dt.weekday
    out["hour"] = out["time"].dt.hour
    out = out[(out["weekday"] <= 4) & (out["hour"].isin(B3_SESSION_HOURS))]
    return out.sort_values("time").reset_index(drop=True).drop(columns=["weekday", "hour"])


def _daily_vwap_7h_returns(base_ticker: str, start_date: str, end_date: str) -> pd.Series:
    df = _filter_b3_session(_load_hourly_series(base_ticker))
    df["pv"] = df["close"] * df["Volume"]
    df["vwap"] = df["pv"].rolling(window=VWAP_WINDOW).sum() / df["Volume"].rolling(window=VWAP_WINDOW).sum()
    df["vwap"] = df["vwap"].fillna(df["close"])
    df["log_ret_vwap"] = np.log(df["vwap"] / df["vwap"].shift(1))
    df = df.dropna(subset=["log_ret_vwap"]).copy()
    df["view_date"] = df["time"].dt.normalize()

    rows: list[tuple[pd.Timestamp, float]] = []
    for day, g in df.groupby("view_date", sort=True):
        if len(g) != VWAP_WINDOW:
            continue
        if set(g["time"].dt.hour.tolist()) != B3_SESSION_HOURS:
            continue
        rows.append((pd.Timestamp(day), float(g["log_ret_vwap"].sum())))

    if not rows:
        raise ValueError(f"Sem retornos VWAP diários válidos para {base_ticker}.")

    out = pd.Series({d: r for d, r in rows}, name=f"{base_ticker}.SA").sort_index()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return out[(out.index >= start_ts) & (out.index <= end_ts)]


def get_vwap_returns_dataframe(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    series_list = [_daily_vwap_7h_returns(_sa_to_base_ticker(t), start_date, end_date) for t in tickers]
    returns = pd.concat(series_list, axis=1).dropna(how="any")
    if returns.empty:
        raise ValueError("Sem retornos VWAP válidos para calcular Sigma.")
    returns.index.name = "view_date"
    return returns


def main() -> None:
    print("Calculando pesos de mercado...")
    market_weights = get_market_cap_weights(ATIVOS)

    print("Calculando retornos VWAP diários (7h) em tickers_data...")
    returns = get_vwap_returns_dataframe(ATIVOS, PRIOR_START_DATE, PRIOR_END_DATE)

    common = [c for c in returns.columns if c in market_weights.index]
    returns = returns[common].copy()
    market_weights = market_weights[common].copy()

    print("Calculando PI e incerteza do prior (tau*Sigma)...")
    pi, sigma, prior_cov, delta = compute_pi_and_prior_uncertainty(
        returns=returns,
        market_weights=market_weights,
        risk_free_annual=RISK_FREE_ANNUAL,
        tau=TAU,
    )
    prior_diag = pd.Series(np.diag(prior_cov.values), index=prior_cov.index, name="tau_sigma_ii")

    pi_df = pd.DataFrame(
        {
            "ticker": pi.index,
            "pi": pi.values,
            "peso_mercado": market_weights.reindex(pi.index).values,
            "tau_sigma_ii": prior_diag.reindex(pi.index).values,
        }
    )
    pi_df.to_csv(OUT_PI_STANDALONE, index=False)
    sigma.to_csv(OUT_SIGMA_STANDALONE)
    prior_cov.to_csv(OUT_PRIOR_COV_STANDALONE)
    prior_diag.to_frame().reset_index().rename(columns={"index": "ticker"}).to_csv(OUT_PRIOR_DIAG_STANDALONE, index=False)

    print(f"Delta (aversão ao risco) = {delta:.6f}")
    print(f"PI salvo em: {OUT_PI_STANDALONE}")
    print(f"Sigma salvo em: {OUT_SIGMA_STANDALONE}")
    print(f"Tau*Sigma salvo em: {OUT_PRIOR_COV_STANDALONE}")
    print(f"Diag(Tau*Sigma) salvo em: {OUT_PRIOR_DIAG_STANDALONE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
