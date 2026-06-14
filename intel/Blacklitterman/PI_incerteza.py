"""
Componente de incerteza do prior PI no Black-Litterman.

Este script calcula:
1) Sigma (covariancia dos retornos)
2) PI (retornos implicitos de equilibrio): PI = delta * Sigma * w_mkt
3) Incerteza do prior: C_prior = tau * Sigma

Saidas:
- pi_equilibrio.csv
- sigma_cov.csv
- prior_cov_tau_sigma.csv
- prior_uncertainty_diag.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
TICKERS_DIR = REPO_ROOT / "tickers_data"

ATIVOS = ["VALE3.SA", "BBAS3.SA", "ITUB3.SA", "BBDC3.SA", "ABEV3.SA"]
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

RISK_FREE_ANNUAL = 0.15
TAU = 0.025
VWAP_WINDOW = 7
B3_SESSION_HOURS = set(range(13, 20))

OUT_PI = PROJECT_DIR / "pi_equilibrio.csv"
OUT_SIGMA = PROJECT_DIR / "sigma_cov.csv"
OUT_PRIOR_COV = PROJECT_DIR / "prior_cov_tau_sigma.csv"
OUT_PRIOR_DIAG = PROJECT_DIR / "prior_uncertainty_diag.csv"


def get_market_cap_weights(tickers: list[str]) -> pd.Series:
    caps: dict[str, float] = {}
    for t in tickers:
        info = yf.Ticker(t).info
        market_cap = info.get("marketCap")
        if market_cap is None or not np.isfinite(market_cap) or market_cap <= 0:
            raise ValueError(f"marketCap invalido para {t}.")
        caps[t] = float(market_cap)

    cap_series = pd.Series(caps, dtype="float64")
    weights = cap_series / cap_series.sum()
    return weights


def _sa_to_base_ticker(ticker: str) -> str:
    return ticker[:-3] if ticker.endswith(".SA") else ticker


def _load_hourly_series_for_ticker(base_ticker: str) -> pd.DataFrame:
    file_path = TICKERS_DIR / f"BMFBOVESPA_DLY_{base_ticker}, 60.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"CSV nao encontrado para {base_ticker}: {file_path}")

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
    out = out.sort_values("time").reset_index(drop=True)
    return out.drop(columns=["weekday", "hour"])


def _daily_vwap_7h_returns(base_ticker: str, start_date: str, end_date: str) -> pd.Series:
    df = _load_hourly_series_for_ticker(base_ticker)
    df = _filter_b3_session(df)

    df["pv"] = df["close"] * df["Volume"]
    df["vwap"] = df["pv"].rolling(window=VWAP_WINDOW).sum() / df["Volume"].rolling(window=VWAP_WINDOW).sum()
    df["vwap"] = df["vwap"].fillna(df["close"])
    df["log_ret_vwap"] = np.log(df["vwap"] / df["vwap"].shift(1))
    df = df.dropna(subset=["log_ret_vwap"]).copy()
    df["view_date"] = df["time"].dt.normalize()

    grouped = df.groupby("view_date", sort=True)
    rows: list[tuple[pd.Timestamp, float]] = []
    for day, g in grouped:
        if len(g) != VWAP_WINDOW:
            continue
        if set(g["time"].dt.hour.tolist()) != B3_SESSION_HOURS:
            continue
        rows.append((pd.Timestamp(day), float(g["log_ret_vwap"].sum())))

    if not rows:
        raise ValueError(f"Sem retornos VWAP diarios validos para {base_ticker}.")

    out = pd.Series({d: r for d, r in rows}, name=f"{base_ticker}.SA").sort_index()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    out = out[(out.index >= start_ts) & (out.index <= end_ts)]
    return out


def get_vwap_returns_dataframe(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    for ticker in tickers:
        base = _sa_to_base_ticker(ticker)
        series_list.append(_daily_vwap_7h_returns(base, start_date, end_date))
    returns = pd.concat(series_list, axis=1).dropna(how="any")
    if returns.empty:
        raise ValueError("Sem retornos VWAP validos para calcular Sigma.")
    returns.index.name = "view_date"
    return returns


def compute_pi_and_prior_uncertainty(
    returns: pd.DataFrame,
    market_weights: pd.Series,
    risk_free_annual: float,
    tau: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, float]:
    sigma = returns.cov()

    risk_free_daily = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
    market_returns = returns @ market_weights
    delta = (market_returns.mean() - risk_free_daily) / market_returns.var()

    pi = delta * (sigma @ market_weights)
    pi.name = "pi"

    prior_cov = tau * sigma
    prior_diag = pd.Series(np.diag(prior_cov.values), index=prior_cov.index, name="tau_sigma_ii")

    return pi, sigma, prior_cov, prior_diag, float(delta)


def main() -> None:
    print("Calculando pesos de mercado...")
    market_weights = get_market_cap_weights(ATIVOS)

    print("Calculando retornos VWAP diarios (7h) em tickers_data...")
    returns = get_vwap_returns_dataframe(ATIVOS, START_DATE, END_DATE)

    # Alinha ordem/colunas para garantir multiplicacoes corretas.
    common = [c for c in returns.columns if c in market_weights.index]
    returns = returns[common].copy()
    market_weights = market_weights[common].copy()

    print("Calculando PI e incerteza do prior (tau*Sigma)...")
    pi, sigma, prior_cov, prior_diag, delta = compute_pi_and_prior_uncertainty(
        returns=returns,
        market_weights=market_weights,
        risk_free_annual=RISK_FREE_ANNUAL,
        tau=TAU,
    )

    pi_df = pd.DataFrame(
        {
            "ticker": pi.index,
            "pi": pi.values,
            "peso_mercado": market_weights.reindex(pi.index).values,
            "tau_sigma_ii": prior_diag.reindex(pi.index).values,
        }
    )
    pi_df.to_csv(OUT_PI, index=False)
    sigma.to_csv(OUT_SIGMA)
    prior_cov.to_csv(OUT_PRIOR_COV)
    prior_diag.to_frame().reset_index().rename(columns={"index": "ticker"}).to_csv(OUT_PRIOR_DIAG, index=False)

    print(f"Delta (aversao ao risco) = {delta:.6f}")
    print(f"PI salvo em: {OUT_PI}")
    print(f"Sigma salvo em: {OUT_SIGMA}")
    print(f"Tau*Sigma salvo em: {OUT_PRIOR_COV}")
    print(f"Diag(Tau*Sigma) salvo em: {OUT_PRIOR_DIAG}")


if __name__ == "__main__":
    main()
