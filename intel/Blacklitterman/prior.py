"""Prior de mercado (PI), posterior Black-Litterman e controles de peso."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from .config import DATA_DAILY_DIR, EPS, RISK_FREE_ANNUAL
    from .weights import long_only_capped_weights
except ImportError:
    from config import DATA_DAILY_DIR, EPS, RISK_FREE_ANNUAL
    from weights import long_only_capped_weights

__all__ = [
    "apply_weight_controls",
    "build_rebalanced_weights",
    "compute_daily_bl_posterior",
    "compute_pi_and_prior_uncertainty",
    "get_daily_returns_for_prior",
    "get_market_cap_weights",
    "ticker_from_sa",
    "ticker_to_sa",
]


def ticker_to_sa(ticker: str) -> str:
    return ticker if ticker.endswith(".SA") else f"{ticker}.SA"


def ticker_from_sa(ticker: str) -> str:
    return ticker[:-3] if ticker.endswith(".SA") else ticker


def get_market_cap_weights(tickers: list[str]) -> pd.Series:
    caps: dict[str, float] = {}
    for t in tickers:
        info = yf.Ticker(t).info
        cap = info.get("marketCap")
        if cap is None or not np.isfinite(cap) or cap <= 0:
            raise ValueError(f"marketCap inválido para {t}")
        caps[t] = float(cap)
    cap_series = pd.Series(caps, dtype="float64")
    return cap_series / cap_series.sum()


def get_daily_returns_for_prior(
    tickers: list[str],
    data_dir: Path | None = None,
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """Retornos logarítmicos diários a partir de ``dados_diarios/``."""
    data_dir = data_dir or DATA_DAILY_DIR
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    series_list: list[pd.Series] = []

    for ticker_sa in tickers:
        base = ticker_from_sa(ticker_sa)
        fp = data_dir / f"{base}.csv"
        if not fp.exists():
            raise FileNotFoundError(f"Dados diários não encontrados: {fp}")

        df = pd.read_csv(fp, usecols=["date", "close"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        close = pd.to_numeric(df["close"], errors="coerce")
        log_ret = np.log(close / close.shift(1))
        log_ret.index = df["date"].values
        log_ret = log_ret.dropna()
        log_ret.name = ticker_sa
        log_ret = log_ret[(log_ret.index >= start_ts) & (log_ret.index <= end_ts)]
        if log_ret.empty:
            raise ValueError(f"Sem retornos diários para {ticker_sa} no período.")
        series_list.append(log_ret)

    returns = pd.concat(series_list, axis=1).dropna(how="any")
    if returns.empty:
        raise ValueError("Sem retornos diários válidos para calcular Sigma do prior.")
    returns.index.name = "view_date"
    return returns


def compute_pi_and_prior_uncertainty(
    returns: pd.DataFrame,
    market_weights: pd.Series,
    risk_free_annual: float,
    tau: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, float]:
    sigma = returns.cov()
    risk_free_daily = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
    market_returns = returns @ market_weights
    delta = (market_returns.mean() - risk_free_daily) / market_returns.var()
    pi = delta * (sigma @ market_weights)
    pi.name = "pi"
    prior_cov = tau * sigma
    return pi, sigma, prior_cov, float(delta)


def compute_daily_bl_posterior(
    pi: pd.Series,
    sigma: pd.DataFrame,
    tau: float,
    q_omega_long: pd.DataFrame,
    delta: float,
    market_weights: pd.Series | None = None,
    returns_history: pd.DataFrame | None = None,
    use_rolling_prior: bool = False,
    rolling_min_obs: int = 60,
    risk_free_annual: float = RISK_FREE_ANNUAL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(pi.index)
    n = len(tickers)
    static_tau_sigma = tau * sigma.values
    static_inv_tau_sigma = np.linalg.pinv(static_tau_sigma)
    static_pi_vec = pi.reindex(tickers).values.reshape(n, 1)
    static_delta = float(delta)

    mu_rows: list[dict[str, object]] = []
    w_rows: list[dict[str, object]] = []

    for view_date, day_df in q_omega_long.sort_values(["view_date", "ticker"]).groupby("view_date", sort=True):
        inv_tau_sigma = static_inv_tau_sigma
        pi_vec = static_pi_vec
        delta_used = static_delta
        sigma_used = sigma.values
        prior_obs = 0
        prior_end_date = ""

        if use_rolling_prior and returns_history is not None and market_weights is not None:
            vd = pd.Timestamp(view_date)
            hist = returns_history[returns_history.index < vd].copy()
            hist = hist[[c for c in tickers if c in hist.columns]].dropna(how="any")
            prior_obs = int(len(hist))
            if not hist.empty:
                prior_end_date = pd.Timestamp(hist.index.max()).strftime("%Y-%m-%d")

            if len(hist) >= rolling_min_obs:
                aligned_weights = market_weights.reindex(hist.columns)
                pi_roll, sigma_roll, _, delta_roll = compute_pi_and_prior_uncertainty(
                    returns=hist,
                    market_weights=aligned_weights,
                    risk_free_annual=risk_free_annual,
                    tau=tau,
                )
                sigma_roll = sigma_roll.reindex(index=tickers, columns=tickers)
                pi_roll = pi_roll.reindex(tickers)
                sigma_used = sigma_roll.values
                inv_tau_sigma = np.linalg.pinv(tau * sigma_used)
                pi_vec = pi_roll.values.reshape(n, 1)
                delta_used = float(delta_roll)

        valid_rows: list[tuple[str, float, float]] = []
        for _, row in day_df.iterrows():
            view_ticker_sa = ticker_to_sa(str(row["ticker"]))
            if view_ticker_sa not in tickers:
                continue
            omega = max(float(row["omega"]), EPS)
            valid_rows.append((view_ticker_sa, float(row["Q"]), omega))

        if not valid_rows:
            continue

        k = len(valid_rows)
        p = np.zeros((k, n), dtype=float)
        q_vec = np.zeros((k, 1), dtype=float)
        omega_diag = np.zeros(k, dtype=float)
        view_tickers_used: list[str] = []

        for i, (view_ticker_sa, q_val, omega_val) in enumerate(valid_rows):
            p[i, tickers.index(view_ticker_sa)] = 1.0
            q_vec[i, 0] = q_val
            omega_diag[i] = omega_val
            view_tickers_used.append(view_ticker_sa)

        inv_omega = np.diag(1.0 / omega_diag)
        a_mat = inv_tau_sigma + p.T @ inv_omega @ p
        b_vec = inv_tau_sigma @ pi_vec + p.T @ inv_omega @ q_vec
        mu_post = np.linalg.pinv(a_mat) @ b_vec

        w_post = np.linalg.pinv(delta_used * sigma_used) @ mu_post
        if np.isfinite(w_post).all() and abs(float(w_post.sum())) > EPS:
            w_post = w_post / float(w_post.sum())

        mu_entry: dict[str, object] = {"view_date": pd.Timestamp(view_date).strftime("%Y-%m-%d")}
        w_entry: dict[str, object] = {"view_date": pd.Timestamp(view_date).strftime("%Y-%m-%d")}
        for i, tk in enumerate(tickers):
            mu_entry[tk] = float(mu_post[i, 0])
            w_entry[tk] = float(w_post[i, 0])

        for entry in (mu_entry, w_entry):
            entry["views_count"] = k
            entry["view_tickers"] = ";".join(view_tickers_used)
            entry["Q_mean"] = float(np.mean(q_vec))
            entry["Omega_mean"] = float(np.mean(omega_diag))
            entry["prior_obs"] = int(prior_obs)
            entry["prior_end_date"] = prior_end_date

        mu_rows.append(mu_entry)
        w_rows.append(w_entry)

    return pd.DataFrame(mu_rows), pd.DataFrame(w_rows)


def apply_weight_controls(
    posterior_w: pd.DataFrame,
    long_only: bool,
    max_weight_per_asset: float,
) -> pd.DataFrame:
    out = posterior_w.copy()
    asset_cols = [c for c in out.columns if c.endswith(".SA")]
    if not asset_cols:
        return out

    raw_mat = out[asset_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    controlled_mat = np.zeros_like(raw_mat)

    for i in range(raw_mat.shape[0]):
        raw = raw_mat[i]
        if long_only:
            controlled_mat[i] = long_only_capped_weights(raw, max_weight_per_asset)
        else:
            total = float(raw.sum())
            controlled_mat[i] = raw / total if abs(total) > EPS else np.full_like(raw, 1.0 / len(raw))

    out[asset_cols] = controlled_mat
    out["gross_exposure"] = np.abs(controlled_mat).sum(axis=1)
    out["max_weight_used"] = np.max(controlled_mat, axis=1)
    return out


def build_rebalanced_weights(weights: pd.DataFrame, mode: str) -> pd.DataFrame:
    valid_modes = {"daily", "weekly", "monthly"}
    if mode not in valid_modes:
        raise ValueError(f"Modo inválido: {mode}. Use {sorted(valid_modes)}")

    out = weights.copy()
    out["view_date"] = pd.to_datetime(out["view_date"], errors="coerce")
    out = out.dropna(subset=["view_date"]).sort_values("view_date").reset_index(drop=True)
    asset_cols = [c for c in out.columns if c.endswith(".SA")]
    if not asset_cols:
        return out

    if mode == "daily":
        out["rebalance_mode"] = "daily"
        out["rebalance_flag"] = 1
        out["rebalance_source_date"] = out["view_date"].dt.strftime("%Y-%m-%d")
        return out

    period_key = (
        out["view_date"].dt.to_period("W-FRI").astype(str)
        if mode == "weekly"
        else out["view_date"].dt.to_period("M").astype(str)
    )

    first_idx = out.groupby(period_key, sort=False).head(1).index
    rebalance_mask = out.index.isin(first_idx)
    held = out[asset_cols].to_numpy(dtype=float).copy()
    source_dates = np.empty(len(out), dtype=object)
    current_weights: np.ndarray | None = None
    current_source_date = None

    for i in range(len(out)):
        if rebalance_mask[i] or current_weights is None:
            current_weights = held[i].copy()
            current_source_date = out.loc[i, "view_date"]
        else:
            held[i] = current_weights
        source_dates[i] = pd.Timestamp(current_source_date).strftime("%Y-%m-%d")

    out[asset_cols] = held
    out["rebalance_mode"] = mode
    out["rebalance_flag"] = rebalance_mask.astype(int)
    out["rebalance_source_date"] = source_dates
    out["gross_exposure"] = np.abs(held).sum(axis=1)
    out["max_weight_used"] = np.max(np.abs(held), axis=1)
    return out
