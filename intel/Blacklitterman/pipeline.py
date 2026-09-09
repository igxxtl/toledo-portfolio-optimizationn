"""
Pipeline Black-Litterman híbrido (prior de mercado + views XGBoost/sentimento).

Saídas principais em ``intel/Blacklitterman/outputs/``:
- bl_hibrido_q_long.csv
- bl_hibrido_omega_long.csv
- bl_hibrido_posterior_mu.csv
- bl_hibrido_posterior_weights_controlled_*.csv
"""
from __future__ import annotations

import sys
from typing import Any

import pandas as pd

try:
    from .config import (
        ALPHA_Q,
        ATIVOS,
        LONG_ONLY,
        MAX_WEIGHT_PER_ASSET,
        OUT_OMEGA_LONG,
        OUT_PI,
        OUT_POSTERIOR_MU,
        OUT_POSTERIOR_W,
        OUT_POSTERIOR_W_CONTROLLED,
        OUT_POSTERIOR_W_CTRL_DAILY,
        OUT_POSTERIOR_W_CTRL_MONTHLY,
        OUT_POSTERIOR_W_CTRL_WEEKLY,
        OUT_PRIOR_COV,
        OUT_Q_LONG,
        OUT_Q_OMEGA_LONG,
        OUT_SIGMA,
        PERIOD_END,
        PERIOD_START,
        PRIOR_END_DATE,
        PRIOR_START_DATE,
        RISK_FREE_ANNUAL,
        ROLLING_MIN_OBS,
        TAU,
        USE_ROLLING_PRIOR,
    )
    from .io_utils import save_csv_with_date
    from .omega_calc import calculate_omega
    from .params import BLParams
    from .prior import (
        apply_weight_controls,
        build_rebalanced_weights,
        compute_daily_bl_posterior,
        compute_pi_and_prior_uncertainty,
        get_daily_returns_for_prior,
        get_market_cap_weights,
        ticker_from_sa,
    )
    from .xgb_views import build_q_for_tickers
except ImportError:
    from config import (
        ALPHA_Q,
        ATIVOS,
        LONG_ONLY,
        MAX_WEIGHT_PER_ASSET,
        OUT_OMEGA_LONG,
        OUT_PI,
        OUT_POSTERIOR_MU,
        OUT_POSTERIOR_W,
        OUT_POSTERIOR_W_CONTROLLED,
        OUT_POSTERIOR_W_CTRL_DAILY,
        OUT_POSTERIOR_W_CTRL_MONTHLY,
        OUT_POSTERIOR_W_CTRL_WEEKLY,
        OUT_PRIOR_COV,
        OUT_Q_LONG,
        OUT_Q_OMEGA_LONG,
        OUT_SIGMA,
        PERIOD_END,
        PERIOD_START,
        PRIOR_END_DATE,
        PRIOR_START_DATE,
        RISK_FREE_ANNUAL,
        ROLLING_MIN_OBS,
        TAU,
        USE_ROLLING_PRIOR,
    )
    from io_utils import save_csv_with_date
    from omega_calc import calculate_omega
    from params import BLParams
    from prior import (
        apply_weight_controls,
        build_rebalanced_weights,
        compute_daily_bl_posterior,
        compute_pi_and_prior_uncertainty,
        get_daily_returns_for_prior,
        get_market_cap_weights,
        ticker_from_sa,
    )
    from xgb_views import build_q_for_tickers


def run_pipeline(
    params: BLParams | None = None,
    *,
    period_start: str = PERIOD_START,
    period_end: str = PERIOD_END,
    q_long: pd.DataFrame | None = None,
    market_weights: pd.Series | None = None,
    returns_history: pd.DataFrame | None = None,
    save_outputs: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Executa o pipeline BL e retorna artefatos em memória."""
    p = (params or BLParams.from_config()).normalized()
    view_tickers = [ticker_from_sa(t) for t in ATIVOS]

    if q_long is None:
        if verbose:
            print("1) Gerando Q diário (XGBoost) para todos os ativos...")
        q_long = build_q_for_tickers(
            view_tickers,
            alpha=p.alpha_q,
            period_start=period_start,
            period_end=period_end,
        )
        if verbose:
            for ticker in view_tickers:
                n = len(q_long[q_long["ticker"] == ticker])
                if n:
                    print(f"  - {ticker}: OK ({n} dias)")
    elif verbose:
        print("1) Usando Q pré-calculado...")

    if verbose:
        print("2) Gerando Omega diário...")
    omega_long = calculate_omega(
        q_long,
        tau=p.tau,
        err_window=p.err_window,
        err_min_periods=p.err_min_periods,
        w_model_conf=p.w_model_conf,
        w_news_conf=p.w_news_conf,
        confidence_floor=p.confidence_floor,
        confidence_cap=p.confidence_cap,
    )
    q_omega_long = (
        q_long.merge(
            omega_long[["view_date", "ticker", "omega", "omega_base", "view_confidence"]],
            on=["view_date", "ticker"],
            how="inner",
        )
        .sort_values(["view_date", "ticker"])
        .reset_index(drop=True)
    )

    if verbose:
        print("3) Gerando PI e incerteza do prior (tau*Sigma) com retornos diários...")
    if market_weights is None:
        market_weights = get_market_cap_weights(ATIVOS)
    if returns_history is None:
        returns = get_daily_returns_for_prior(
            tickers=ATIVOS,
            start_date=PRIOR_START_DATE,
            end_date=PRIOR_END_DATE,
        )
    else:
        returns = returns_history.copy()
    common = [c for c in returns.columns if c in market_weights.index]
    returns = returns[common].copy()
    market_weights = market_weights[common].copy()
    pi, sigma, prior_cov, delta = compute_pi_and_prior_uncertainty(
        returns=returns,
        market_weights=market_weights,
        risk_free_annual=RISK_FREE_ANNUAL,
        tau=p.tau,
    )

    if verbose:
        print("4) Calculando posterior diário do Black-Litterman...")
    posterior_mu, posterior_w = compute_daily_bl_posterior(
        pi=pi,
        sigma=sigma,
        tau=p.tau,
        q_omega_long=q_omega_long,
        delta=delta,
        market_weights=market_weights,
        returns_history=returns,
        use_rolling_prior=p.use_rolling_prior,
        rolling_min_obs=p.rolling_min_obs,
        risk_free_annual=RISK_FREE_ANNUAL,
    )
    posterior_w_controlled = apply_weight_controls(
        posterior_w=posterior_w,
        long_only=p.long_only,
        max_weight_per_asset=p.max_weight_per_asset,
    )
    posterior_w_ctrl_daily = build_rebalanced_weights(posterior_w_controlled, mode="daily")
    posterior_w_ctrl_weekly = build_rebalanced_weights(posterior_w_controlled, mode="weekly")
    posterior_w_ctrl_monthly = build_rebalanced_weights(posterior_w_controlled, mode="monthly")

    weights_by_mode = {
        "daily": posterior_w_ctrl_daily,
        "weekly": posterior_w_ctrl_weekly,
        "monthly": posterior_w_ctrl_monthly,
    }

    if save_outputs:
        if verbose:
            print("5) Salvando resultados...")
        save_csv_with_date(q_long, OUT_Q_LONG)
        save_csv_with_date(omega_long, OUT_OMEGA_LONG)
        save_csv_with_date(q_omega_long, OUT_Q_OMEGA_LONG)
        pi.to_frame().reset_index().rename(columns={"index": "ticker"}).to_csv(OUT_PI, index=False)
        sigma.to_csv(OUT_SIGMA)
        prior_cov.to_csv(OUT_PRIOR_COV)
        posterior_mu.to_csv(OUT_POSTERIOR_MU, index=False)
        posterior_w.to_csv(OUT_POSTERIOR_W, index=False)
        posterior_w_controlled.to_csv(OUT_POSTERIOR_W_CONTROLLED, index=False)
        posterior_w_ctrl_daily.to_csv(OUT_POSTERIOR_W_CTRL_DAILY, index=False)
        posterior_w_ctrl_weekly.to_csv(OUT_POSTERIOR_W_CTRL_WEEKLY, index=False)
        posterior_w_ctrl_monthly.to_csv(OUT_POSTERIOR_W_CTRL_MONTHLY, index=False)

        if verbose:
            print(f"Q diário: {OUT_Q_LONG}")
            print(f"Omega diário: {OUT_OMEGA_LONG}")
            print(f"Posterior mu: {OUT_POSTERIOR_MU}")
            print(f"Pesos controlados daily: {OUT_POSTERIOR_W_CTRL_DAILY}")
            print(f"Views processadas: {len(q_omega_long)} | Dias no posterior: {len(posterior_mu)}")
            print(f"Delta: {delta:.6f}")

    return {
        "params": p,
        "delta": delta,
        "q_long": q_long,
        "omega_long": omega_long,
        "q_omega_long": q_omega_long,
        "pi": pi,
        "sigma": sigma,
        "prior_cov": prior_cov,
        "posterior_mu": posterior_mu,
        "posterior_w": posterior_w,
        "posterior_w_controlled": posterior_w_controlled,
        "weights_by_mode": weights_by_mode,
        "returns_history": returns,
        "market_weights": market_weights,
    }


def main() -> None:
    run_pipeline(params=BLParams.from_config(), save_outputs=True, verbose=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro no pipeline híbrido: {exc}")
        sys.exit(1)
