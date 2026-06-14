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


def run_pipeline() -> None:
    view_tickers = [ticker_from_sa(t) for t in ATIVOS]

    print("1) Gerando Q diário (XGBoost) para todos os ativos...")
    q_long = build_q_for_tickers(
        view_tickers,
        alpha=ALPHA_Q,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
    )
    for ticker in view_tickers:
        n = len(q_long[q_long["ticker"] == ticker])
        if n:
            print(f"  - {ticker}: OK ({n} dias)")

    print("2) Gerando Omega diário...")
    omega_long = calculate_omega(q_long, tau=TAU)
    q_omega_long = (
        q_long.merge(
            omega_long[["view_date", "ticker", "omega", "omega_base", "view_confidence"]],
            on=["view_date", "ticker"],
            how="inner",
        )
        .sort_values(["view_date", "ticker"])
        .reset_index(drop=True)
    )

    print("3) Gerando PI e incerteza do prior (tau*Sigma) com retornos diários...")
    market_weights = get_market_cap_weights(ATIVOS)
    returns = get_daily_returns_for_prior(
        tickers=ATIVOS,
        start_date=PRIOR_START_DATE,
        end_date=PRIOR_END_DATE,
    )
    common = [c for c in returns.columns if c in market_weights.index]
    returns = returns[common].copy()
    market_weights = market_weights[common].copy()
    pi, sigma, prior_cov, delta = compute_pi_and_prior_uncertainty(
        returns=returns,
        market_weights=market_weights,
        risk_free_annual=RISK_FREE_ANNUAL,
        tau=TAU,
    )

    print("4) Calculando posterior diário do Black-Litterman...")
    posterior_mu, posterior_w = compute_daily_bl_posterior(
        pi=pi,
        sigma=sigma,
        tau=TAU,
        q_omega_long=q_omega_long,
        delta=delta,
        market_weights=market_weights,
        returns_history=returns,
        use_rolling_prior=USE_ROLLING_PRIOR,
        rolling_min_obs=ROLLING_MIN_OBS,
        risk_free_annual=RISK_FREE_ANNUAL,
    )
    posterior_w_controlled = apply_weight_controls(
        posterior_w=posterior_w,
        long_only=LONG_ONLY,
        max_weight_per_asset=MAX_WEIGHT_PER_ASSET,
    )
    posterior_w_ctrl_daily = build_rebalanced_weights(posterior_w_controlled, mode="daily")
    posterior_w_ctrl_weekly = build_rebalanced_weights(posterior_w_controlled, mode="weekly")
    posterior_w_ctrl_monthly = build_rebalanced_weights(posterior_w_controlled, mode="monthly")

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

    print(f"Q diário: {OUT_Q_LONG}")
    print(f"Omega diário: {OUT_OMEGA_LONG}")
    print(f"Posterior mu: {OUT_POSTERIOR_MU}")
    print(f"Pesos controlados daily: {OUT_POSTERIOR_W_CTRL_DAILY}")
    print(f"Views processadas: {len(q_omega_long)} | Dias no posterior: {len(posterior_mu)}")
    print(f"Delta: {delta:.6f}")


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro no pipeline híbrido: {exc}")
        sys.exit(1)
