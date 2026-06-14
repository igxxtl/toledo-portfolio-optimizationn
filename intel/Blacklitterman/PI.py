"""Prior π de equilíbrio de mercado — script standalone."""
from __future__ import annotations

import sys

try:
    from .config import ATIVOS, PRIOR_END_DATE, PRIOR_START_DATE, RISK_FREE_ANNUAL, TAU
    from .prior import (
        compute_pi_and_prior_uncertainty,
        get_daily_returns_for_prior,
        get_market_cap_weights,
    )
except ImportError:
    from config import ATIVOS, PRIOR_END_DATE, PRIOR_START_DATE, RISK_FREE_ANNUAL, TAU
    from prior import (
        compute_pi_and_prior_uncertainty,
        get_daily_returns_for_prior,
        get_market_cap_weights,
    )


def main() -> None:
    print("Calculando pesos de mercado...")
    market_weights = get_market_cap_weights(ATIVOS)

    print("Calculando retornos diários...")
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

    print("\nPrior π (retornos implícitos de equilíbrio):")
    print(pi.round(6))
    print(f"\nDelta (aversão ao risco): {delta:.6f}")
    print(f"Sigma shape: {sigma.shape}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
