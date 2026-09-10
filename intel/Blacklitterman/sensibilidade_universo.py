"""Sensibilidade de universo: full vs sem ativos fracos (ex.: VALE3).

Uso:
    cd intel/Blacklitterman
    python sensibilidade_universo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from . import config
    from .backtest import build_gain_series, portfolio_metrics
    from .config import (
        ATIVOS_SENSITIVITY_DROP,
        OPT_DEFAULT_REBALANCE_MODE,
        OUT_SCORE_SENSITIVITY,
        PERIOD_END,
        PERIOD_START,
    )
    from .params import BLParams
    from .pipeline import run_pipeline
    from .prior import get_daily_returns_for_prior, ticker_from_sa
except ImportError:
    import config
    from backtest import build_gain_series, portfolio_metrics
    from config import (
        ATIVOS_SENSITIVITY_DROP,
        OPT_DEFAULT_REBALANCE_MODE,
        OUT_SCORE_SENSITIVITY,
        PERIOD_END,
        PERIOD_START,
    )
    from params import BLParams
    from pipeline import run_pipeline
    from prior import get_daily_returns_for_prior, ticker_from_sa


def _run_universe(label: str, tickers: list[str], params: BLParams) -> dict[str, float | str]:
    original = list(config.ATIVOS)
    try:
        config.ATIVOS.clear()
        config.ATIVOS.extend(tickers)
        outputs = run_pipeline(
            params,
            period_start=PERIOD_START,
            period_end=PERIOD_END,
            save_outputs=False,
            verbose=False,
        )
        mode = params.rebalance_mode or OPT_DEFAULT_REBALANCE_MODE
        w = outputs["weights_by_mode"][mode]
        mu = outputs["posterior_mu"]
        returns = outputs["returns_history"]
        daily = returns.copy()
        daily.index = pd.to_datetime(daily.index)
        gain = build_gain_series(mu, w, daily, mode_label=mode)
        metrics = portfolio_metrics(gain)
        return {
            "universo": label,
            "tickers": ",".join(ticker_from_sa(t) for t in tickers),
            "n_ativos": len(tickers),
            "rebalance_mode": mode,
            **metrics,
        }
    finally:
        config.ATIVOS.clear()
        config.ATIVOS.extend(original)


def main() -> None:
    params = BLParams.from_config().normalized()
    full = list(config.ATIVOS)
    drop = [t for t in ATIVOS_SENSITIVITY_DROP if t in full]
    reduced = [t for t in full if t not in drop]
    if len(reduced) < 2:
        raise RuntimeError("Universo reduzido ficou com menos de 2 ativos.")

    print(f"Full: {full}")
    print(f"Drop: {drop}")
    print(f"Reduced: {reduced}")

    rows = [
        _run_universe("full", full, params),
        _run_universe("sem_" + "_".join(ticker_from_sa(t) for t in drop), reduced, params),
    ]
    out = pd.DataFrame(rows)
    OUT_SCORE_SENSITIVITY.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_SCORE_SENSITIVITY, index=False)
    print(out.to_string(index=False))
    print(f"Salvo em: {OUT_SCORE_SENSITIVITY}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
