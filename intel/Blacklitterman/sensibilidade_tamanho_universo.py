"""Sensibilidade ao tamanho do universo: k=1..N ativos (por hit ratio).

Para cada k, roda o pipeline BL só com os k melhores tickers (hit ratio XGB)
e registra métricas de backtest. Params Optuna ficam fixos; só o universo muda.
O teto por ativo sobe para max(config, 1/k) quando k é pequeno (viabilidade).

Uso:
    python -m intel.Blacklitterman.sensibilidade_tamanho_universo
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

try:
    from . import config
    from .backtest import build_gain_series, portfolio_metrics
    from .config import (
        OPT_DEFAULT_REBALANCE_MODE,
        OUT_SCORE_PRED_TICKER,
        PERIOD_END,
        PERIOD_START,
        SCORE_DIR,
    )
    from .params import BLParams
    from .pipeline import run_pipeline
    from .prior import ticker_from_sa
except ImportError:
    import config
    from backtest import build_gain_series, portfolio_metrics
    from config import (
        OPT_DEFAULT_REBALANCE_MODE,
        OUT_SCORE_PRED_TICKER,
        PERIOD_END,
        PERIOD_START,
        SCORE_DIR,
    )
    from params import BLParams
    from pipeline import run_pipeline
    from prior import ticker_from_sa

OUT_CSV = SCORE_DIR / "sensibilidade_tamanho_universo.csv"


def _ranked_ativos_by_hit() -> list[str]:
    """Ordena ATIVOS pelo hit_ratio em score (desc); fallback = ordem do config."""
    full = list(config.ATIVOS)
    if not OUT_SCORE_PRED_TICKER.exists():
        return full
    hits = pd.read_csv(OUT_SCORE_PRED_TICKER)
    if "ticker" not in hits.columns or "hit_ratio" not in hits.columns:
        return full
    order = hits.sort_values("hit_ratio", ascending=False)["ticker"].astype(str).tolist()
    by_base = {ticker_from_sa(t): t for t in full}
    ranked = [by_base[t] for t in order if t in by_base]
    for t in full:
        if t not in ranked:
            ranked.append(t)
    return ranked


def _run_universe(label: str, tickers: list[str], params: BLParams) -> dict[str, float | str | int]:
    original = list(config.ATIVOS)
    try:
        config.ATIVOS.clear()
        config.ATIVOS.extend(tickers)
        # teto viável: max_w * n >= 1
        min_cap = 1.0 / len(tickers) + 1e-9
        params_k = replace(params, max_weight_per_asset=max(float(params.max_weight_per_asset), min_cap))
        outputs = run_pipeline(
            params_k,
            period_start=PERIOD_START,
            period_end=PERIOD_END,
            save_outputs=False,
            verbose=False,
        )
        mode = params_k.rebalance_mode or OPT_DEFAULT_REBALANCE_MODE
        w = outputs["weights_by_mode"][mode]
        mu = outputs["posterior_mu"]
        returns = outputs["returns_history"]
        daily = returns.copy()
        daily.index = pd.to_datetime(daily.index)
        gain = build_gain_series(mu, w, daily, mode_label=mode)
        metrics = portfolio_metrics(gain)
        return {
            "universo": label,
            "n_ativos": len(tickers),
            "tickers": ",".join(ticker_from_sa(t) for t in tickers),
            "max_weight_used_cap": params_k.max_weight_per_asset,
            "rebalance_mode": mode,
            **metrics,
        }
    finally:
        config.ATIVOS.clear()
        config.ATIVOS.extend(original)


def main() -> None:
    ranked = _ranked_ativos_by_hit()
    params = BLParams.from_config(rebalance_mode=OPT_DEFAULT_REBALANCE_MODE).normalized()
    print(f"Ordem (hit ratio): {[ticker_from_sa(t) for t in ranked]}")
    print(f"Rebalance: {params.rebalance_mode} | periodo {PERIOD_START}-{PERIOD_END}")
    print(f"k = 1 .. {len(ranked)}\n")

    rows: list[dict[str, float | str | int]] = []
    for k in range(1, len(ranked) + 1):
        subset = ranked[:k]
        label = f"top{k}"
        print(f"[{k}/{len(ranked)}] {label}: {', '.join(ticker_from_sa(t) for t in subset)}")
        try:
            row = _run_universe(label, subset, params)
            rows.append(row)
            print(
                f"  sharpe={row.get('sharpe', float('nan')):.3f} | "
                f"ret={row.get('total_return_pct', float('nan')):.2f}% | "
                f"final={row.get('final_capital_brl', float('nan')):.0f}"
            )
        except Exception as exc:
            print(f"  FALHA: {exc}")
            rows.append(
                {
                    "universo": label,
                    "n_ativos": k,
                    "tickers": ",".join(ticker_from_sa(t) for t in subset),
                    "error": str(exc),
                }
            )

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nSalvo em: {OUT_CSV}")


if __name__ == "__main__":
    # ponytail: ranking estável e teto viável para k=1
    assert 1.0 / 1 + 1e-9 >= 1.0
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
