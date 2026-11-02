"""Compara os limiares de noticias n=0..5 sem sobrescrever o pipeline padrao.

Executar: python -m intel.Blacklitterman.simular_cenarios_noticias
Saidas: outputs/score/cenarios_noticias_percentuais.csv e cenarios_noticias_scores.csv
"""
from __future__ import annotations

from dataclasses import replace

import pandas as pd

try:
    from .backtest import build_gain_series
    from .config import ATIVOS, OPT_DEFAULT_REBALANCE_MODE, SCORE_DIR
    from .params import BLParams
    from .pipeline import run_pipeline
    from .prior import get_daily_returns_for_prior
    from .score_modelo import W_PORT, W_PRED, W_RISK, score_portfolio, score_predicao, score_risco
except ImportError:
    from backtest import build_gain_series
    from config import ATIVOS, OPT_DEFAULT_REBALANCE_MODE, SCORE_DIR
    from params import BLParams
    from pipeline import run_pipeline
    from prior import get_daily_returns_for_prior
    from score_modelo import W_PORT, W_PRED, W_RISK, score_portfolio, score_predicao, score_risco


def simular_cenarios(
    *,
    minimo: int = 0,
    maximo: int = 5,
    modo: str = OPT_DEFAULT_REBALANCE_MODE,
    salvar: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna (percentuais_de_noticias, scores_finais), indexados pelo limiar n.

    Percentuais usam como denominador todas as observacoes ticker-dia de Q,
    inclusive dias sem noticias. n=0 inclui todas as observacoes.
    """
    if not (0 <= minimo <= maximo):
        raise ValueError("Intervalo invalido: esperado 0 <= minimo <= maximo.")
    if modo not in {"daily", "weekly", "monthly"}:
        raise ValueError("modo deve ser daily, weekly ou monthly")

    base = BLParams.from_config()
    pct_rows = []
    score_rows = []
    # Precos sao comuns a todos os cenarios; nao ha download de mercado.
    returns = get_daily_returns_for_prior(
        tickers=ATIVOS,
        start_date="2023-01-01",
        end_date="2025-12-31",
    )
    for n in range(minimo, maximo + 1):
        params = replace(base, news_count_min=n, rebalance_mode=modo)
        result = run_pipeline(
            params=params,
            returns_history=returns,
            save_outputs=False,
            verbose=False,
        )
        q = result["q_long"]
        counts = pd.to_numeric(q["news_count"], errors="coerce").fillna(0)
        total = len(counts)
        if not total:
            raise ValueError(f"Nenhuma observacao de Q no cenario n={n}")
        eligible = counts >= n
        with_news = counts > 0
        pct_rows.append({
            "n": n,
            "observacoes_total": total,
            "observacoes_elegiveis": int(eligible.sum()),
            "pct_elegiveis": float(eligible.mean() * 100),
            "pct_abaixo_limiar": float((~eligible).mean() * 100),
            "pct_com_noticias": float(with_news.mean() * 100),
            "pct_sem_noticias": float((~with_news).mean() * 100),
        })

        weights = result["weights_by_mode"][modo]
        mu = result["posterior_mu"]
        assets = [c for c in weights.columns if c.endswith(".SA") and c in mu.columns]
        daily_returns = returns.reindex(columns=assets)
        gain = build_gain_series(mu, weights, daily_returns, mode_label=modo)
        sp, _ = score_predicao(q)
        so, port = score_portfolio(gain)
        sr, _ = score_risco(weights)
        score_rows.append({
            "n": n,
            "modo": modo,
            "score_total": 100 * (W_PRED * sp + W_PORT * so + W_RISK * sr),
            "score_pred": 100 * sp,
            "score_port": 100 * so,
            "score_risk": 100 * sr,
            "retorno_real": port.get("total_return"),
            "sharpe_real": port.get("sharpe"),
            "max_drawdown_real": port.get("max_drawdown"),
        })
    percentuais = pd.DataFrame(pct_rows)
    scores = pd.DataFrame(score_rows)
    if salvar:
        SCORE_DIR.mkdir(parents=True, exist_ok=True)
        percentuais.to_csv(SCORE_DIR / "cenarios_noticias_percentuais.csv", index=False)
        scores.to_csv(SCORE_DIR / "cenarios_noticias_scores.csv", index=False)
    return percentuais, scores


if __name__ == "__main__":
    percentuais, scores = simular_cenarios()
    print("\nPercentuais de noticias:\n", percentuais.to_string(index=False))
    print("\nScores finais:\n", scores.to_string(index=False))
