"""Gera o vetor Q (views do investidor) via XGBoost + sentimento — script standalone."""
from __future__ import annotations

import sys

import numpy as np

try:
    from .config import (
        ALPHA_Q,
        OUT_Q_STANDALONE_LONG,
        OUT_Q_STANDALONE_MATRIX,
        PERIOD_END,
        PERIOD_START,
    )
    from .io_utils import save_csv_with_date
    from .sentiment import load_daily_sentiment_for_ticker
    from .xgb_views import build_q_for_ticker
except ImportError:
    from config import (
        ALPHA_Q,
        OUT_Q_STANDALONE_LONG,
        OUT_Q_STANDALONE_MATRIX,
        PERIOD_END,
        PERIOD_START,
    )
    from io_utils import save_csv_with_date
    from sentiment import load_daily_sentiment_for_ticker
    from xgb_views import build_q_for_ticker

DEFAULT_TICKER = "ABEV3"


def run_q_standalone(ticker: str = DEFAULT_TICKER) -> None:
    print(f"Gerando Q com XGBoost para {ticker} ({PERIOD_START} -> {PERIOD_END})...")

    sentiment = load_daily_sentiment_for_ticker(ticker, match_company=False)
    q_long = build_q_for_ticker(
        ticker,
        sentiment=sentiment,
        alpha=ALPHA_Q,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        match_company_sentiment=False,
    )

    q_matrix = q_long.pivot(index="view_date", columns="ticker", values="Q").sort_index()
    save_csv_with_date(q_long, OUT_Q_STANDALONE_LONG)
    q_matrix_out = q_matrix.copy()
    q_matrix_out.index = q_matrix_out.index.strftime("%Y-%m-%d")
    OUT_Q_STANDALONE_LONG.parent.mkdir(parents=True, exist_ok=True)
    q_matrix_out.to_csv(OUT_Q_STANDALONE_MATRIX)

    hit = (np.sign(q_long["ret_pred"]) == np.sign(q_long["ret_real"])).mean() * 100

    print(f"\nViews salvas em: {OUT_Q_STANDALONE_LONG}")
    print(f"Matriz Q salva em: {OUT_Q_STANDALONE_MATRIX}")
    print(f"Linhas (dias com previsão): {len(q_long)}")
    print(f"Dias com notícia: {(q_long['news_count'] > 0).sum()}")
    print(f"Acurácia direcional XGB (pred vs real): {hit:.2f}%")
    print("\nÚltima linha da matriz Q:")
    print(q_matrix.tail(1).round(6))


def main() -> None:
    run_q_standalone()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
