"""Check mínimo do gate de cobertura de notícias (Q + Ω).

Rode: python _check_news_gate.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from omega_calc import calculate_omega
from xgb_views import combine_model_and_sentiment


def _toy_views(news_counts: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(news_counts)
    dates = pd.date_range("2025-01-02", periods=n, freq="B")
    views = pd.DataFrame(
        {
            "view_date": dates,
            "ticker": "TEST3",
            "ret_pred": np.full(n, 0.02),
            "ret_real": np.full(n, 0.01),
            "q_source": "xgb",
            "ret_scale": np.full(n, 0.01),
            "horizon": 5,
        }
    )
    sentiment = pd.DataFrame(
        {
            "view_date": dates,
            "ticker": "TEST3",
            "sentiment_signal": np.full(n, 1.0),
            "news_count": news_counts,
        }
    )
    return views, sentiment


def main() -> None:
    alpha = 0.6
    k = 2
    views, sentiment = _toy_views([0, 1, 2, 5])
    q = combine_model_and_sentiment(views, sentiment, alpha, news_count_min=k)

    row0 = q.iloc[0]
    assert abs(float(row0["Q"]) - float(row0["ret_pred"])) < 1e-12, "news=0 → Q=ret_pred"
    assert float(row0["news_coverage"]) == 0.0

    row_full = q.iloc[2]  # news_count=2 → coverage=1
    sent_comp = float(row_full["sentiment_component"])
    expected = alpha * float(row_full["ret_pred"]) + (1.0 - alpha) * sent_comp
    assert abs(float(row_full["Q"]) - expected) < 1e-12, "coverage cheia → híbrido clássico"
    assert abs(float(row_full["news_coverage"]) - 1.0) < 1e-12

    row_partial = q.iloc[1]  # news_count=1 → coverage=0.5
    c = 0.5
    expected_p = (alpha + (1 - alpha) * (1 - c)) * float(row_partial["ret_pred"]) + (
        1 - alpha
    ) * c * float(row_partial["sentiment_component"])
    assert abs(float(row_partial["Q"]) - expected_p) < 1e-12, "cobertura parcial"

    # Ω: low news deve ter omega maior que cobertura cheia (mesmo erro de modelo)
    omega_soft = calculate_omega(
        q,
        tau=0.05,
        news_count_min=k,
        news_omega_boost=3.0,
        hard_prior_when_low_news=False,
        confidence_floor=0.2,
        confidence_cap=0.9,
        w_model_conf=0.5,
        w_news_conf=0.5,
        err_window=3,
        err_min_periods=2,
    )
    low = omega_soft.loc[omega_soft["news_count"] == 0].iloc[0]
    full = omega_soft.loc[omega_soft["news_count"] >= k].iloc[0]
    assert float(low["omega_boost"]) == 3.0
    assert float(full["omega_boost"]) == 1.0
    assert float(low["omega"]) > float(full["omega"]), "low news → omega maior"

    omega_hard = calculate_omega(
        q,
        tau=0.05,
        news_count_min=k,
        news_omega_boost=3.0,
        hard_prior_when_low_news=True,
        confidence_floor=0.2,
        confidence_cap=0.9,
        w_model_conf=0.5,
        w_news_conf=0.5,
        err_window=3,
        err_min_periods=2,
    )
    hard_low = omega_hard.loc[omega_hard["news_count"] == 0].iloc[0]
    assert float(hard_low["view_confidence"]) < 1e-6, "hard prior → conf ~ 0"

    print("OK: news coverage gate (Q + Omega)")


if __name__ == "__main__":
    main()
