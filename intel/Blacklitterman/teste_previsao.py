"""
Teste de acuracia das previsoes do modelo (ret_7h_pred vs ret_7h_real).

Entradas:
- bl_hibrido_q_long.csv

Saidas:
- bl_hibrido_previsao_metricas_gerais.csv
- bl_hibrido_previsao_por_ticker.csv
- bl_hibrido_previsao_por_mes.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BL_DIR = Path(__file__).resolve().parent
Q_LONG_PATH = BL_DIR / "bl_hibrido_q_long.csv"

OUT_GERAL = BL_DIR / "bl_hibrido_previsao_metricas_gerais.csv"
OUT_TICKER = BL_DIR / "bl_hibrido_previsao_por_ticker.csv"
OUT_MES = BL_DIR / "bl_hibrido_previsao_por_mes.csv"


def _direction(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    out = np.zeros_like(arr)
    out[arr > eps] = 1.0
    out[arr < -eps] = -1.0
    return pd.Series(out, index=x.index)


def _metrics(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "n": 0,
            "hit_ratio": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    pred = pd.to_numeric(df["ret_7h_pred"], errors="coerce")
    real = pd.to_numeric(df["ret_7h_real"], errors="coerce")
    valid = pred.notna() & real.notna()
    pred = pred[valid]
    real = real[valid]
    if pred.empty:
        return {
            "n": 0,
            "hit_ratio": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    dir_pred = _direction(pred)
    dir_real = _direction(real)

    hit_ratio = float((dir_pred == dir_real).mean())
    mae = float((pred - real).abs().mean())
    rmse = float(np.sqrt(np.mean((pred - real) ** 2)))

    # Matriz de confusao binaria simplificada: positivo vs nao-positivo
    pos_pred = dir_pred > 0
    pos_real = dir_real > 0
    tp = int((pos_pred & pos_real).sum())
    tn = int((~pos_pred & ~pos_real).sum())
    fp = int((pos_pred & ~pos_real).sum())
    fn = int((~pos_pred & pos_real).sum())

    return {
        "n": int(len(pred)),
        "hit_ratio": hit_ratio,
        "mae": mae,
        "rmse": rmse,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    q = pd.read_csv(Q_LONG_PATH)
    q["view_date"] = pd.to_datetime(q["view_date"], errors="coerce")
    q = q.dropna(subset=["view_date", "ret_7h_pred", "ret_7h_real"]).copy()
    if q.empty:
        raise ValueError("Arquivo de views vazio ou sem colunas validas.")

    geral = pd.DataFrame([_metrics(q)])
    geral.to_csv(OUT_GERAL, index=False)

    ticker_rows = []
    for ticker, g in q.groupby("ticker", sort=True):
        m = _metrics(g)
        m["ticker"] = str(ticker)
        ticker_rows.append(m)
    by_ticker = pd.DataFrame(ticker_rows)[
        ["ticker", "n", "hit_ratio", "mae", "rmse", "tp", "tn", "fp", "fn"]
    ].sort_values("ticker")
    by_ticker.to_csv(OUT_TICKER, index=False)

    q["mes"] = q["view_date"].dt.to_period("M").astype(str)
    month_rows = []
    for month, g in q.groupby("mes", sort=True):
        m = _metrics(g)
        m["mes"] = month
        month_rows.append(m)
    by_month = pd.DataFrame(month_rows)[
        ["mes", "n", "hit_ratio", "mae", "rmse", "tp", "tn", "fp", "fn"]
    ].sort_values("mes")
    by_month.to_csv(OUT_MES, index=False)

    print(f"Hit ratio geral: {float(geral['hit_ratio'].iloc[0]) * 100:.2f}%")
    print(f"MAE geral: {float(geral['mae'].iloc[0]):.6f}")
    print(f"RMSE geral: {float(geral['rmse'].iloc[0]):.6f}")
    print(f"Saidas: {OUT_GERAL}, {OUT_TICKER}, {OUT_MES}")


if __name__ == "__main__":
    main()
