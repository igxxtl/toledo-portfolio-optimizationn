"""Views (Q) a partir das predições walk-forward do XGBoost."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import (
        PERIOD_END,
        PERIOD_START,
        RET_SCALE_FALLBACK,
        RET_SCALE_MIN_PERIODS,
        RET_SCALE_WINDOW,
        XGB_DIR,
    )
    from .sentiment import load_daily_sentiment_for_ticker
except ImportError:
    from config import (
        PERIOD_END,
        PERIOD_START,
        RET_SCALE_FALLBACK,
        RET_SCALE_MIN_PERIODS,
        RET_SCALE_WINDOW,
        XGB_DIR,
    )
    from sentiment import load_daily_sentiment_for_ticker

__all__ = [
    "add_ret_scale",
    "build_q_for_ticker",
    "build_q_for_tickers",
    "combine_model_and_sentiment",
    "load_xgb_predictions",
    "month_bounds",
    "normalize_return_columns",
    "prediction_file_for_ticker",
]


def month_bounds(period_start: str, period_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=int(period_start[:4]), month=int(period_start[4:]), day=1)
    end_month = pd.Timestamp(year=int(period_end[:4]), month=int(period_end[4:]), day=1)
    end = end_month + pd.offsets.MonthEnd(1)
    return start.normalize(), end.normalize()


def normalize_return_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compatibilidade com CSVs legados (``ret_7h_pred`` → ``ret_pred``)."""
    out = df.copy()
    rename: dict[str, str] = {}
    if "ret_pred" not in out.columns and "ret_7h_pred" in out.columns:
        rename["ret_7h_pred"] = "ret_pred"
    if "ret_real" not in out.columns and "ret_7h_real" in out.columns:
        rename["ret_7h_real"] = "ret_real"
    return out.rename(columns=rename) if rename else out


def prediction_file_for_ticker(ticker: str, xgb_dir: Path) -> Path | None:
    """Localiza o CSV de predições OOF do XGB para o ticker."""
    base = ticker.upper().replace(".SA", "")
    for name in (f"{base}_predicoes.csv", f"{base}_h21d_predicoes.csv", f"{base}_h1d_predicoes.csv"):
        path = xgb_dir / name
        if path.exists():
            return path
    return None


def load_xgb_predictions(
    ticker: str,
    xgb_dir: Path | None = None,
    period_start: str = PERIOD_START,
    period_end: str = PERIOD_END,
) -> pd.DataFrame:
    """Carrega predições OOF do XGBoost. Colunas: view_date, ticker, ret_pred, ret_real."""
    xgb_dir = xgb_dir or XGB_DIR
    base = ticker.upper().replace(".SA", "")
    pred_path = prediction_file_for_ticker(base, xgb_dir)
    if pred_path is None:
        raise FileNotFoundError(
            f"Predições XGB não encontradas para {base} em {xgb_dir}. "
            "Gere os arquivos *_predicoes.csv antes de rodar o pipeline."
        )

    df = pd.read_csv(pred_path)
    missing = {"date", "y_pred", "y_real"}.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {pred_path.name}: {sorted(missing)}")

    start_date, end_date = month_bounds(period_start, period_end)
    out = pd.DataFrame(
        {
            "view_date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
            "ticker": base,
            "ret_pred": pd.to_numeric(df["y_pred"], errors="coerce"),
            "ret_real": pd.to_numeric(df["y_real"], errors="coerce"),
            "q_source": "xgb",
        }
    )
    out = out.dropna(subset=["view_date", "ret_pred", "ret_real"])
    mask = (out["view_date"] >= start_date) & (out["view_date"] <= end_date)
    return out.loc[mask].sort_values("view_date").reset_index(drop=True)


def add_ret_scale(views: pd.DataFrame) -> pd.DataFrame:
    """Escala de retorno para converter sentimento em unidade de retorno."""
    out = views.sort_values("view_date").copy()
    out["ret_scale"] = (
        out["ret_real"].abs().rolling(RET_SCALE_WINDOW, min_periods=RET_SCALE_MIN_PERIODS).mean().shift(1)
    )
    fallback = float(out["ret_real"].abs().median())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = RET_SCALE_FALLBACK
    out["ret_scale"] = out["ret_scale"].fillna(fallback)
    return out


def combine_model_and_sentiment(
    views: pd.DataFrame,
    sentiment: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Combina previsão XGB (``ret_pred``) com componente de sentimento."""
    views = normalize_return_columns(views)
    merged = views.merge(sentiment, on=["ticker", "view_date"], how="left")
    merged["sentiment_signal"] = merged["sentiment_signal"].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["sentiment_component"] = merged["sentiment_signal"] * merged["ret_scale"]
    merged["Q"] = alpha * merged["ret_pred"] + (1.0 - alpha) * merged["sentiment_component"]

    cols = [
        "view_date",
        "ticker",
        "Q",
        "ret_pred",
        "ret_real",
        "q_source",
        "sentiment_signal",
        "sentiment_component",
        "news_count",
        "ret_scale",
    ]
    present = [c for c in cols if c in merged.columns]
    return merged[present].sort_values(["view_date", "ticker"]).reset_index(drop=True)


def build_q_for_ticker(
    ticker: str,
    sentiment: pd.DataFrame | None = None,
    *,
    alpha: float,
    xgb_dir: Path | None = None,
    period_start: str = PERIOD_START,
    period_end: str = PERIOD_END,
    match_company_sentiment: bool = True,
) -> pd.DataFrame:
    """Pipeline completo de Q para um único ticker."""
    base = ticker.upper().replace(".SA", "")
    views = load_xgb_predictions(base, xgb_dir, period_start, period_end)
    if views.empty:
        raise RuntimeError(f"Nenhuma view XGB no período para {base}.")

    views = add_ret_scale(views)
    if sentiment is None:
        sentiment = load_daily_sentiment_for_ticker(base, match_company=match_company_sentiment)
    return combine_model_and_sentiment(views, sentiment, alpha)


def build_q_for_tickers(
    tickers: list[str],
    *,
    alpha: float,
    xgb_dir: Path | None = None,
    period_start: str = PERIOD_START,
    period_end: str = PERIOD_END,
) -> pd.DataFrame:
    """Gera Q diário para múltiplos tickers."""
    parts: list[pd.DataFrame] = []
    for ticker in tickers:
        base = ticker.upper().replace(".SA", "")
        try:
            parts.append(
                build_q_for_ticker(
                    base,
                    alpha=alpha,
                    xgb_dir=xgb_dir,
                    period_start=period_start,
                    period_end=period_end,
                    match_company_sentiment=True,
                )
            )
        except Exception as exc:
            print(f"  - {base}: ignorado ({exc})")

    if not parts:
        raise RuntimeError("Nenhuma view Q foi gerada para os tickers configurados.")
    return pd.concat(parts, axis=0, ignore_index=True).sort_values(["view_date", "ticker"]).reset_index(drop=True)
