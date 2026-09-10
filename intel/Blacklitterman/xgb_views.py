"""Views (Q) a partir das predições walk-forward do XGBoost."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import (
        DEFAULT_VIEW_HORIZON,
        NEWS_COUNT_MIN,
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
        DEFAULT_VIEW_HORIZON,
        NEWS_COUNT_MIN,
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
    "align_q_to_daily",
    "build_q_from_cached_views",
    "build_q_for_ticker",
    "build_q_for_tickers",
    "combine_model_and_sentiment",
    "load_ticker_horizons",
    "load_xgb_predictions",
    "month_bounds",
    "normalize_return_columns",
    "prediction_file_for_ticker",
    "resolve_view_horizon",
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


def load_ticker_horizons(xgb_dir: Path | None = None) -> dict[str, int]:
    """Lê horizonte vencedor por ticker em ``metricas_vencedores.csv``."""
    xgb_dir = xgb_dir or XGB_DIR
    path = xgb_dir / "metricas_vencedores.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "ticker" not in df.columns or "horizon" not in df.columns:
        return {}
    out: dict[str, int] = {}
    for _, row in df.iterrows():
        ticker = str(row["ticker"]).upper().replace(".SA", "")
        try:
            h = int(row["horizon"])
        except (TypeError, ValueError):
            continue
        if h >= 1:
            out[ticker] = h
    return out


def resolve_view_horizon(
    ticker: str,
    horizons: dict[str, int] | None = None,
    *,
    default: int = DEFAULT_VIEW_HORIZON,
) -> int:
    base = ticker.upper().replace(".SA", "")
    if horizons and base in horizons:
        return max(int(horizons[base]), 1)
    return max(int(default), 1)


def align_q_to_daily(q_long: pd.DataFrame) -> pd.DataFrame:
    """Converte Q (escala do horizonte XGB) para retorno diário equivalente.

    Mantém ``ret_pred`` / ``ret_real`` no horizonte original (métricas de previsão).
    Guarda ``Q_horizon`` e define ``q_time_scale='daily'`` para não reescalar duas vezes.
    """
    out = normalize_return_columns(q_long).copy()
    if out.empty:
        return out
    if "q_time_scale" in out.columns and (out["q_time_scale"] == "daily").all():
        return out

    if "horizon" not in out.columns:
        horizons = load_ticker_horizons()
        out["horizon"] = out["ticker"].map(
            lambda t: resolve_view_horizon(str(t), horizons)
        )
    h = pd.to_numeric(out["horizon"], errors="coerce").fillna(DEFAULT_VIEW_HORIZON).clip(lower=1.0)
    out["horizon"] = h.astype(int)
    if "Q_horizon" not in out.columns:
        out["Q_horizon"] = pd.to_numeric(out["Q"], errors="coerce")
    out["Q"] = out["Q_horizon"] / h
    out["q_time_scale"] = "daily"
    return out


def load_xgb_predictions(
    ticker: str,
    xgb_dir: Path | None = None,
    period_start: str = PERIOD_START,
    period_end: str = PERIOD_END,
    *,
    horizon: int | None = None,
) -> pd.DataFrame:
    """Carrega predições OOF do XGBoost.

    ``ret_pred`` / ``ret_real``: log-retorno do horizonte do modelo (ex.: 5/10/21 dias).
    Usar ``align_q_to_daily`` antes do BL; para score de previsão, manter a escala original.
    """
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

    if horizon is None:
        horizon = resolve_view_horizon(base, load_ticker_horizons(xgb_dir))

    start_date, end_date = month_bounds(period_start, period_end)
    out = pd.DataFrame(
        {
            "view_date": pd.to_datetime(df["date"], errors="coerce").dt.normalize(),
            "ticker": base,
            "ret_pred": pd.to_numeric(df["y_pred"], errors="coerce"),
            "ret_real": pd.to_numeric(df["y_real"], errors="coerce"),
            "q_source": "xgb",
            "horizon": int(horizon),
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
    *,
    news_count_min: int = NEWS_COUNT_MIN,
) -> pd.DataFrame:
    """Combina previsão XGB (``ret_pred``) com componente de sentimento.

    Gate de cobertura: ``coverage = min(1, news_count / news_count_min)``.
    Sem notícias → ``Q = ret_pred``; com cobertura cheia → híbrido α clássico.

    ``Q`` sai na escala do horizonte do modelo; chamar ``align_q_to_daily`` antes do BL.
    """
    views = normalize_return_columns(views)
    merged = views.merge(sentiment, on=["ticker", "view_date"], how="left")
    merged["sentiment_signal"] = merged["sentiment_signal"].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["sentiment_component"] = merged["sentiment_signal"] * merged["ret_scale"]
    k = max(int(news_count_min), 1)
    coverage = (merged["news_count"].astype(float) / float(k)).clip(upper=1.0)
    merged["news_coverage"] = coverage
    # Q = [α + (1-α)·(1-c)] · ret_pred + (1-α)·c · sentiment_component
    merged["Q"] = (
        (alpha + (1.0 - alpha) * (1.0 - coverage)) * merged["ret_pred"]
        + (1.0 - alpha) * coverage * merged["sentiment_component"]
    )

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
        "news_coverage",
        "ret_scale",
        "horizon",
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
    horizons: dict[str, int] | None = None,
    news_count_min: int = NEWS_COUNT_MIN,
) -> pd.DataFrame:
    """Pipeline completo de Q para um único ticker (escala do horizonte; alinhar depois)."""
    base = ticker.upper().replace(".SA", "")
    xgb_dir = xgb_dir or XGB_DIR
    if horizons is None:
        horizons = load_ticker_horizons(xgb_dir)
    horizon = resolve_view_horizon(base, horizons)
    views = load_xgb_predictions(
        base, xgb_dir, period_start, period_end, horizon=horizon
    )
    if views.empty:
        raise RuntimeError(f"Nenhuma view XGB no período para {base}.")

    views = add_ret_scale(views)
    if sentiment is None:
        sentiment = load_daily_sentiment_for_ticker(base, match_company=match_company_sentiment)
    return combine_model_and_sentiment(
        views, sentiment, alpha, news_count_min=news_count_min
    )


def build_q_for_tickers(
    tickers: list[str],
    *,
    alpha: float,
    xgb_dir: Path | None = None,
    period_start: str = PERIOD_START,
    period_end: str = PERIOD_END,
    news_count_min: int = NEWS_COUNT_MIN,
) -> pd.DataFrame:
    """Gera Q alinhado à escala diária (compatível com π) para múltiplos tickers."""
    xgb_dir = xgb_dir or XGB_DIR
    horizons = load_ticker_horizons(xgb_dir)
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
                    horizons=horizons,
                    news_count_min=news_count_min,
                )
            )
        except Exception as exc:
            print(f"  - {base}: ignorado ({exc})")

    if not parts:
        raise RuntimeError("Nenhuma view Q foi gerada para os tickers configurados.")
    q_long = pd.concat(parts, axis=0, ignore_index=True).sort_values(["view_date", "ticker"]).reset_index(drop=True)
    return align_q_to_daily(q_long)


def build_q_from_cached_views(
    views_by_ticker: dict[str, pd.DataFrame],
    sentiment_by_ticker: dict[str, pd.DataFrame],
    *,
    alpha: float,
    news_count_min: int = NEWS_COUNT_MIN,
) -> pd.DataFrame:
    """Monta Q a partir de views XGB (com ret_scale) e sentimento já carregados."""
    horizons = load_ticker_horizons()
    parts: list[pd.DataFrame] = []
    for ticker, views in views_by_ticker.items():
        sentiment = sentiment_by_ticker.get(ticker)
        if sentiment is None:
            raise KeyError(f"Sentimento não encontrado para {ticker}")
        v = normalize_return_columns(views).copy()
        if "horizon" not in v.columns:
            v["horizon"] = resolve_view_horizon(ticker, horizons)
        parts.append(
            combine_model_and_sentiment(
                v, sentiment, alpha, news_count_min=news_count_min
            )
        )
    if not parts:
        raise RuntimeError("Nenhuma view Q foi montada a partir do cache.")
    q_long = pd.concat(parts, axis=0, ignore_index=True).sort_values(["view_date", "ticker"]).reset_index(drop=True)
    return align_q_to_daily(q_long)
