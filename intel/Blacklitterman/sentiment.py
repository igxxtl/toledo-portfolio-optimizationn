"""Agregação de sentimento diário por ticker a partir das notícias."""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import NEWS_PATH, SENTIMENT_MAP
except ImportError:
    from config import NEWS_PATH, SENTIMENT_MAP

__all__ = [
    "load_daily_sentiment_for_ticker",
    "parse_news_datetime",
    "ticker_company_key",
]


def parse_news_datetime(data_str: str) -> datetime | None:
    if not data_str or not data_str.strip():
        return None
    try:
        return datetime.strptime(
            data_str.replace(" GMT", "").strip(),
            "%a, %d %b %Y %H:%M:%S",
        )
    except ValueError:
        return None


def ticker_company_key(ticker: str) -> str:
    clean = ticker.upper().replace(".SA", "")
    letters = "".join(re.findall(r"[A-Z]", clean))
    return letters[:4] if len(letters) >= 4 else letters


def load_daily_sentiment_for_ticker(
    ticker: str,
    news_path: Path | None = None,
    *,
    match_company: bool = True,
) -> pd.DataFrame:
    """
    Agrega sentimento diário para um ticker.

    ``match_company=True``: agrupa VALE3/VALE4 pela chave da empresa (usado no BL).
    ``match_company=False``: exige ticker exato (usado em testes de ativo único).
    """
    path = news_path or NEWS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de notícias não encontrado: {path}")

    rows = json.loads(path.read_text(encoding="utf-8"))
    parsed: list[dict[str, object]] = []
    target_key = ticker_company_key(ticker) if match_company else ticker.upper()

    for item in rows:
        item_ticker = (item.get("ticker") or "").strip().upper()
        if match_company:
            if ticker_company_key(item_ticker) != target_key:
                continue
        elif item_ticker != target_key:
            continue

        dt = parse_news_datetime(item.get("data") or "")
        if dt is None:
            janela_inicio = item.get("janela_inicio") or ""
            try:
                dt = datetime.strptime(janela_inicio, "%Y-%m-%d")
            except ValueError:
                continue

        sentimento = (item.get("sentimento") or "neutro").strip().lower()
        parsed.append(
            {
                "ticker": ticker,
                "view_date": pd.Timestamp(dt.date()),
                "sentimento_score": SENTIMENT_MAP.get(sentimento, 0.0),
            }
        )

    if not parsed:
        return pd.DataFrame(columns=["ticker", "view_date", "sentiment_signal", "news_count"])

    df = pd.DataFrame(parsed)
    grouped = df.groupby(["ticker", "view_date"], as_index=False).agg(
        sentiment_raw=("sentimento_score", "mean"),
        news_count=("sentimento_score", "size"),
    )
    grouped["sentiment_signal"] = np.tanh(grouped["sentiment_raw"] * np.log1p(grouped["news_count"]))
    return grouped[["ticker", "view_date", "sentiment_signal", "news_count"]]
