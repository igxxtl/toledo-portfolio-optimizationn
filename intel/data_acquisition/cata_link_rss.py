"""Coleta notícias B3 via Google News RSS."""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
import pandas as pd

try:
    from .config import (
        COLLECTION_END,
        COLLECTION_START,
        NEWS_RAW,
        RSS_COUNTRY,
        RSS_LANGUAGE,
        RSS_MAX_RETRIES,
        RSS_PAUSE_SECONDS,
        RSS_USER_AGENT,
        TICKERS,
    )
except ImportError:
    from config import (
        COLLECTION_END,
        COLLECTION_START,
        NEWS_RAW,
        RSS_COUNTRY,
        RSS_LANGUAGE,
        RSS_MAX_RETRIES,
        RSS_PAUSE_SECONDS,
        RSS_USER_AGENT,
        TICKERS,
    )


def generate_weekly_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Gera pares (início, fim_exclusive) para janelas semanais."""
    windows: list[tuple[date, date]] = []
    current = start
    while current <= end:
        end_exclusive = current + timedelta(days=7)
        if end_exclusive > end:
            end_exclusive = end + timedelta(days=1)
        windows.append((current, end_exclusive))
        current += timedelta(days=7)
    return windows


def build_rss_url(term: str, start: date, end_exclusive: date) -> str:
    query = f"{term} after:{start} before:{end_exclusive}"
    q_encoded = quote_plus(query)
    return (
        f"https://news.google.com/rss/search?"
        f"q={q_encoded}&hl={RSS_LANGUAGE}&gl={RSS_COUNTRY}&ceid={RSS_COUNTRY}%3Apt-419"
    )


def fetch_feed_with_retry(url: str) -> feedparser.FeedParserDict:
    for attempt in range(1, RSS_MAX_RETRIES + 1):
        try:
            return feedparser.parse(url, request_headers={"User-Agent": RSS_USER_AGENT})
        except Exception as exc:
            if attempt == RSS_MAX_RETRIES:
                raise
            wait = 5 * (2 ** (attempt - 1))
            print(f"  Erro ({exc.__class__.__name__}), retry em {wait}s ({attempt}/{RSS_MAX_RETRIES})...")
            time.sleep(wait)
    raise RuntimeError("Falha inesperada ao buscar feed RSS.")


def _parse_rss_date(raw: str | None) -> datetime:
    if not raw or not str(raw).strip():
        return datetime.min
    try:
        return datetime.strptime(str(raw)[:25], "%a, %d %b %Y %H:%M:%S")
    except ValueError:
        return datetime.min


def collect_news() -> list[dict[str, object]]:
    windows = generate_weekly_windows(COLLECTION_START, COLLECTION_END)
    print(f"Janelas: {len(windows)} semanas ({COLLECTION_START} a {COLLECTION_END})\n")

    df_total = pd.DataFrame()
    for ticker in TICKERS:
        for start, end_exclusive in windows:
            feed = fetch_feed_with_retry(build_rss_url(ticker, start, end_exclusive))
            records = [
                {
                    "ticker": ticker,
                    "janela_inicio": start.isoformat(),
                    "janela_fim": (end_exclusive - timedelta(days=1)).isoformat(),
                    "titulo": entry.title,
                    "fonte": entry.source.title if "source" in entry else None,
                    "data": entry.published,
                    "link_google": entry.link,
                }
                for entry in feed.entries
            ]
            df_total = pd.concat([df_total, pd.DataFrame(records)], ignore_index=True)
            df_total.drop_duplicates(subset="link_google", inplace=True)
            print(f"  {ticker} {start} a {end_exclusive} → total: {len(df_total)}")
            time.sleep(RSS_PAUSE_SECONDS)

    df_total["_data_ordem"] = pd.to_datetime(df_total["data"], errors="coerce")
    df_total = df_total.sort_values("_data_ordem", ascending=False).drop(columns=["_data_ordem"])

    records = df_total.to_dict(orient="records")
    for row in records:
        value = row.get("data")
        row["data"] = str(value) if value is not None and pd.notna(value) else None

    records.sort(key=lambda r: _parse_rss_date(r.get("data") if isinstance(r.get("data"), str) else None), reverse=True)
    return records


def save_news(records: list[dict[str, object]], path: Path | None = None) -> Path:
    output = path or NEWS_RAW
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    records = collect_news()
    output = save_news(records)
    print(f"\nTotal final de notícias: {len(records)}")
    print(f"Salvo em: {output}")


if __name__ == "__main__":
    main()
