"""Coleta notícias B3 via Google News RSS."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

import feedparser

try:
    from .config import (
        COLLECTION_END,
        COLLECTION_START,
        NEWS_RAW,
        RSS_COUNTRY,
        RSS_FAIL_COOLDOWN_SECONDS,
        RSS_LANGUAGE,
        RSS_MAX_BACKOFF_ROUNDS,
        RSS_MAX_RETRIES,
        RSS_MAX_WORKERS,
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
        RSS_FAIL_COOLDOWN_SECONDS,
        RSS_LANGUAGE,
        RSS_MAX_BACKOFF_ROUNDS,
        RSS_MAX_RETRIES,
        RSS_MAX_WORKERS,
        RSS_PAUSE_SECONDS,
        RSS_USER_AGENT,
        TICKERS,
    )

Job = tuple[str, date, date]
_HARD_FAIL_STATUSES = {403, 408, 429, 500, 502, 503, 504}


class FeedFetchError(RuntimeError):
    """Falha recuperável de fetch RSS (rate-limit, HTTP ruim, rede)."""


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
    # aspas no ticker + janela semanal: RSS do Google corta ~100 itens;
    # mensal perde manchetes em tickers quentes (VALE3 etc.)
    query = f'"{term}" after:{start} before:{end_exclusive}'
    q_encoded = quote_plus(query)
    return (
        f"https://news.google.com/rss/search?"
        f"q={q_encoded}&hl={RSS_LANGUAGE}&gl={RSS_COUNTRY}&ceid={RSS_COUNTRY}%3Apt-419"
    )


def _feed_looks_failed(feed: feedparser.FeedParserDict) -> str | None:
    status = int(getattr(feed, "status", 200) or 200)
    if status in _HARD_FAIL_STATUSES or status >= 500:
        return f"HTTP {status}"
    # resposta HTML/captcha em vez de RSS
    if getattr(feed, "bozo", False) and not feed.entries:
        return f"bozo ({getattr(feed, 'bozo_exception', 'parse error')})"
    return None


def fetch_feed_with_retry(url: str) -> feedparser.FeedParserDict:
    last_err: Exception | None = None
    for attempt in range(1, RSS_MAX_RETRIES + 1):
        try:
            feed = feedparser.parse(url, request_headers={"User-Agent": RSS_USER_AGENT})
            reason = _feed_looks_failed(feed)
            if reason is None:
                return feed
            last_err = FeedFetchError(reason)
        except Exception as exc:
            last_err = exc
        if attempt == RSS_MAX_RETRIES:
            break
        wait = 5 * (2 ** (attempt - 1))
        print(f"  Erro ({last_err}), retry em {wait}s ({attempt}/{RSS_MAX_RETRIES})...")
        time.sleep(wait)
    raise FeedFetchError(f"Falha ao buscar feed após {RSS_MAX_RETRIES} tentativas: {last_err}")


def _parse_rss_date(raw: str | None) -> datetime:
    if not raw or not str(raw).strip():
        return datetime.min
    try:
        return datetime.strptime(str(raw)[:25], "%a, %d %b %Y %H:%M:%S")
    except ValueError:
        return datetime.min


def _backoff(workers: int, pause: float) -> tuple[int, float]:
    """Reduz paralelismo e aumenta pausa após falhas."""
    if workers > 1:
        return max(1, workers // 2), pause * 1.5
    return 1, pause * 2.0


def _fetch_window(ticker: str, start: date, end_exclusive: date, pause: float) -> list[dict[str, object]]:
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
    time.sleep(pause)
    return records


def _run_batch(
    jobs: list[Job],
    workers: int,
    pause: float,
) -> tuple[list[dict[str, object]], list[Job]]:
    """Executa jobs; devolve (records ok, jobs que falharam)."""
    ok_records: list[dict[str, object]] = []
    failed: list[Job] = []
    total = len(jobs)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_window, ticker, start, end_ex, pause): (ticker, start, end_ex)
            for ticker, start, end_ex in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            ticker, start, end_ex = job
            try:
                batch = future.result()
                ok_records.extend(batch)
            except Exception as exc:
                print(f"  Falha {ticker} {start}→{end_ex}: {exc}")
                failed.append(job)
                batch = []
            done += 1
            if done % 25 == 0 or done == total:
                print(
                    f"  [{done}/{total}] +{len(batch)} | ok acum: {len(ok_records)} | "
                    f"falhas: {len(failed)}"
                )
    return ok_records, failed


def collect_news() -> list[dict[str, object]]:
    windows = generate_weekly_windows(COLLECTION_START, COLLECTION_END)
    pending: list[Job] = [(ticker, start, end_ex) for ticker in TICKERS for start, end_ex in windows]
    total_jobs = len(pending)
    workers = RSS_MAX_WORKERS
    pause = float(RSS_PAUSE_SECONDS)
    print(
        f"Janelas: {len(windows)} semanas ({COLLECTION_START} a {COLLECTION_END}) | "
        f"{total_jobs} requests | começa com {workers} workers\n"
    )

    all_records: list[dict[str, object]] = []
    for round_idx in range(1, RSS_MAX_BACKOFF_ROUNDS + 1):
        print(f"--- Rodada {round_idx}: {len(pending)} jobs | {workers} workers | pause {pause:.1f}s ---")
        ok, failed = _run_batch(pending, workers, pause)
        all_records.extend(ok)

        if not failed:
            print("Todas as janelas ok.")
            break

        if round_idx == RSS_MAX_BACKOFF_ROUNDS:
            raise RuntimeError(
                f"Ainda falharam {len(failed)}/{total_jobs} jobs após {RSS_MAX_BACKOFF_ROUNDS} "
                f"rodadas de backoff (último: {workers} workers, pause {pause:.1f}s)."
            )

        next_workers, next_pause = _backoff(workers, pause)
        print(
            f"Falharam {len(failed)}. Cool-down {RSS_FAIL_COOLDOWN_SECONDS}s, "
            f"workers {workers}→{next_workers}, pause {pause:.1f}s→{next_pause:.1f}s\n"
        )
        time.sleep(RSS_FAIL_COOLDOWN_SECONDS)
        workers, pause = next_workers, next_pause
        pending = failed

    # dedupe por link_google (mantém a 1ª ocorrência)
    seen: set[str] = set()
    unique: list[dict[str, object]] = []
    for row in all_records:
        link = row.get("link_google")
        if not isinstance(link, str) or link in seen:
            continue
        seen.add(link)
        unique.append(row)

    for row in unique:
        value = row.get("data")
        row["data"] = str(value) if value is not None else None

    unique.sort(
        key=lambda r: _parse_rss_date(r.get("data") if isinstance(r.get("data"), str) else None),
        reverse=True,
    )
    return unique


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
    # ponytail: check mínimo do backoff (workers→1, pause sobe)
    assert _backoff(8, 2.0) == (4, 3.0)
    assert _backoff(2, 3.0) == (1, 4.5)
    assert _backoff(1, 4.5) == (1, 9.0)
    main()
