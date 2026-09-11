"""Coleta notícias B3 via Google News RSS."""
from __future__ import annotations

import json
import threading
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
        NEWS_JOBS_DONE,
        NEWS_RAW,
        RSS_CHECKPOINT_EVERY,
        RSS_COUNTRY,
        RSS_FAIL_COOLDOWN_SECONDS,
        RSS_LANGUAGE,
        RSS_MAX_BACKOFF_ROUNDS,
        RSS_MAX_CONSECUTIVE_503,
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
        NEWS_JOBS_DONE,
        NEWS_RAW,
        RSS_CHECKPOINT_EVERY,
        RSS_COUNTRY,
        RSS_FAIL_COOLDOWN_SECONDS,
        RSS_LANGUAGE,
        RSS_MAX_BACKOFF_ROUNDS,
        RSS_MAX_CONSECUTIVE_503,
        RSS_MAX_RETRIES,
        RSS_MAX_WORKERS,
        RSS_PAUSE_SECONDS,
        RSS_USER_AGENT,
        TICKERS,
    )

Job = tuple[str, date, date]
_HARD_FAIL_STATUSES = {403, 408, 429, 500, 502, 503, 504}


def _company_key(ticker: str) -> str:
    """Mesma regra do BL: primeiras 4 letras (ITUB3≈ITUB4, PETR3≈PETR4)."""
    letters = "".join(c for c in ticker.upper() if c.isalpha())
    return letters[:4] if len(letters) >= 4 else letters


def job_key(ticker: str, start: date, end_ex: date) -> str:
    return f"{ticker}|{start.isoformat()}|{end_ex.isoformat()}"


def tickers_one_per_company(configured: list[str]) -> list[str]:
    """Um ticker por empresa (PETR3 cobre PETR4 via company key no BL)."""
    planned: set[str] = set()
    out: list[str] = []
    for ticker in configured:
        key = _company_key(ticker)
        if key in planned:
            continue
        out.append(ticker)
        planned.add(key)
    return out


class FeedFetchError(RuntimeError):
    """Falha recuperável de fetch RSS (rate-limit, HTTP ruim, rede)."""


class RateLimitAbort(RuntimeError):
    """Muitos HTTP 503 seguidos — encerra a coleta após checkpoint."""


_consecutive_503 = 0
_503_lock = threading.Lock()


def _reset_503_streak() -> None:
    global _consecutive_503
    with _503_lock:
        _consecutive_503 = 0


def _note_fetch_ok() -> None:
    global _consecutive_503
    with _503_lock:
        _consecutive_503 = 0


def _note_http_503() -> None:
    global _consecutive_503
    with _503_lock:
        _consecutive_503 += 1
        n = _consecutive_503
    if n >= RSS_MAX_CONSECUTIVE_503:
        raise RateLimitAbort(
            f"{n} erros HTTP 503 seguidos (limite={RSS_MAX_CONSECUTIVE_503}) - "
            "salvando e finalizando."
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
                _note_fetch_ok()
                return feed
            if reason == "HTTP 503":
                _note_http_503()
            last_err = FeedFetchError(reason)
        except RateLimitAbort:
            raise
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


def load_json_list(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def load_jobs_done(path: Path = NEWS_JOBS_DONE) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(x) for x in data}
    return set()


def save_jobs_done(done: set[str], path: Path = NEWS_JOBS_DONE) -> None:
    path.write_text(
        json.dumps(sorted(done), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def merge_records(
    base: list[dict[str, object]],
    extra: list[dict[str, object]],
) -> list[dict[str, object]]:
    seen: set[str] = set()
    merged: list[dict[str, object]] = []
    for row in base + extra:
        link = row.get("link_google")
        if isinstance(link, str) and link:
            if link in seen:
                continue
            seen.add(link)
        merged.append(row)
    merged.sort(
        key=lambda r: _parse_rss_date(r.get("data") if isinstance(r.get("data"), str) else None),
        reverse=True,
    )
    return merged


def save_news(records: list[dict[str, object]], path: Path | None = None) -> Path:
    output = path or NEWS_RAW
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def seed_jobs_done_from_existing(
    existing: list[dict[str, object]],
    tickers: list[str],
    windows: list[tuple[date, date]],
) -> set[str]:
    """Se não há jobs_done, marca empresas já no JSON como 100% coletadas."""
    present = {_company_key(str(r.get("ticker") or "")) for r in existing}
    done: set[str] = set()
    for ticker in tickers:
        if _company_key(ticker) not in present:
            continue
        for start, end_ex in windows:
            done.add(job_key(ticker, start, end_ex))
    return done


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
    *,
    on_success=None,
) -> tuple[list[dict[str, object]], list[Job], bool]:
    """Executa jobs; devolve (records ok, jobs falhos/pendentes, aborted_por_503).
    on_success(job, batch) é chamado a cada janela ok (para checkpoint).
    """
    ok_records: list[dict[str, object]] = []
    failed: list[Job] = []
    total = len(jobs)
    done_n = 0
    aborted = False
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
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
                if on_success is not None:
                    on_success(job, batch)
            except RateLimitAbort as exc:
                print(f"  {exc}")
                failed.append(job)
                aborted = True
                for pending_f in futures:
                    pending_f.cancel()
                break
            except Exception as exc:
                print(f"  Falha {ticker} {start}->{end_ex}: {exc}")
                failed.append(job)
                batch = []
            done_n += 1
            if done_n % 25 == 0 or done_n == total:
                print(
                    f"  [{done_n}/{total}] +{len(batch)} | ok acum: {len(ok_records)} | "
                    f"falhas: {len(failed)}"
                )
        if aborted:
            # Janelas ainda não concluídas continuam pendentes para a próxima execução
            for fut, job in futures.items():
                if fut.cancelled() or not fut.done():
                    if job not in failed:
                        failed.append(job)
                elif fut.done() and not fut.cancelled():
                    # pode ter terminado com erro depois do abort — já tratado acima se veio no loop
                    pass
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return ok_records, failed, aborted


def collect_news(
    tickers: list[str] | None = None,
    *,
    jobs_done: set[str] | None = None,
    existing: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], set[str], list[Job]]:
    """
    Coleta janelas pendentes.

    Retorna (records_novos_da_sessão, jobs_done_atualizado, jobs_ainda_falhos).
    Faz checkpoint em NEWS_RAW a cada RSS_CHECKPOINT_EVERY sucessos.
    """
    tickers = tickers if tickers is not None else tickers_one_per_company(list(TICKERS))
    if not tickers:
        return [], jobs_done or set(), []

    windows = generate_weekly_windows(COLLECTION_START, COLLECTION_END)
    done = set(jobs_done or [])
    stored = list(existing or load_json_list(NEWS_RAW))
    session_new: list[dict[str, object]] = []
    pending: list[Job] = [
        (ticker, start, end_ex)
        for ticker in tickers
        for start, end_ex in windows
        if job_key(ticker, start, end_ex) not in done
    ]
    total_jobs = len(pending)
    workers = RSS_MAX_WORKERS
    pause = float(RSS_PAUSE_SECONDS)
    print(
        f"Tickers: {', '.join(tickers)} | "
        f"Janelas: {len(windows)} semanas ({COLLECTION_START} a {COLLECTION_END}) | "
        f"{total_jobs} pendentes / {len(done)} já ok | "
        f"começa com {workers} workers\n"
    )
    if not pending:
        return [], done, []

    _reset_503_streak()
    since_checkpoint = 0

    def checkpoint(force: bool = False) -> None:
        nonlocal since_checkpoint, stored
        if not force and since_checkpoint < RSS_CHECKPOINT_EVERY:
            return
        stored = merge_records(stored, session_new)
        # session_new já está em stored; limpa para não duplicar no próximo merge lógico
        session_new.clear()
        save_news(stored)
        save_jobs_done(done)
        since_checkpoint = 0
        print(f"  checkpoint: {len(stored)} noticias | {len(done)} jobs ok -> {NEWS_RAW.name}")

    def on_success(job: Job, batch: list[dict[str, object]]) -> None:
        nonlocal since_checkpoint
        ticker, start, end_ex = job
        done.add(job_key(ticker, start, end_ex))
        for row in batch:
            value = row.get("data")
            row["data"] = str(value) if value is not None else None
        session_new.extend(batch)
        since_checkpoint += 1
        checkpoint(force=False)

    final_failed: list[Job] = []
    try:
        for round_idx in range(1, RSS_MAX_BACKOFF_ROUNDS + 1):
            print(
                f"--- Rodada {round_idx}: {len(pending)} jobs | "
                f"{workers} workers | pause {pause:.1f}s ---"
            )
            _ok, failed, aborted = _run_batch(pending, workers, pause, on_success=on_success)
            checkpoint(force=True)

            if aborted:
                final_failed = failed
                print(
                    f"Rate-limit: checkpoint salvo. "
                    f"{len(done)} jobs ok | {len(failed)} pendentes para retomar depois."
                )
                break

            if not failed:
                print("Todas as janelas ok.")
                final_failed = []
                break

            final_failed = failed
            if round_idx == RSS_MAX_BACKOFF_ROUNDS:
                print(
                    f"Parcial: ainda falharam {len(failed)} jobs após "
                    f"{RSS_MAX_BACKOFF_ROUNDS} rodadas (salvo checkpoint)."
                )
                break

            next_workers, next_pause = _backoff(workers, pause)
            print(
                f"Falharam {len(failed)}. Cool-down {RSS_FAIL_COOLDOWN_SECONDS}s, "
                f"workers {workers}->{next_workers}, pause {pause:.1f}s->{next_pause:.1f}s\n"
            )
            time.sleep(RSS_FAIL_COOLDOWN_SECONDS)
            workers, pause = next_workers, next_pause
            pending = failed
    except KeyboardInterrupt:
        print("\nCtrl+C - salvando checkpoint...")
        checkpoint(force=True)
        raise

    checkpoint(force=True)
    return session_new, done, final_failed


def main() -> None:
    existing = load_json_list(NEWS_RAW)
    tickers = tickers_one_per_company(list(TICKERS))
    windows = generate_weekly_windows(COLLECTION_START, COLLECTION_END)

    if NEWS_JOBS_DONE.exists():
        done = load_jobs_done()
    else:
        done = seed_jobs_done_from_existing(existing, tickers, windows)
        if done:
            save_jobs_done(done)
            print(f"Seed jobs_done: {len(done)} janelas (empresas já no JSON).")

    pending_n = sum(
        1
        for t in tickers
        for start, end_ex in windows
        if job_key(t, start, end_ex) not in done
    )
    if pending_n == 0:
        print(f"Nada a coletar: {NEWS_RAW} + {NEWS_JOBS_DONE.name} cobrem o universo.")
        return

    print(f"Já no arquivo: {len(existing)} itens | jobs ok: {len(done)} | pendentes: {pending_n}\n")
    try:
        _new, done, failed = collect_news(tickers, jobs_done=done, existing=existing)
    except KeyboardInterrupt:
        print("Interrompido. Progresso já está em disco.")
        return

    final = load_json_list(NEWS_RAW)
    print(f"\nTotal em disco: {len(final)} | jobs ok: {len(done)} | ainda falhos: {len(failed)}")
    print(f"Salvo em: {NEWS_RAW}")
    if failed:
        print("Reexecute o script depois para retomar só as janelas falhas.")


if __name__ == "__main__":
    # ponytail: check mínimo do backoff / chave de job / circuit breaker 503
    assert _backoff(8, 2.0) == (4, 3.0)
    assert _backoff(2, 3.0) == (1, 4.5)
    assert _backoff(1, 4.5) == (1, 9.0)
    assert job_key("PETR3", date(2023, 1, 1), date(2023, 1, 8)) == "PETR3|2023-01-01|2023-01-08"
    assert tickers_one_per_company(["PETR3", "PETR4", "VALE3"]) == ["PETR3", "VALE3"]
    _reset_503_streak()
    for _ in range(RSS_MAX_CONSECUTIVE_503 - 1):
        _note_http_503()
    try:
        _note_http_503()
        raise AssertionError("deveria ter abortado no 503 consecutivo")
    except RateLimitAbort:
        pass
    _reset_503_streak()
    main()
