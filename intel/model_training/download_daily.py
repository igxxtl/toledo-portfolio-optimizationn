"""Baixa OHLCV diário via yfinance a partir dos tickers em ``tickers_data/``."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from .config import (
        DAILY_COLUMNS,
        DAILY_END_DATE,
        DAILY_START_DATE,
        DATA_DAILY_DIR,
        TICKERS_DATA_DIR,
    )
except ImportError:
    from config import (
        DAILY_COLUMNS,
        DAILY_END_DATE,
        DAILY_START_DATE,
        DATA_DAILY_DIR,
        TICKERS_DATA_DIR,
    )


def extrair_tickers(tickers_dir: Path) -> list[str]:
    """Extrai o nome do ativo a partir do padrão ``BMFBOVESPA_DLY_<TICKER>, 60.csv``."""
    tickers: list[str] = []
    for fp in sorted(tickers_dir.glob("*.csv")):
        match = re.search(r"BMFBOVESPA_DLY_(.+?),", fp.name)
        if match:
            tickers.append(match.group(1).strip())
    return tickers


def baixar_diario(ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
    symbol = f"{ticker}.SA"
    try:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            print(f"  [{ticker}] sem dados retornados.")
            return None

        df = df.reset_index()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if col[1] == "" else col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        df = df.rename(columns={"date": "date", "vol": "volume"})

        date_col = "date" if "date" in df.columns else df.columns[0]
        df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
        df["ticker"] = ticker

        cols_present = [c for c in DAILY_COLUMNS if c in df.columns]
        return df[cols_present].sort_values("date").reset_index(drop=True)

    except Exception as exc:
        print(f"  [{ticker}] erro ao baixar: {exc}")
        return None


def baixar_ibov(start: str, end: str | None) -> pd.DataFrame | None:
    """Baixa o Ibovespa (^BVSP) e retorna DataFrame no mesmo formato dos ativos."""
    try:
        df = yf.download(
            "^BVSP",
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            print("  [IBOV] sem dados retornados.")
            return None

        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if col[1] == "" else col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        date_col = "date" if "date" in df.columns else df.columns[0]
        df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
        df["ticker"] = "IBOV"

        cols_present = [c for c in DAILY_COLUMNS if c in df.columns]
        return df[cols_present].sort_values("date").reset_index(drop=True)

    except Exception as exc:
        print(f"  [IBOV] erro ao baixar: {exc}")
        return None


def main() -> None:
    DATA_DAILY_DIR.mkdir(parents=True, exist_ok=True)

    tickers = extrair_tickers(TICKERS_DATA_DIR)
    if not tickers:
        print(f"Nenhum ticker encontrado em {TICKERS_DATA_DIR}")
        sys.exit(1)

    print(f"Tickers encontrados ({len(tickers)}): {', '.join(tickers)}")
    print(f"Periodo: {DAILY_START_DATE} -> {'hoje' if DAILY_END_DATE is None else DAILY_END_DATE}")
    print(f"Salvando em: {DATA_DAILY_DIR}\n")

    resumo: list[dict[str, object]] = []

    print("[IBOV] baixando ^BVSP ...", end=" ", flush=True)
    df_ibov = baixar_ibov(DAILY_START_DATE, DAILY_END_DATE)
    if df_ibov is not None:
        ibov_path = DATA_DAILY_DIR / "IBOV.csv"
        df_ibov.to_csv(ibov_path, index=False)
        print(f"{len(df_ibov)} dias -> {ibov_path.name}")
        resumo.append({"ticker": "IBOV", "linhas": len(df_ibov), "status": "ok"})
    else:
        resumo.append({"ticker": "IBOV", "linhas": 0, "status": "falhou"})

    for ticker in tickers:
        print(f"[{ticker}] baixando ...", end=" ", flush=True)
        df = baixar_diario(ticker, DAILY_START_DATE, DAILY_END_DATE)
        if df is None:
            resumo.append({"ticker": ticker, "linhas": 0, "status": "falhou"})
            continue

        out_path = DATA_DAILY_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f"{len(df)} dias -> {out_path.name}")
        resumo.append({"ticker": ticker, "linhas": len(df), "status": "ok"})

    print("\n-- Resumo --")
    resumo_df = pd.DataFrame(resumo)
    print(resumo_df.to_string(index=False))

    ok = resumo_df[resumo_df["status"] == "ok"]
    print(f"\n{len(ok)}/{len(tickers) + 1} series coletadas com sucesso.")


if __name__ == "__main__":
    main()
