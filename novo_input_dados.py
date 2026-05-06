from __future__ import annotations

"""
novo_input_dados.py
-------------------
Lê os tickers da pasta tickers_data, busca dados DIÁRIOS via yfinance
e salva em dados_diarios/<TICKER>_diario.csv

Tickers detectados automaticamente a partir dos CSVs existentes.
Para B3: adiciona sufixo ".SA" na busca do yfinance.

Também baixa o Ibovespa (^BVSP) e salva em dados_diarios/IBOV_diario.csv.

Limite yfinance - dados diários: sem limite prático (vai até o IPO).
Por padrão busca desde 2018-01-01 até hoje.
"""

import sys
from pathlib import Path
import re

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
TICKERS_DIR = ROOT / "tickers_data"
OUT_DIR = ROOT / "dados_diarios"
OUT_DIR.mkdir(exist_ok=True)

START_DATE = "2018-01-01"
END_DATE = None          # None = hoje

COLUNAS_SAIDA = ["date", "open", "high", "low", "close", "volume", "ticker"]


def extrair_tickers(tickers_dir: Path) -> list[str]:
    """Extrai o nome do ativo a partir do padrão BMFBOVESPA_DLY_<TICKER>, 60.csv"""
    tickers = []
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

        # yfinance pode retornar MultiIndex nas colunas
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0].lower() if col[1] == "" else col[0].lower() for col in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        df = df.rename(columns={"date": "date", "vol": "volume"})

        date_col = "date" if "date" in df.columns else df.columns[0]
        df["date"] = pd.to_datetime(df[date_col]).dt.normalize()
        df["ticker"] = ticker

        cols_present = [c for c in COLUNAS_SAIDA if c in df.columns]
        return df[cols_present].sort_values("date").reset_index(drop=True)

    except Exception as exc:
        print(f"  [{ticker}] erro ao baixar: {exc}")
        return None


def baixar_ibov(start: str, end: str | None) -> pd.DataFrame | None:
    """Baixa o Ibovespa (^BVSP) e retorna DataFrame no mesmo formato dos ativos."""
    try:
        df = yf.download("^BVSP", start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False)
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

        cols_present = [c for c in COLUNAS_SAIDA if c in df.columns]
        return df[cols_present].sort_values("date").reset_index(drop=True)

    except Exception as exc:
        print(f"  [IBOV] erro ao baixar: {exc}")
        return None


def main() -> None:
    tickers = extrair_tickers(TICKERS_DIR)
    if not tickers:
        print(f"Nenhum ticker encontrado em {TICKERS_DIR}")
        sys.exit(1)

    print(f"Tickers encontrados ({len(tickers)}): {', '.join(tickers)}")
    print(f"Periodo: {START_DATE} → {'hoje' if END_DATE is None else END_DATE}")
    print(f"Salvando em: {OUT_DIR}\n")

    resumo = []

    # ── Ibovespa ──────────────────────────────────────────────────────────
    print("[IBOV] baixando ^BVSP ...", end=" ", flush=True)
    df_ibov = baixar_ibov(START_DATE, END_DATE)
    if df_ibov is not None:
        ibov_path = OUT_DIR / "IBOV.csv"
        df_ibov.to_csv(ibov_path, index=False)
        print(f"{len(df_ibov)} dias → {ibov_path.name}")
        resumo.append({"ticker": "IBOV", "linhas": len(df_ibov), "status": "ok"})
    else:
        resumo.append({"ticker": "IBOV", "linhas": 0, "status": "falhou"})

    # ── ativos ────────────────────────────────────────────────────────────
    for ticker in tickers:
        print(f"[{ticker}] baixando ...", end=" ", flush=True)
        df = baixar_diario(ticker, START_DATE, END_DATE)
        if df is None:
            resumo.append({"ticker": ticker, "linhas": 0, "status": "falhou"})
            continue

        out_path = OUT_DIR / f"{ticker}.csv"
        df.to_csv(out_path, index=False)
        print(f"{len(df)} dias → {out_path.name}")
        resumo.append({"ticker": ticker, "linhas": len(df), "status": "ok"})

    print("\n── Resumo ──────────────────────────────────────")
    resumo_df = pd.DataFrame(resumo)
    print(resumo_df.to_string(index=False))

    ok = resumo_df[resumo_df["status"] == "ok"]
    print(f"\n{len(ok)}/{len(tickers) + 1} séries coletadas com sucesso.")


if __name__ == "__main__":
    main()
