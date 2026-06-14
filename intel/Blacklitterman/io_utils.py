"""Utilitários de leitura/escrita de artefatos CSV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

__all__ = [
    "asset_columns",
    "read_csv_with_date",
    "save_csv_with_date",
]


def asset_columns(df: pd.DataFrame) -> list[str]:
    """Retorna colunas de tickers B3 (sufixo ``.SA``)."""
    return [c for c in df.columns if c.endswith(".SA")]


def read_csv_with_date(path: Path, *, date_col: str = "view_date") -> pd.DataFrame:
    """Lê CSV e normaliza coluna de data para ``datetime64``."""
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)


def save_csv_with_date(df: pd.DataFrame, path: Path, *, date_col: str = "view_date") -> None:
    """Salva DataFrame com coluna de data formatada como ``YYYY-MM-DD``."""
    out = df.copy()
    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col]).dt.strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
