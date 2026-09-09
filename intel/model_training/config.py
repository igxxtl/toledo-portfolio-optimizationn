"""Configuração central do pipeline de treino XGBoost (vetor Q)."""
from __future__ import annotations

from pathlib import Path

MT_DIR = Path(__file__).resolve().parent
REPO_ROOT = MT_DIR.parents[1]

TICKERS_DATA_DIR = REPO_ROOT / "tickers_data"
DATA_DAILY_DIR = REPO_ROOT / "dados_diarios"
XGB_OUTPUT_DIR = REPO_ROOT / "criacao_modelo_xgb"

# Download diário (yfinance)
DAILY_START_DATE = "2018-01-01"
DAILY_END_DATE = None  # None = hoje

DAILY_COLUMNS = ["date", "open", "high", "low", "close", "volume", "ticker"]

# Features relativas ao Ibovespa (prepare_features.py)
IBOV_FILE = DATA_DAILY_DIR / "IBOV.csv"
FEATURE_WINDOWS: list[int] = [3, 5, 14, 21, 42, 63]
