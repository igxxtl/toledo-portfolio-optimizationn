"""Configuração central da aquisição de notícias."""
from __future__ import annotations

from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[1]

# Arquivos JSON do pipeline de notícias
NEWS_RAW = DATA_DIR / "noticias_b3.json"
NEWS_DEDUPED = DATA_DIR / "noticias_b3_sem_duplicados.json"
NEWS_WITH_REAL_LINK = DATA_DIR / "noticias_b3_sem_duplicados_com_link_real.json"
NEWS_WITH_SENTIMENT = DATA_DIR / "noticias_b3_sem_duplicados_sentimento.json"

SEARCH_XML = DATA_DIR / "search.xml"
EXTRACT_DIR = DATA_DIR / "files"
EXTRACT_JSON = EXTRACT_DIR / "search_extract.json"
EXTRACT_JSON_SORTED = EXTRACT_DIR / "search_extract_ordenado.json"

# Coleta RSS
TICKERS = ["VALE3", "BBAS3", "ITUB3", "BBDC3", "ABEV3"]
COLLECTION_START = date(2023, 1, 1)
COLLECTION_END = date(2025, 12, 31)
RSS_LANGUAGE = "pt-BR"
RSS_COUNTRY = "BR"
RSS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
RSS_MAX_RETRIES = 4
RSS_PAUSE_SECONDS = 2
RSS_MAX_WORKERS = 8
RSS_FAIL_COOLDOWN_SECONDS = 15
RSS_MAX_BACKOFF_ROUNDS = 6

# Sentimento (OpenAI) — classificação 3 classes em volume → luna (tier nano da família 5.6)
SENTIMENT_MODEL = "gpt-5.6-luna"
SENTIMENT_MAX_WORKERS = 8

# Termos para filtrar títulos genéricos (limpa_noticias)
TICKER_TITLE_TERMS: dict[str, list[str]] = {
    "VALE3": ["VALE3", "Vale"],
    "BBAS3": ["BBAS3", "Banco do Brasil"],
    "ITUB3": ["ITUB3", "Itaú", "Itau"],
    "BBDC3": ["BBDC3", "Bradesco"],
    "ABEV3": ["ABEV3", "Ambev"],
}
