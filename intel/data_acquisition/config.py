"""Configuração central da aquisição de notícias."""
from __future__ import annotations

from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parents[1]

# Arquivos JSON do pipeline de notícias
NEWS_RAW = DATA_DIR / "noticias_b3.json"
NEWS_JOBS_DONE = DATA_DIR / "noticias_b3_jobs_done.json"
NEWS_DEDUPED = DATA_DIR / "noticias_b3_sem_duplicados.json"
NEWS_WITH_REAL_LINK = DATA_DIR / "noticias_b3_sem_duplicados_com_link_real.json"
NEWS_WITH_SENTIMENT = DATA_DIR / "noticias_b3_sem_duplicados_sentimento.json"

SEARCH_XML = DATA_DIR / "search.xml"
EXTRACT_DIR = DATA_DIR / "files"
EXTRACT_JSON = EXTRACT_DIR / "search_extract.json"
EXTRACT_JSON_SORTED = EXTRACT_DIR / "search_extract_ordenado.json"

# Coleta RSS — alinhado aos tickers com modelo XGB (BL usa .SA)
TICKERS = [
    "ABEV3",
    "B3SA3",
    "BBAS3",
    "BBDC4",
    "CMIG4",
    "EQTL3",
    "GGBR4",
    "ITUB4",
    "LREN3",
    "MGLU3",
    "PETR3",
    "PETR4",
    "PRIO3",
    "RADL3",
    "RAIL3",
    "SANB11",
    "SBSP3",
    "SUZB3",
    "VALE3",
    "WEGE3",
]
COLLECTION_START = date(2023, 1, 1)
COLLECTION_END = date(2025, 12, 31)
RSS_LANGUAGE = "pt-BR"
RSS_COUNTRY = "BR"
RSS_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
RSS_MAX_RETRIES = 4
# 8 workers gerava 503 em massa no Google News; 2 + pause maior é mais lento e estável
RSS_PAUSE_SECONDS = 4
RSS_MAX_WORKERS = 2
RSS_FAIL_COOLDOWN_SECONDS = 30
RSS_MAX_BACKOFF_ROUNDS = 6
RSS_CHECKPOINT_EVERY = 25
# Circuit breaker: N respostas HTTP 503 seguidas → checkpoint e encerra (retoma depois)
RSS_MAX_CONSECUTIVE_503 = 12

# Sentimento (OpenAI) — classificação 3 classes em volume → luna (tier nano da família 5.6)
SENTIMENT_MODEL = "gpt-5.6-luna"
SENTIMENT_MAX_WORKERS = 8

# Termos para filtrar títulos genéricos (limpa_noticias)
TICKER_TITLE_TERMS: dict[str, list[str]] = {
    "ABEV3": ["ABEV3", "Ambev"],
    "B3SA3": ["B3SA3", "B3", "Bovespa"],
    "BBAS3": ["BBAS3", "Banco do Brasil"],
    "BBDC4": ["BBDC4", "BBDC3", "Bradesco"],
    "CMIG4": ["CMIG4", "CMIG3", "Cemig"],
    "EQTL3": ["EQTL3", "Equatorial"],
    "GGBR4": ["GGBR4", "GGBR3", "Gerdau"],
    "ITUB4": ["ITUB4", "ITUB3", "Itaú", "Itau"],
    "LREN3": ["LREN3", "Renner"],
    "MGLU3": ["MGLU3", "Magazine Luiza", "Magalu"],
    "PETR3": ["PETR3", "PETR4", "Petrobras"],
    "PETR4": ["PETR4", "PETR3", "Petrobras"],
    "PRIO3": ["PRIO3", "PRIO", "PetroRio", "Petro Rio"],
    "RADL3": ["RADL3", "Raia Drogasil", "Drogasil", "Raia"],
    "RAIL3": ["RAIL3", "Rumo"],
    "SANB11": ["SANB11", "SANB4", "SANB3", "Santander"],
    "SBSP3": ["SBSP3", "Sabesp"],
    "SUZB3": ["SUZB3", "Suzano"],
    "VALE3": ["VALE3", "Vale"],
    "WEGE3": ["WEGE3", "WEG", "Weg"],
    # legado (coleta antiga ON) — limpa_noticias ainda encontra esses tickers no JSON
    "ITUB3": ["ITUB3", "ITUB4", "Itaú", "Itau"],
    "BBDC3": ["BBDC3", "BBDC4", "Bradesco"],
}
