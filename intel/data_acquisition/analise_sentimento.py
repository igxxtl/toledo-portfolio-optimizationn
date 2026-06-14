"""Classifica sentimento de manchetes via OpenAI."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .config import NEWS_DEDUPED, NEWS_WITH_SENTIMENT, SENTIMENT_MAX_WORKERS, SENTIMENT_MODEL
except ImportError:
    from config import NEWS_DEDUPED, NEWS_WITH_SENTIMENT, SENTIMENT_MAX_WORKERS, SENTIMENT_MODEL

PROMPT_TEMPLATE = """
Classifique o sentimento da manchete abaixo em relação ao impacto no preço da ação {ticker}.

Responda apenas com uma palavra:

positivo
negativo
neutro

Manchete:
{headline}
"""


def classify_headline(client: OpenAI, ticker: str, title: str) -> str:
    if not title or not title.strip():
        return "neutro"

    prompt = PROMPT_TEMPLATE.format(ticker=ticker, headline=title)
    response = client.responses.create(
        model=SENTIMENT_MODEL,
        input=prompt,
        temperature=1,
    )
    return (getattr(response, "output_text", None) or "").strip()


def run_sentiment_analysis(
    input_path: Path = NEWS_DEDUPED,
    output_path: Path = NEWS_WITH_SENTIMENT,
) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY não definida no ambiente ou .env")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    unique: dict[str, tuple[str, str]] = {}
    for item in data:
        ticker = (item.get("ticker") or "").strip()
        title = (item.get("titulo") or "").strip()
        key = f"{ticker}||{title}"
        if key not in unique:
            unique[key] = (ticker, title)

    cache: dict[str, str] = {}
    client = OpenAI(api_key=api_key)
    total_unique = len(unique)

    with ThreadPoolExecutor(max_workers=SENTIMENT_MAX_WORKERS) as executor:
        futures = {
            executor.submit(classify_headline, client, ticker, title): key
            for key, (ticker, title) in unique.items()
        }
        done = 0
        for future in as_completed(futures):
            key = futures[future]
            try:
                cache[key] = future.result()
            except Exception as exc:
                print(f"Erro ao classificar {key[:50]}...: {exc}")
                cache[key] = "neutro"
            done += 1
            if done % 50 == 0 or done == total_unique:
                print(f"[{done}/{total_unique}] manchetes únicas classificadas")

    for item in data:
        ticker = (item.get("ticker") or "").strip()
        title = (item.get("titulo") or "").strip()
        key = f"{ticker}||{title}"
        item["sentimento"] = cache.get(key, "neutro")

    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Concluído. {len(data)} itens processados. Salvo em: {output_path}")


def main() -> None:
    run_sentiment_analysis()


if __name__ == "__main__":
    main()
