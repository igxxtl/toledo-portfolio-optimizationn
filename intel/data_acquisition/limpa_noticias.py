"""Remove notícias com títulos genéricos (sem menção ao ticker ou à empresa)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from .config import NEWS_DEDUPED, TICKER_TITLE_TERMS
except ImportError:
    from config import NEWS_DEDUPED, TICKER_TITLE_TERMS


def title_is_relevant(ticker: str, title: str) -> bool:
    if not title or not title.strip():
        return False
    title_lower = title.strip().lower()
    terms = TICKER_TITLE_TERMS.get(ticker, [ticker])
    return any(term.strip().lower() in title_lower for term in terms)


def clean_news(input_path: Path, output_path: Path | None = None) -> tuple[int, int]:
    """
    Remove itens com título genérico.

    Retorna (total_antes, total_depois). Se ``output_path`` for None, sobrescreve a entrada.
    """
    output_path = output_path or input_path
    data = json.loads(input_path.read_text(encoding="utf-8"))
    before = len(data)
    cleaned = [item for item in data if title_is_relevant(item["ticker"], item.get("titulo") or "")]
    output_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return before, len(cleaned)


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else NEWS_DEDUPED
    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        sys.exit(1)

    before, after = clean_news(path)
    print(f"Antes: {before} notícias")
    print(f"Depois: {after} notícias (removidas {before - after} com título genérico)")
    print(f"Salvo em: {path}")


if __name__ == "__main__":
    main()
