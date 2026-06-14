"""Ordena notícias extraídas de XML por data de publicação."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .config import EXTRACT_JSON, EXTRACT_JSON_SORTED
except ImportError:
    from config import EXTRACT_JSON, EXTRACT_JSON_SORTED


def parse_publication_date(raw: Any) -> datetime | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, datetime):
        return raw

    text = str(raw).strip()
    try:
        return datetime.strptime(text[:25], "%a, %d %b %Y %H:%M:%S")
    except ValueError:
        pass

    text = text.replace("Z", "").replace("+00:00", "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text[:19] if len(text) >= 19 else text[:10], fmt)
        except ValueError:
            continue
    return None


def sort_news_by_date(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> int:
    src = input_path or EXTRACT_JSON
    dst = output_path or EXTRACT_JSON_SORTED

    if not src.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {src}")

    news: list[dict[str, Any]] = json.loads(src.read_text(encoding="utf-8"))

    def sort_key(item: dict[str, Any]) -> datetime:
        parsed = parse_publication_date(item.get("data_publicacao"))
        return parsed if parsed else datetime.min

    sorted_news = sorted(news, key=sort_key)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(sorted_news, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(sorted_news)


def main() -> None:
    count = sort_news_by_date()
    print(f"Lidas {count} notícias de {EXTRACT_JSON}")
    print(f"Salvas em ordem ascendente em {EXTRACT_JSON_SORTED}")


if __name__ == "__main__":
    main()
