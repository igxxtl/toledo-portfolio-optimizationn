"""Remove duplicatas de notícias pelo título."""
from __future__ import annotations

import json
from pathlib import Path

try:
    from .config import NEWS_DEDUPED, NEWS_RAW
except ImportError:
    from config import NEWS_DEDUPED, NEWS_RAW


def normalize_title(title: str) -> str:
    if not title:
        return ""
    return " ".join(title.strip().split())


def deduplicate_news(
    input_path: Path = NEWS_RAW,
    output_path: Path = NEWS_DEDUPED,
) -> tuple[int, int]:
    """Remove entradas com título duplicado. Retorna (total_antes, total_depois)."""
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

    news = json.loads(input_path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    deduped: list[dict[str, object]] = []

    for item in news:
        title_key = normalize_title(item.get("titulo") or "")
        if not title_key:
            deduped.append(item)
            continue
        if title_key in seen:
            continue
        seen.add(title_key)
        deduped.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(news), len(deduped)


def main() -> None:
    before, after = deduplicate_news()
    print(f"Entradas originais: {before}")
    print(f"Duplicatas removidas (mesmo título): {before - after}")
    print(f"Entradas no novo JSON: {after}")
    print(f"Salvo em: {NEWS_DEDUPED}")


if __name__ == "__main__":
    main()
