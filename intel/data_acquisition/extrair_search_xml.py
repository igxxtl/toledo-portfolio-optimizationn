"""Extrai título, link e data de publicação de search.xml (RSS/Atom)."""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    from .config import EXTRACT_DIR, EXTRACT_JSON, SEARCH_XML
except ImportError:
    from config import EXTRACT_DIR, EXTRACT_JSON, SEARCH_XML

EXTRACT_CSV = EXTRACT_DIR / "search_extract.csv"


def _strip_namespace(tag: str) -> str:
    if tag and "}" in tag:
        return tag.split("}", 1)[1]
    return tag or ""


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return (element.text or "").strip() + "".join((child.tail or "").strip() for child in element)


def _link_from_item(item: ET.Element) -> str:
    for child in item:
        if _strip_namespace(child.tag).lower() == "link":
            href = child.get("href")
            if href:
                return href.strip()
            return _element_text(child).strip()
    return ""


def extract_items(xml_path: Path) -> list[dict[str, Any]]:
    with xml_path.open(encoding="utf-8", errors="replace") as handle:
        root = ET.parse(handle).getroot()

    if root is None:
        return []

    tag_root = _strip_namespace(root.tag).lower()
    if tag_root == "rss":
        channel = root.find("channel")
        items = list(channel.findall("item")) if channel is not None else []
    elif tag_root == "feed":
        items = list(root.findall("entry"))
    else:
        items = [elem for elem in root.iter() if _strip_namespace(elem.tag).lower() in ("item", "entry")]

    results: list[dict[str, Any]] = []
    for item in items:
        title = link = pub_date = ""
        for child in item:
            name = _strip_namespace(child.tag).lower()
            if name == "title":
                title = _element_text(child)
            elif name == "link":
                link = child.get("href") or _element_text(child)
            elif name in ("pubdate", "published", "updated"):
                pub_date = _element_text(child)

        if not link:
            link = _link_from_item(item)

        results.append({"titulo": title, "link": link, "data_publicacao": pub_date.strip()})

    return results


def save_extract(items: list[dict[str, Any]]) -> tuple[Path, Path]:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with EXTRACT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["titulo", "link", "data_publicacao"])
        writer.writeheader()
        writer.writerows(items)
    EXTRACT_JSON.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return EXTRACT_CSV, EXTRACT_JSON


def main() -> None:
    if not SEARCH_XML.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {SEARCH_XML}")

    items = extract_items(SEARCH_XML)
    csv_path, json_path = save_extract(items)
    print(f"Extraídos {len(items)} itens de {SEARCH_XML}")
    print(f"CSV salvo: {csv_path}")
    print(f"JSON salvo: {json_path}")


if __name__ == "__main__":
    main()
