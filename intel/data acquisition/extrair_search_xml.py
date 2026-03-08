"""
Extrai título, link e data de publicação de search.xml (RSS/Atom).
Salva em CSV e JSON.
"""

import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

# Ajuste o caminho se o XML estiver em outro lugar
ARQUIVO_XML = Path(__file__).parent / "search.xml"
SAIDA_CSV = Path(__file__).parent / "files" / "search_extract.csv"
SAIDA_JSON = Path(__file__).parent / "files" / "search_extract.json"


def strip_ns(tag: str) -> str:
    """Remove namespace do nome da tag (ex: {http://...}item -> item)."""
    if tag and "}" in tag:
        return tag.split("}", 1)[1]
    return tag or ""


def get_text(el: ET.Element | None) -> str:
    """Retorna o texto do elemento (incluindo texto direto)."""
    if el is None:
        return ""
    return (el.text or "").strip() + "".join(
        (e.tail or "").strip() for e in el
    ).strip()


def get_link_from_item(item: ET.Element) -> str:
    """Extrai URL do item. RSS usa <link>; Atom usa <link href="..."/>."""
    for child in item:
        if strip_ns(child.tag).lower() == "link":
            href = child.get("href")
            if href:
                return href.strip()
            return get_text(child).strip()
    return ""


def parse_pubdate(raw: str) -> str:
    """Mantém a data como string; opcionalmente normaliza formato."""
    if not raw or not raw.strip():
        return ""
    return raw.strip()


def extrair_itens(xml_path: Path) -> List[Dict[str, Any]]:
    """Lê o XML e extrai título, link e data de publicação de cada item/entry."""
    with open(xml_path, "r", encoding="utf-8", errors="replace") as f:
        tree = ET.parse(f)
    root = tree.getroot()

    # RSS: root é <rss>, itens em channel/item
    # Atom: root é <feed>, itens em entry
    items: List[ET.Element] = []

    if root is None:
        return []

    tag_root = strip_ns(root.tag).lower()
    if tag_root == "rss":
        channel = root.find("channel")
        if channel is not None:
            items = list(channel.findall("item"))
    elif tag_root == "feed":
        items = list(root.findall("entry"))
    else:
        # Fallback: procurar qualquer item ou entry
        for elem in root.iter():
            if strip_ns(elem.tag).lower() in ("item", "entry"):
                items.append(elem)

    resultados: List[Dict[str, Any]] = []
    for item in items:
        title = ""
        link = ""
        pub_date = ""

        for child in item:
            name = strip_ns(child.tag).lower()
            if name == "title":
                title = get_text(child)
            elif name == "link":
                link = child.get("href") or get_text(child)
            elif name in ("pubdate", "published", "updated"):
                pub_date = get_text(child)

        if not link:
            link = get_link_from_item(item)

        resultados.append({
            "titulo": title,
            "link": link,
            "data_publicacao": parse_pubdate(pub_date),
        })

    return resultados


def main() -> None:
    if not ARQUIVO_XML.exists():
        print(f"Arquivo não encontrado: {ARQUIVO_XML}")
        return

    itens = extrair_itens(ARQUIVO_XML)
    print(f"Extraídos {len(itens)} itens de {ARQUIVO_XML}")

    SAIDA_CSV.parent.mkdir(parents=True, exist_ok=True)

    # CSV
    with open(SAIDA_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["titulo", "link", "data_publicacao"])
        w.writeheader()
        w.writerows(itens)
    print(f"CSV salvo: {SAIDA_CSV}")

    # JSON
    with open(SAIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(itens, f, ensure_ascii=False, indent=2)
    print(f"JSON salvo: {SAIDA_JSON}")


if __name__ == "__main__":
    main()
