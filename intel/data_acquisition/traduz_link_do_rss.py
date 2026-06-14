"""Resolve links reais de notícias a partir de link_google no JSON."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright

try:
    from .config import NEWS_DEDUPED, NEWS_WITH_REAL_LINK
except ImportError:
    from config import NEWS_DEDUPED, NEWS_WITH_REAL_LINK


def resolve_google_news_url(page, url: str, max_retries: int = 3) -> str:
    if not url:
        return ""

    for attempt in range(1, max_retries + 1):
        try:
            page.goto(url, timeout=60_000, wait_until="domcontentloaded")
            page.wait_for_timeout(1200)
            return page.url
        except Exception as exc:
            if attempt == max_retries:
                print(f"  Falha para URL: {url[:120]}... ({exc})")
                return ""
            wait = 2**attempt
            print(f"  Erro ao resolver URL, retry em {wait}s ({attempt}/{max_retries})")
            time.sleep(wait)
    return ""


def process_json(
    input_path: Path,
    output_path: Path,
    *,
    google_link_field: str = "link_google",
    real_link_field: str = "link_real",
) -> None:
    data: list[dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))
    total = len(data)
    resolved = 0

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_context().new_page()
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ["image", "stylesheet", "font", "media"]
            else route.continue_(),
        )

        for idx, item in enumerate(data, start=1):
            google_url = (item.get(google_link_field) or "").strip()
            if not google_url:
                item[real_link_field] = ""
            elif not item.get(real_link_field):
                item[real_link_field] = resolve_google_news_url(page, google_url)
                if item[real_link_field]:
                    resolved += 1

            if idx % 50 == 0 or idx == total:
                print(f"[{idx}/{total}] processadas | links resolvidos: {resolved}")
                output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        browser.close()

    print(f"\nConcluído. Total de notícias: {total}")
    print(f"Links reais resolvidos: {resolved}")
    print(f"Arquivo salvo em: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve link real a partir de link_google no JSON.")
    parser.add_argument("--input", default=str(NEWS_DEDUPED), help="Arquivo JSON de entrada.")
    parser.add_argument("--output", default=str(NEWS_WITH_REAL_LINK), help="Arquivo JSON de saída.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_path}")

    process_json(input_path=input_path, output_path=output_path)


if __name__ == "__main__":
    main()
