import argparse
import json
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


def resolver_url_google_news(page, url: str, max_tentativas: int = 3) -> str:
    """Abre o link do Google News e retorna a URL final (site da notícia)."""
    if not url:
        return ""

    for tentativa in range(1, max_tentativas + 1):
        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_timeout(1200)
            return page.url
        except Exception as exc:
            if tentativa == max_tentativas:
                print(f"  x Falha para URL: {url[:120]}... ({exc})")
                return ""
            espera = 2 ** tentativa
            print(f"  ! Erro ao resolver URL, retry em {espera}s ({tentativa}/{max_tentativas})")
            time.sleep(espera)
    return ""


def processar_json(
    arquivo_entrada: Path,
    arquivo_saida: Path,
    campo_link_google: str = "link_google",
    campo_link_real: str = "link_real",
) -> None:
    dados: list[dict[str, Any]] = json.loads(arquivo_entrada.read_text(encoding="utf-8"))
    total = len(dados)
    resolvidos = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Bloqueia recursos pesados para ficar mais rápido
        page.route(
            "**/*",
            lambda route: route.abort()
            if route.request.resource_type in ["image", "stylesheet", "font", "media"]
            else route.continue_(),
        )

        for idx, item in enumerate(dados, start=1):
            url_google = (item.get(campo_link_google) or "").strip()
            if not url_google:
                item[campo_link_real] = ""
            elif item.get(campo_link_real):
                # já preenchido
                pass
            else:
                item[campo_link_real] = resolver_url_google_news(page, url_google)
                if item[campo_link_real]:
                    resolvidos += 1

            if idx % 50 == 0 or idx == total:
                print(f"[{idx}/{total}] processadas | links resolvidos: {resolvidos}")
                arquivo_saida.write_text(
                    json.dumps(dados, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

        browser.close()

    print(f"\nConcluído. Total de notícias: {total}")
    print(f"Links reais resolvidos: {resolvidos}")
    print(f"Arquivo salvo em: {arquivo_saida}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve link real de notícias a partir de link_google no JSON."
    )
    raiz = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--input",
        default=str(raiz / "noticias_b3_sem_duplicados.json"),
        help="Arquivo JSON de entrada.",
    )
    parser.add_argument(
        "--output",
        default=str(raiz / "noticias_b3_sem_duplicados_com_link_real.json"),
        help="Arquivo JSON de saída.",
    )
    args = parser.parse_args()

    arquivo_entrada = Path(args.input)
    arquivo_saida = Path(args.output)

    if not arquivo_entrada.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {arquivo_entrada}")

    processar_json(arquivo_entrada=arquivo_entrada, arquivo_saida=arquivo_saida)


if __name__ == "__main__":
    main()