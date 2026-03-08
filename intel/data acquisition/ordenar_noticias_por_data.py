"""
Lê files/search_extract.json e reorganiza as notícias por data de publicação
em ordem ascendente (mais antigas primeiro). Salva o resultado em novo arquivo.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ARQUIVO_ENTRADA = Path(__file__).parent / "files" / "search_extract.json"
ARQUIVO_SAIDA = Path(__file__).parent / "files" / "search_extract_ordenado.json"


def parse_data_publicacao(raw: Any) -> Optional[datetime]:
    """
    Converte data_publicacao para datetime para ordenação.
    Aceita formato RFC 2822 (ex: "Mon, 26 Jan 2026 08:00:00 GMT") e ISO.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, datetime):
        return raw

    s = str(raw).strip()
    # RFC 2822 (Google RSS): "Mon, 26 Jan 2026 08:00:00 GMT"
    try:
        return datetime.strptime(s[:25], "%a, %d %b %Y %H:%M:%S")
    except ValueError:
        pass
    # ISO
    s = s.replace("Z", "").replace("+00:00", "").strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19] if len(s) >= 19 else s[:10], fmt)
        except ValueError:
            continue
    return None


def main() -> None:
    if not ARQUIVO_ENTRADA.exists():
        print(f"Arquivo não encontrado: {ARQUIVO_ENTRADA}")
        return

    with open(ARQUIVO_ENTRADA, "r", encoding="utf-8") as f:
        noticias: List[Dict[str, Any]] = json.load(f)

    # Ordenar por data de publicação ascendente (mais antigas primeiro)
    # Itens sem data vão para o início (tratamos como muito antigos)
    def chave_ordenacao(item: Dict[str, Any]) -> datetime:
        dt = parse_data_publicacao(item.get("data_publicacao"))
        return dt if dt else datetime.min

    noticias_ordenadas = sorted(noticias, key=chave_ordenacao)

    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        json.dump(noticias_ordenadas, f, ensure_ascii=False, indent=2)

    print(f"Lidas {len(noticias)} notícias de {ARQUIVO_ENTRADA}")
    print(f"Salvas em ordem ascendente (data de publicação) em {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    main()
