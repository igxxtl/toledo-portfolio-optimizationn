"""
Remove do JSON de notícias os itens com títulos genéricos (sem menção ao ticker ou à empresa).
Mantém apenas notícias em que o título contém o ticker ou um dos nomes associados à empresa.
"""

import json
import sys
from pathlib import Path

# Ticker -> termos que, se aparecerem no título, indicam que a notícia é sobre a empresa
# (ticker exato + nome da empresa / variações comuns)
TICKER_TERMOS: dict[str, list[str]] = {
    "VALE3": ["VALE3", "Vale"],
    "BBAS3": ["BBAS3", "Banco do Brasil"],
    "ITUB3": ["ITUB3", "Itaú", "Itau"],
    "BBDC3": ["BBDC3", "Bradesco"],
    "ABEV3": ["ABEV3", "Ambev"],
}


def titulo_relevante(ticker: str, titulo: str) -> bool:
    """True se o título contém o ticker ou algum termo associado à empresa."""
    if not titulo or not titulo.strip():
        return False
    titulo_lower = titulo.strip().lower()
    termos = TICKER_TERMOS.get(ticker, [ticker])
    for t in termos:
        if t.strip().lower() in titulo_lower:
            return True
    return False


def limpar_noticias(entrada: Path, saida: Path | None = None) -> tuple[int, int]:
    """
    Lê o JSON de notícias, remove itens com título genérico (sem ticker/empresa no título).
    Retorna (total_antes, total_depois).
    Se saida for None, sobrescreve o arquivo de entrada.
    """
    saida = saida or entrada
    with open(entrada, encoding="utf-8") as f:
        dados = json.load(f)

    total_antes = len(dados)
    limpos = [item for item in dados if titulo_relevante(item["ticker"], item.get("titulo") or "")]
    total_depois = len(limpos)

    with open(saida, "w", encoding="utf-8") as f:
        json.dump(limpos, f, ensure_ascii=False, indent=2)

    return total_antes, total_depois


def main():
    # Arquivo padrão; aceita argumento opcional
    default = Path(__file__).parent / "noticias_b3_sem_duplicados.json"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default

    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        sys.exit(1)

    antes, depois = limpar_noticias(path)
    removidos = antes - depois
    print(f"Antes: {antes} notícias")
    print(f"Depois: {depois} notícias (removidas {removidos} com título genérico)")
    print(f"Salvo em: {path}")


if __name__ == "__main__":
    main()
