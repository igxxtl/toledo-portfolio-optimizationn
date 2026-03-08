"""
Lê o JSON de notícias, remove duplicatas pelo título (mesmo título = uma única entrada)
e grava um novo JSON com a lista limpa.
"""

import json
from pathlib import Path

ARQUIVO_ENTRADA = Path(__file__).parent / "noticias_b3.json"
ARQUIVO_SAIDA = Path(__file__).parent / "noticias_b3_sem_duplicados.json"


def normalizar_titulo(titulo: str) -> str:
    """Remove espaços extras para comparar títulos."""
    if not titulo:
        return ""
    return " ".join(titulo.strip().split())


def main():
    if not ARQUIVO_ENTRADA.exists():
        print(f"Arquivo não encontrado: {ARQUIVO_ENTRADA}")
        return

    with open(ARQUIVO_ENTRADA, "r", encoding="utf-8") as f:
        noticias = json.load(f)

    vistos = set()
    sem_duplicados = []

    for item in noticias:
        titulo = item.get("titulo") or ""
        chave = normalizar_titulo(titulo)
        if not chave:
            sem_duplicados.append(item)
            continue
        if chave in vistos:
            continue
        vistos.add(chave)
        sem_duplicados.append(item)

    removidos = len(noticias) - len(sem_duplicados)
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)

    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        json.dump(sem_duplicados, f, ensure_ascii=False, indent=2)

    print(f"Entradas originais: {len(noticias)}")
    print(f"Duplicatas removidas (mesmo título): {removidos}")
    print(f"Entradas no novo JSON: {len(sem_duplicados)}")
    print(f"Salvo em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    main()
