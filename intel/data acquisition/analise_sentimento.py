import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ARQUIVO_ENTRADA = "noticias_b3_sem_duplicados.json"
ARQUIVO_SAIDA = "noticias_b3_sem_duplicados_sentimento.json"
MODELO = "gpt-5-mini"
# Número de chamadas simultâneas à API (aumente com cuidado por causa de rate limits)
MAX_WORKERS = 8

PROMPT = """
Classifique o sentimento da manchete abaixo em relação ao impacto no preço da ação {TICKER}.

Responda apenas com uma palavra:

positivo
negativo
neutro

Manchete:
{HEADLINE}
"""

def classificar_manchete(client: OpenAI, ticker: str, titulo: str) -> str:
    """Chama a API para classificar o sentimento da manchete em relação ao ticker."""
    if not titulo or not titulo.strip():
        return "neutro"

    prompt_preenchido = PROMPT.format(TICKER=ticker, HEADLINE=titulo)

    response = client.responses.create(
        model=MODELO,
        input=prompt_preenchido,
        temperature=1
    )

    texto = (getattr(response, "output_text", None) or "").strip()

    return texto


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    caminho = Path(ARQUIVO_ENTRADA)
    dados = json.loads(caminho.read_text(encoding="utf-8"))

    # Monta o conjunto único (chave, ticker, titulo) para não chamar a API em duplicata
    unique_to_classify = {}
    for item in dados:
        ticker = (item.get("ticker") or "").strip()
        titulo = (item.get("titulo") or "").strip()
        chave = f"{ticker}||{titulo}"
        if chave not in unique_to_classify:
            unique_to_classify[chave] = (ticker, titulo)

    cache = {}
    total_unique = len(unique_to_classify)
    total_itens = len(dados)

    client = OpenAI(api_key=api_key)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(classificar_manchete, client, ticker, titulo): chave
            for chave, (ticker, titulo) in unique_to_classify.items()
        }
        concluidos = 0
        for future in as_completed(futures):
            chave = futures[future]
            try:
                cache[chave] = future.result()
            except Exception as e:
                print(f"Erro ao classificar {chave[:50]}...: {e}")
                cache[chave] = "neutro"
            concluidos += 1
            if concluidos % 50 == 0 or concluidos == total_unique:
                print(f"[{concluidos}/{total_unique}] manchetes únicas classificadas")

    # Atribui sentimento a cada item a partir do cache
    for item in dados:
        ticker = (item.get("ticker") or "").strip()
        titulo = (item.get("titulo") or "").strip()
        chave = f"{ticker}||{titulo}"
        item["sentimento"] = cache.get(chave, "neutro")

    Path(ARQUIVO_SAIDA).write_text(
        json.dumps(dados, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Concluído. {total_itens} itens processados. Salvo em: {ARQUIVO_SAIDA}")

if __name__ == "__main__":
    main()
