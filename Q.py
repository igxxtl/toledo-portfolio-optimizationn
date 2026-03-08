"""
Lê noticias_b3_sem_duplicados_sentimento.json e agrupa por semana (crescente)
e por ticker, exibindo totais de notícias positivas, negativas e neutras.
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ARQUIVO = "noticias_b3_sem_duplicados_sentimento.json"


def parse_data(data_str: str) -> datetime | None:
    """Converte string de data (ex: 'Wed, 31 Dec 2025 08:00:00 GMT') em datetime."""
    if not data_str or not data_str.strip():
        return None
    try:
        return datetime.strptime(
            data_str.replace(" GMT", "").strip(),
            "%a, %d %b %Y %H:%M:%S",
        )
    except ValueError:
        return None


def main():
    caminho = Path(ARQUIVO)
    if not caminho.exists():
        print(f"Arquivo não encontrado: {ARQUIVO}")
        return

    dados = json.loads(caminho.read_text(encoding="utf-8"))

    # Agrupa por (ano, semana) -> ticker -> { "positivo": n, "negativo": n, "neutro": n }
    por_semana_ticker = defaultdict(
        lambda: defaultdict(lambda: {"positivo": 0, "negativo": 0, "neutro": 0})
    )

    for item in dados:
        data_str = item.get("data") or ""
        dt = parse_data(data_str)
        if dt is None:
            janela = item.get("janela_inicio") or ""
            try:
                dt = datetime.strptime(janela, "%Y-%m-%d")
            except ValueError:
                continue
        ano, semana, _ = dt.isocalendar()
        ticker = (item.get("ticker") or "").strip() or "N/A"
        sent = (item.get("sentimento") or "neutro").strip().lower()
        cont = por_semana_ticker[(ano, semana)][ticker]
        if sent not in cont:
            cont[sent] = 0
        cont[sent] += 1

    semanas_ordenadas = sorted(por_semana_ticker.keys())

    for (ano, num_semana) in semanas_ordenadas:
        rotulo = f"Semana {num_semana} {ano}"
        print(f"\n{rotulo}")
        print("  Ticker   | positivas | negativas | neutras | total")
        print("  " + "-" * 55)
        tickers = sorted(por_semana_ticker[(ano, num_semana)].keys())
        for ticker in tickers:
            cont = por_semana_ticker[(ano, num_semana)][ticker]
            pos = cont.get("positivo", 0)
            neg = cont.get("negativo", 0)
            neu = cont.get("neutro", 0)
            total = pos + neg + neu
            print(f"  {ticker:9} | {pos:9} | {neg:9} | {neu:6} | {total:5}")

    # Salva resumo por semana e por ticker em JSON
    resumo = []
    for (ano, num_semana) in semanas_ordenadas:
        tickers_list = []
        for ticker in sorted(por_semana_ticker[(ano, num_semana)].keys()):
            cont = por_semana_ticker[(ano, num_semana)][ticker]
            tickers_list.append({
                "ticker": ticker,
                "positivas": cont.get("positivo", 0),
                "negativas": cont.get("negativo", 0),
                "neutras": cont.get("neutro", 0),
                "total": sum(cont.values()),
            })
        resumo.append({
            "semana": num_semana,
            "ano": ano,
            "rotulo": f"Semana {num_semana} {ano}",
            "tickers": tickers_list,
        })
    out_path = Path("resumo_sentimento_por_semana_ticker.json")
    out_path.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResumo (por semana e ticker) salvo em: {out_path}")


if __name__ == "__main__":
    main()
