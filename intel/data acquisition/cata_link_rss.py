import json
import time
from datetime import date, datetime, timedelta
from urllib.parse import quote_plus

import feedparser
import pandas as pd

# User-Agent para reduzir chance de o servidor fechar a conexão
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
MAX_TENTATIVAS = 4
PAUSA_ENTRE_REQUISICOES = 2  # segundos

# =========================
# CONFIGURAÇÃO
# =========================

# Tickers da B3
tickers = ["VALE3", "BBAS3", "ITUB3", "BBDC3", "ABEV3"]

# Intervalo por datas: do começo ao fim de 2025, em janelas de 1 semana
DATA_INICIO_2025 = date(2025, 1, 1)
DATA_FIM_2025 = date(2025, 12, 31)

idioma = "pt-BR"
pais = "BR"


def gerar_semanas_2025():
    """Gera pares (data_inicio, data_fim_exclusive) para cada semana de 2025."""
    semanas = []
    cur = DATA_INICIO_2025
    while cur <= DATA_FIM_2025:
        fim_exc = cur + timedelta(days=7)
        if fim_exc > DATA_FIM_2025:
            fim_exc = DATA_FIM_2025 + timedelta(days=1)  # before: primeiro dia de 2026
        semanas.append((cur, fim_exc))
        cur = cur + timedelta(days=7)
    return semanas


# =========================
# FUNÇÃO PARA MONTAR URL
# =========================


def montar_url(termo: str, data_inicio: date, data_fim_exclusive: date) -> str:
    """URL no padrão: q=termo+after:YYYY-MM-DD+before:YYYY-MM-DD"""
    query = f"{termo} after:{data_inicio} before:{data_fim_exclusive}"
    q_encoded = quote_plus(query)
    return (
        f"https://news.google.com/rss/search?"
        f"q={q_encoded}&hl={idioma}&gl={pais}&ceid={pais}%3Apt-419"
    )


def buscar_feed_com_retry(url):
    """Baixa o feed com retentativas e backoff; evita abortar por conexão fechada."""
    for tentativa in range(1, MAX_TENTATIVAS + 1):
        try:
            return feedparser.parse(
                url,
                request_headers={"User-Agent": USER_AGENT},
            )
        except Exception as e:
            if tentativa == MAX_TENTATIVAS:
                raise
            espera = 5 * (2 ** (tentativa - 1))  # 5, 10, 20 s
            print(f"  ⚠ Erro ({e.__class__.__name__}), nova tentativa em {espera}s ({tentativa}/{MAX_TENTATIVAS})...")
            time.sleep(espera)


# =========================
# COLETA
# =========================

semanas = gerar_semanas_2025()
print(f"Janelas: {len(semanas)} semanas ({DATA_INICIO_2025} a {DATA_FIM_2025})\n")

df_total = pd.DataFrame()

for ticker in tickers:
    for data_ini, data_fim in semanas:

        url = montar_url(ticker, data_ini, data_fim)
        feed = buscar_feed_com_retry(url)

        registros = []

        for entry in feed.entries:
            registros.append({
                "ticker": ticker,
                "janela_inicio": data_ini.isoformat(),
                "janela_fim": (data_fim - timedelta(days=1)).isoformat(),  # último dia da semana
                "titulo": entry.title,
                "fonte": entry.source.title if "source" in entry else None,
                "data": entry.published,
                "link_google": entry.link,
            })

        df_parcial = pd.DataFrame(registros)

        # concatena
        df_total = pd.concat([df_total, df_parcial], ignore_index=True)

        # remove duplicatas progressivamente
        df_total.drop_duplicates(subset="link_google", inplace=True)

        print(f"✔ {ticker} {data_ini} a {data_fim} → total acumulado: {len(df_total)}")

        time.sleep(PAUSA_ENTRE_REQUISICOES)  # evita bloqueios / conexão fechada

# =========================
# TRATAMENTO FINAL
# =========================
# Mantém "data" como string do RSS; usa coluna auxiliar só para ordenar
df_total["_data_ordem"] = pd.to_datetime(df_total["data"], errors="coerce")
df_total = df_total.sort_values("_data_ordem", ascending=False)
df_total = df_total.drop(columns=["_data_ordem"])

# salva dataset em JSON (data = string exatamente como veio do RSS)
arquivo_json = "noticias_b3.json"
lista_registros = df_total.to_dict(orient="records")
# Garante que "data" seja string (valor bruto do RSS)
for row in lista_registros:
    d = row.get("data")
    row["data"] = str(d) if d is not None and pd.notna(d) else None


def _data_para_ordem(s: str | None):
    """Converte data do RSS em datetime para ordenação; inválidas vão para o fim."""
    if not s or not str(s).strip():
        return datetime.min
    try:
        # Formato típico RSS: "Mon, 26 Jan 2026 08:00:00 GMT"
        return datetime.strptime(str(s)[:25], "%a, %d %b %Y %H:%M:%S")
    except ValueError:
        return datetime.min


# Ordenação decrescente por data de publicação (mais recente primeiro)
lista_registros.sort(key=lambda r: _data_para_ordem(r.get("data")), reverse=True)

with open(arquivo_json, "w", encoding="utf-8") as f:
    json.dump(lista_registros, f, ensure_ascii=False, indent=2)

print(f"\nTotal final de notícias: {len(df_total)}")
print(f"Salvo em: {arquivo_json}")
df_total.head()