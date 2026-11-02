"""Métricas de cobertura para comparações do gate de notícias.

Unidade primária: um registro (ticker, data) da matriz Q.
'Notícias' significa soma de news_count nessas observações, e não matérias
únicas do JSON (uma notícia pode aparecer para vários tickers da empresa).
"""
from __future__ import annotations

import pandas as pd


def calcular_metricas_cobertura(q: pd.DataFrame, n: int) -> dict[str, int | float]:
    """Calcula estatísticas de ativos-dia, dias e ocorrências de notícias.

    Critérios inclusivo (>= n) e estrito (> n) são informados separadamente.
    Dias são as datas presentes em Q, não todos os dias do calendário.
    Para métricas com notícias no denominador, zero notícias produz NaN.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("n deve ser um inteiro >= 0.")
    required = {"view_date", "ticker", "news_count"}
    missing = required.difference(q.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em Q: {sorted(missing)}")
    if q.empty:
        raise ValueError("Não há observações de Q para calcular a cobertura.")

    df = q[list(required)].copy()
    df["view_date"] = pd.to_datetime(df["view_date"], errors="coerce").dt.normalize()
    if df["view_date"].isna().any() or df["ticker"].isna().any():
        raise ValueError("Datas/tickers inválidos em Q.")
    if df.duplicated(["view_date", "ticker"]).any():
        raise ValueError("Q contém mais de uma observação por ticker e dia.")

    counts = pd.to_numeric(df["news_count"], errors="coerce").fillna(0)
    if (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("news_count deve conter inteiros não negativos.")
    df["news_count"] = counts.astype("int64")

    ge = df["news_count"] >= n
    gt = df["news_count"] > n
    has_news = df["news_count"] > 0
    by_day = df.groupby("view_date", sort=True)["news_count"].agg(
        total_noticias_dia="sum", max_noticias_ativo="max"
    )
    total_pairs = len(df)
    total_days = len(by_day)
    total_news = int(df["news_count"].sum())
    inside_news = int(df.loc[ge, "news_count"].sum())
    outside_news = total_news - inside_news

    def pct(part: int | float, total: int | float) -> float:
        return float(100.0 * part / total) if total else float("nan")

    dias_algum_ge = int((by_day["max_noticias_ativo"] >= n).sum())
    dias_algum_gt = int((by_day["max_noticias_ativo"] > n).sum())
    dias_soma_ge = int((by_day["total_noticias_dia"] >= n).sum())
    dias_soma_gt = int((by_day["total_noticias_dia"] > n).sum())
    dias_com_news = int((by_day["total_noticias_dia"] > 0).sum())

    return {
        "n": n,
        # Compatibilidade com o CSV anterior (denominador: pares ativo-dia).
        "observacoes_total": total_pairs,
        "observacoes_elegiveis": int(ge.sum()),
        "pct_elegiveis": pct(int(ge.sum()), total_pairs),
        "pct_abaixo_limiar": pct(int((~ge).sum()), total_pairs),
        "pct_com_noticias": pct(int(has_news.sum()), total_pairs),
        "pct_sem_noticias": pct(int((~has_news).sum()), total_pairs),
        # Notícias: ocorrências contadas em pares ativo-dia elegíveis.
        "noticias_total_ocorrencias": total_news,
        "noticias_dentro_padrao": inside_news,
        "noticias_fora_padrao": outside_news,
        "pct_noticias_dentro_padrao": pct(inside_news, total_news),
        "pct_noticias_fora_padrao": pct(outside_news, total_news),
        # Frequência por ativo-dia.
        "ativos_total": int(df["ticker"].nunique()),
        "ativo_dias_ge_n": int(ge.sum()),
        "ativo_dias_gt_n": int(gt.sum()),
        "pct_ativo_dias_ge_n": pct(int(ge.sum()), total_pairs),
        "pct_ativo_dias_gt_n": pct(int(gt.sum()), total_pairs),
        "pct_ativo_dias_abaixo_n_com_noticias": pct(int((~ge & has_news).sum()), total_pairs),
        "media_noticias_por_ativo_dia": float(df["news_count"].mean()),
        "mediana_noticias_por_ativo_dia": float(df["news_count"].median()),
        "p90_noticias_por_ativo_dia": float(df["news_count"].quantile(0.9)),
        # Dias: 'algum ativo' respeita o gate por ticker.
        # 'total notícias' soma notícias dos tickers (pode duplicar uma manchete).
        "dias_total": total_days,
        "dias_com_noticias": dias_com_news,
        "pct_dias_com_noticias": pct(dias_com_news, total_days),
        "dias_com_algum_ativo_ge_n": dias_algum_ge,
        "pct_dias_com_algum_ativo_ge_n": pct(dias_algum_ge, total_days),
        "dias_com_algum_ativo_gt_n": dias_algum_gt,
        "pct_dias_com_algum_ativo_gt_n": pct(dias_algum_gt, total_days),
        "dias_com_total_noticias_ge_n": dias_soma_ge,
        "pct_dias_com_total_noticias_ge_n": pct(dias_soma_ge, total_days),
        "dias_com_total_noticias_gt_n": dias_soma_gt,
        "pct_dias_com_total_noticias_gt_n": pct(dias_soma_gt, total_days),
        "media_noticias_por_dia": float(by_day["total_noticias_dia"].mean()),
    }
