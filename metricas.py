"""
metricas.py
-----------
Imprime as métricas dos modelos XGBoost gerados por criacao_modelo_xgb.py.

Uso:
    python metricas.py                  # vencedores resumido
    python metricas.py --todos          # todos os horizontes
    python metricas.py --comparacao     # diff vs rodada anterior
"""

import sys
import io
from pathlib import Path

# Forca UTF-8 no stdout do Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "criacao_modelo_xgb"


def sep(char="─", n=90):
    print(char * n)


def print_vencedores(df: pd.DataFrame) -> None:
    sep("═")
    print("  MODELOS VENCEDORES — Walk-Forward OOF")
    sep("═")

    cols = [
        "ticker", "horizon_nome", "n_oof",
        "dir_acc_pct", "rmse_wf", "mae_wf", "r2_wf",
        "q_simples", "omega_ajustado", "penalidade_fator",
    ]
    if "q_source" in df.columns:
        cols.append("q_source")
    cols = [c for c in cols if c in df.columns]

    view = df[cols].sort_values("dir_acc_pct", ascending=False).copy()
    view["q_simples"] = (view["q_simples"] * 100).round(3).astype(str) + "%"

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(view.to_string(index=False))

    sep()
    n = len(df)
    print(f"  Média Dir Acc  : {df['dir_acc_pct'].mean():.1f}%")
    print(f"  Acima de 52%   : {(df['dir_acc_pct'] > 52).sum()} / {n} tickers")
    print(f"  Acima de 55%   : {(df['dir_acc_pct'] > 55).sum()} / {n} tickers")
    print(f"  Acima de 60%   : {(df['dir_acc_pct'] > 60).sum()} / {n} tickers")
    print(f"  Média R²       : {df['r2_wf'].mean():.4f}")
    print(f"  Média RMSE     : {df['rmse_wf'].mean():.5f}")

    if "q_source" in df.columns:
        src = df["q_source"].value_counts().to_dict()
        print(f"  Fonte do Q     : {src}")

    if "penalidade_fator" in df.columns:
        pen = df[df["penalidade_fator"] > 1.0]
        if not pen.empty:
            print(f"\n  [!] Penalidade aplicada ({len(pen)} tickers):")
            for _, r in pen.iterrows():
                print(f"      {r['ticker']:<8} dir_acc={r['dir_acc_pct']:.1f}%  "
                      f"fator={r['penalidade_fator']:.2f}x")
    sep("═")


def print_todos_horizontes(df: pd.DataFrame) -> None:
    sep("═")
    print("  TODOS OS HORIZONTES — Dir Acc (%) por ticker")
    sep("═")
    pivot = df.pivot_table(
        index="ticker", columns="horizon_nome",
        values="dir_acc_pct", aggfunc="first",
    ).round(1)
    # Marca vencedor por ticker com *
    if "vencedor" in df.columns:
        for _, row in df[df["vencedor"]].iterrows():
            t, h = row["ticker"], row["horizon_nome"]
            if t in pivot.index and h in pivot.columns:
                pivot.loc[t, h] = str(pivot.loc[t, h]) + " *"
    print(pivot.to_string())
    sep()
    print("  * = horizonte vencedor por ticker")
    sep("═")


def print_comparacao(ant: pd.DataFrame, atual: pd.DataFrame) -> None:
    comp = ant[["ticker", "dir_acc_pct", "rmse_wf"]].merge(
        atual[["ticker", "dir_acc_pct", "rmse_wf"]],
        on="ticker", suffixes=("_ant", "_atual"),
    )
    comp["d_dir"] = comp["dir_acc_pct_atual"] - comp["dir_acc_pct_ant"]
    comp["d_rmse"] = comp["rmse_wf_atual"] - comp["rmse_wf_ant"]
    comp = comp.sort_values("d_dir", ascending=False)

    sep("═")
    print("  COMPARAÇÃO: rodada anterior → atual")
    sep("═")
    fmt = "  {:<8}  {:>10}  {:>12}  {:>10}  {:>10}"
    print(fmt.format("ticker", "DirAcc ant", "DirAcc atual", "Δ DirAcc", "Δ RMSE"))
    sep()
    ganhos = perdas = 0
    for _, r in comp.iterrows():
        sinal = "▲" if r["d_dir"] > 0 else ("▼" if r["d_dir"] < 0 else "=")
        if r["d_dir"] > 0: ganhos += 1
        elif r["d_dir"] < 0: perdas += 1
        print(fmt.format(
            r["ticker"],
            f"{r['dir_acc_pct_ant']:.1f}%",
            f"{r['dir_acc_pct_atual']:.1f}%",
            f"{sinal} {abs(r['d_dir']):.2f}pp",
            f"{r['d_rmse']:+.5f}",
        ))
    sep()
    print(f"  Média Δ DirAcc : {comp['d_dir'].mean():+.2f}pp  "
          f"|  melhoraram: {ganhos}  |  pioraram: {perdas}  "
          f"|  iguais: {len(comp)-ganhos-perdas}")
    sep("═")


def main() -> None:
    args = sys.argv[1:]

    arq_venc = OUT_DIR / "metricas_vencedores.csv"
    arq_todos = OUT_DIR / "metricas_todos_horizontes.csv"
    arq_ant   = OUT_DIR / "metricas_vencedores_anterior.csv"

    if not arq_venc.exists():
        print(f"Arquivo não encontrado: {arq_venc}")
        print("Rode criacao_modelo_xgb.py primeiro.")
        sys.exit(1)

    df_venc = pd.read_csv(arq_venc)

    if "--comparacao" in args and arq_ant.exists():
        df_ant = pd.read_csv(arq_ant)
        print_comparacao(df_ant, df_venc)

    if "--todos" in args and arq_todos.exists():
        df_todos = pd.read_csv(arq_todos)
        print_todos_horizontes(df_todos)

    print_vencedores(df_venc)

    # Vetor Q resumido
    arq_q = OUT_DIR / "q_vetor.csv"
    if arq_q.exists():
        sep()
        print("  VETOR Q (entrada Black-Litterman)")
        sep()
        qdf = pd.read_csv(arq_q)
        qdf["q_pct"] = (qdf["q_simples"] * 100).round(3).astype(str) + "%"
        qdf["q_log_pct"] = (qdf["q_log"] * 100).round(3).astype(str) + "%"
        print(qdf[["ticker", "horizonte_vencedor", "q_log_pct", "q_pct"]
                   + (["q_source"] if "q_source" in qdf.columns else [])
                   ].to_string(index=False))
        sep()


if __name__ == "__main__":
    main()
