"""
plot_previsoes.py
-----------------
Compara retorno real x previsto para cada ticker.
Usa o horizonte VENCEDOR de cada ticker (lido de metricas_vencedores.csv).

Saídas em criacao_modelo_xgb/:
  painel_previsoes.png    grade de subplots (série temporal + sinal direcional)
  painel_dispersao.png    dispersão real x previsto por ticker
"""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path(__file__).resolve().parent / "criacao_modelo_xgb"

# ── config ────────────────────────────────────────────────────────────────────
SUAVIZAR = 3    # média móvel sobre as predições para visualização (1 = sem suavização)
COLS     = 4    # colunas na grade de subplots


# ── carrega horizonte vencedor por ticker ─────────────────────────────────────
def carregar_dados() -> list[tuple[str, str, pd.DataFrame]]:
    """
    Retorna lista de (ticker, horizonte_nome, df_predicoes).
    Usa metricas_vencedores.csv para descobrir o horizonte correto por ticker.
    Se não existir, cai para _predicoes.csv (arquivo padrão copiado pelo script).
    """
    arq_venc = OUT_DIR / "metricas_vencedores.csv"
    tickers_dados = []

    if arq_venc.exists():
        meta = pd.read_csv(arq_venc)[["ticker", "horizon", "horizon_nome"]]
        for _, row in meta.iterrows():
            ticker  = row["ticker"]
            h       = int(row["horizon"])
            h_nome  = row["horizon_nome"]
            fp = OUT_DIR / f"{ticker}_h{h}d_predicoes.csv"
            if not fp.exists():
                fp = OUT_DIR / f"{ticker}_predicoes.csv"  # fallback ao padrão
            if fp.exists():
                df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
                tickers_dados.append((ticker, h_nome, df))
    else:
        # fallback: usa todos os _predicoes.csv padrão
        for fp in sorted(OUT_DIR.glob("*_predicoes.csv")):
            if "_h" in fp.stem:
                continue  # ignora artefatos de horizontes específicos
            ticker = fp.stem.replace("_predicoes", "")
            df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            tickers_dados.append((ticker, "?", df))

    if not tickers_dados:
        raise FileNotFoundError(
            f"Nenhum arquivo de predições encontrado em {OUT_DIR}.\n"
            "Rode criacao_modelo_xgb.py primeiro."
        )
    return tickers_dados


def calcular_metricas(df: pd.DataFrame) -> dict:
    y_real = df["y_real"].values
    y_pred = df["y_pred"].values
    return {
        "dir_acc": float(np.mean(np.sign(y_pred) == np.sign(y_real)) * 100),
        "rmse":    float(np.sqrt(mean_squared_error(y_real, y_pred)) * 100),
        "mae":     float(mean_absolute_error(y_real, y_pred) * 100),
        "r2":      float(r2_score(y_real, y_pred)),
        "correl":  float(np.corrcoef(y_real, y_pred)[0, 1]),
    }


# ── Painel 1: Série temporal real × previsto ──────────────────────────────────
def plot_painel_series(tickers_dados: list) -> None:
    n    = len(tickers_dados)
    rows = (n + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(22, rows * 5), sharex=False)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, (ticker, h_nome, df) in enumerate(tickers_dados):
        ax = axes_flat[idx]
        m  = calcular_metricas(df)

        dates  = df["date"]
        y_real = df["y_real"] * 100
        y_pred = df["y_pred"] * 100
        y_plot = y_pred.rolling(SUAVIZAR, min_periods=1).mean() if SUAVIZAR > 1 else y_pred

        # colorir fundo por sinal previsto
        pos = y_pred > 0
        for d0, d1, p in zip(dates[:-1], dates[1:], pos[:-1]):
            ax.axvspan(d0, d1, alpha=0.04,
                       color="tab:green" if p else "tab:red", lw=0)

        ax.plot(dates, y_real, color="#333333", lw=0.7, alpha=0.75, label="Real")
        ax.plot(dates, y_plot, color="#E07000", lw=1.1, alpha=0.95, label="Previsto")
        ax.axhline(0, color="gray", lw=0.5, ls="--")

        cor_dir = "#2ca02c" if m["dir_acc"] >= 55 else ("#ff7f0e" if m["dir_acc"] >= 50 else "#d62728")
        ax.set_title(
            f"{ticker}  [{h_nome}]\n"
            f"Dir {m['dir_acc']:.1f}%   RMSE {m['rmse']:.2f}%   "
            f"MAE {m['mae']:.2f}%   R²={m['r2']:.3f}   r={m['correl']:.2f}",
            fontsize=8.5, pad=4, color=cor_dir if m["dir_acc"] != 50 else "gray",
        )
        ax.set_ylabel("Log-ret (%)", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(True, alpha=0.18, lw=0.5)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "XGBoost — Retorno Real vs Previsto (Walk-Forward OOF)\n"
        "Horizonte vencedor por ticker | Verde = prevê alta | Vermelho = prevê queda\n"
        "Cor do título: verde ≥55%  laranja ≥50%  vermelho <50% (acurácia direcional)",
        fontsize=11, y=1.002,
    )
    fig.tight_layout()
    saida = OUT_DIR / "painel_previsoes.png"
    fig.savefig(saida, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Painel de series salvo em: {saida}")


# ── Painel 2: Dispersão real × previsto ──────────────────────────────────────
def plot_painel_dispersao(tickers_dados: list) -> None:
    n    = len(tickers_dados)
    rows = (n + COLS - 1) // COLS
    fig, axes = plt.subplots(rows, COLS, figsize=(20, rows * 4.5), sharex=False)
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, (ticker, h_nome, df) in enumerate(tickers_dados):
        ax = axes_flat[idx]
        m  = calcular_metricas(df)

        y_real = df["y_real"].values * 100
        y_pred = df["y_pred"].values * 100

        # colore por quadrante (acerto direcional = verde, erro = vermelho)
        acerto = np.sign(y_real) == np.sign(y_pred)
        ax.scatter(y_real[acerto],  y_pred[acerto],  s=6, alpha=0.35,
                   color="tab:green",  label="Acerto dir.")
        ax.scatter(y_real[~acerto], y_pred[~acerto], s=6, alpha=0.35,
                   color="tab:red",    label="Erro dir.")

        lim = float(np.nanmax(np.abs(np.concatenate([y_real, y_pred])))) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7, label="Ideal")
        ax.axhline(0, color="gray", lw=0.4, ls=":")
        ax.axvline(0, color="gray", lw=0.4, ls=":")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax.set_title(
            f"{ticker}  [{h_nome}]\n"
            f"Dir {m['dir_acc']:.1f}%   r={m['correl']:.2f}   R²={m['r2']:.3f}",
            fontsize=8.5, pad=4,
        )
        ax.set_xlabel("Real (%)", fontsize=7)
        ax.set_ylabel("Previsto (%)", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)
        ax.grid(True, alpha=0.20)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "XGBoost — Dispersão: Real vs Previsto (Walk-Forward OOF)\n"
        "Verde = acertou direção | Vermelho = errou direção",
        fontsize=11, y=1.002,
    )
    fig.tight_layout()
    saida = OUT_DIR / "painel_dispersao.png"
    fig.savefig(saida, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Painel de dispersao salvo em: {saida}")


# ── Resumo no terminal ────────────────────────────────────────────────────────
def print_resumo(tickers_dados: list) -> None:
    rows = []
    for ticker, h_nome, df in tickers_dados:
        m = calcular_metricas(df)
        rows.append({"ticker": ticker, "horizonte": h_nome, "n_obs": len(df), **m})

    res = pd.DataFrame(rows).sort_values("dir_acc", ascending=False)

    print()
    print("=" * 75)
    print("  Real vs Previsto — Resumo por ticker (Walk-Forward OOF)")
    print("=" * 75)
    print(f"  {'ticker':<8} {'horiz':<7} {'n_obs':>5} {'DirAcc':>8} "
          f"{'RMSE%':>7} {'MAE%':>7} {'R2':>7} {'r':>6}")
    print("  " + "-" * 62)
    for _, r in res.iterrows():
        flag = " *" if r["dir_acc"] >= 55 else ("  " if r["dir_acc"] >= 50 else " !")
        print(f"  {r['ticker']:<8} {r['horizonte']:<7} {r['n_obs']:>5} "
              f"{r['dir_acc']:>7.1f}%{flag} "
              f"{r['rmse']:>6.2f}% {r['mae']:>6.2f}% "
              f"{r['r2']:>7.3f} {r['correl']:>6.2f}")
    print("  " + "-" * 62)
    print(f"  {'MEDIA':<8} {'':>7} {'':>5} {res['dir_acc'].mean():>7.1f}%   "
          f"{res['rmse'].mean():>6.2f}% {res['mae'].mean():>6.2f}% "
          f"{res['r2'].mean():>7.3f} {res['correl'].mean():>6.2f}")
    print("  * = DirAcc >= 55%    ! = DirAcc < 50% (pior que aleatório)")
    print("=" * 75)


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tickers_dados = carregar_dados()
    print_resumo(tickers_dados)
    plot_painel_series(tickers_dados)
    plot_painel_dispersao(tickers_dados)
