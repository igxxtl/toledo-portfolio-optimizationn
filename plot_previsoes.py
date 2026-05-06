"""
plot_previsoes.py
-----------------
Plota grade de graficos: retorno real (preto) x retorno previsto (laranja)
para cada ticker, usando os arquivos *_h21d_predicoes.csv gerados pelo
criacao_modelo_xgb.py.

Saida: criacao_modelo_xgb/painel_previsoes.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
OUT_DIR  = Path(__file__).resolve().parent / "criacao_modelo_xgb"
HORIZONTE = "h21d"           # sufixo dos arquivos a usar
SUAVIZAR  = 5                # media movel das predicoes (dias); 1 = sem suavizacao
COLS      = 4                # colunas na grade
FIGSIZE   = (22, 30)         # largura x altura em polegadas

# ── leitura ───────────────────────────────────────────────────────────────────
arquivos = sorted(OUT_DIR.glob(f"*_{HORIZONTE}_predicoes.csv"))
if not arquivos:
    raise FileNotFoundError(
        f"Nenhum arquivo *_{HORIZONTE}_predicoes.csv em {OUT_DIR}.\n"
        "Rode criacao_modelo_xgb.py primeiro."
    )

tickers_dados: list[tuple[str, pd.DataFrame]] = []
for fp in arquivos:
    ticker = fp.stem.replace(f"_{HORIZONTE}_predicoes", "")
    df = pd.read_csv(fp, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    tickers_dados.append((ticker, df))

n = len(tickers_dados)
rows = (n + COLS - 1) // COLS

# ── figura ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(rows, COLS, figsize=FIGSIZE, sharex=False)
axes_flat = axes.flatten()

for idx, (ticker, df) in enumerate(tickers_dados):
    ax = axes_flat[idx]

    dates  = df["date"]
    y_real = df["y_real"] * 100          # converte para %
    y_pred = df["y_pred"] * 100

    # suavizacao opcional das predicoes (media movel) para reduzir ruido visual
    if SUAVIZAR > 1:
        y_pred_plot = y_pred.rolling(SUAVIZAR, min_periods=1).mean()
    else:
        y_pred_plot = y_pred

    # metricas basicas
    dir_acc = np.mean(np.sign(df["y_pred"]) == np.sign(df["y_real"])) * 100
    rmse    = np.sqrt(np.mean(df["erro"] ** 2)) * 100  # erro em %
    correl  = np.corrcoef(df["y_real"], df["y_pred"])[0, 1]

    # plot
    ax.plot(dates, y_real,      color="#333333", lw=0.8, alpha=0.7, label="Real")
    ax.plot(dates, y_pred_plot, color="#E07000", lw=1.2, alpha=0.9, label="Previsto")
    ax.axhline(0, color="gray", lw=0.5, ls="--")

    # faixa de zero (+-1%) para referencia
    ax.axhspan(-1, 1, color="gray", alpha=0.05)

    ax.set_title(
        f"{ticker}   |   Dir {dir_acc:.1f}%   RMSE {rmse:.2f}%   r={correl:.2f}",
        fontsize=9, pad=4,
    )
    ax.set_ylabel("Log-ret (%)", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.18, lw=0.5)

    # legenda apenas no primeiro painel
    if idx == 0:
        ax.legend(fontsize=7, loc="upper left")

# apaga subplots vazios
for j in range(n, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle(
    "XGBoost — Retorno Real vs Previsto (horizonte 21 dias uteis)\n"
    "Linha preta = real | Linha laranja = previsto | Dir = acuracia direcional",
    fontsize=12, y=1.002,
)
fig.tight_layout()

saida = OUT_DIR / "painel_previsoes.png"
fig.savefig(saida, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"Grafico salvo em: {saida}")
