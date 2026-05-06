from __future__ import annotations

"""
treina_gru.py — Classificador GRU (sobe / cai)
-----------------------------------------------
Target binário: 1 se retorno acumulado dos próximos HORIZON dias > 0, senão 0.

Modos:
  MODE = "daily"   → HORIZON = 1  (pergunta: amanhã sobe?)
  MODE = "weekly"  → HORIZON = 5  (pergunta: semana que vem sobe?)

Pipeline
────────
  1. Carrega CSV diário com features do prepara_dados.py
  2. Remove warm-up (NaN nos indicadores)
  3. Calcula label binário a partir do retorno acumulado
  4. Split temporal 80/20 (sem embaralhamento)
  5. Normaliza features (StandardScaler ajustado só no treino)
  6. Janelas deslizantes LOOKBACK → label t+HORIZON
  7. Treina GRU com BCEWithLogitsLoss + peso de classe para balancear
  8. Métricas: Acurácia, Precisão, Recall, F1, AUC-ROC
  9. Gráficos: probabilidade vs real, curva ROC, matriz de confusão, loss
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "dados_diarios"
OUT_DIR   = ROOT / "resultados_gru"
OUT_DIR.mkdir(exist_ok=True)
IBOV_STEM = "IBOV"   # exclui o arquivo do benchmark da lista de tickers

# ── configuração ──────────────────────────────────────────────────────────────

TICKERS: list[str] | str = "all"   # "all" ou ex: ["ABEV3", "PETR4"]

# Janelas e features espelham a configuração do prepara_dados.py
_WINDOWS = [3, 5, 14, 21, 42, 63]   # mesmos horizontes do prepara_dados.py
_FEATURE_BASE = [
    # 1. Momentum Relativo
    "spread_retorno",
    "dist_sma",
    # 2. Sensibilidade e Risco
    "spread_vol",
    "beta",
    # 3. Microestrutura e Liquidez
    "spread_volratio",
    # 4. Excesso Estatístico
    "z_alpha",
    # 6. Regime de Mercado
    "trend_mercado",
    "risco_mercado",
    "z_mercado",
]
# gera automaticamente: ["spread_retorno_3d", "spread_retorno_5d", ..., "z_mercado_63d"]
FEATURE_COLS: list[str] = [
    f"{feat}_{n}d"
    for n in _WINDOWS
    for feat in _FEATURE_BASE
]

# Preços usados para calcular o alpha diário (R_A - R_M) em treinar()
# Ambas as colunas já existem nos CSVs gerados pelo prepara_dados.py
PRICE_COL_A = "close"       # preço de fechamento do ativo
PRICE_COL_M = "ibov_close"  # preço de fechamento do Ibovespa

MODE    = "weekly"      # "daily" | "weekly"
HORIZON = {"daily": 1, "weekly": 5}[MODE]

THRESHOLD   = 0.5  # limiar de decisão — ajuste para ↑precisão (0.55) ou ↑recall (0.45)
LOOKBACK    = 35   # dias de contexto: features já codificam até 63d, GRU cobre o curto prazo
TRAIN_RATIO = 0.80
BATCH_SIZE  = 16   # gradientes mais estáveis do que 16 para 54 features de entrada
EPOCHS      = 100 # teto generoso — early stopping controla a parada real
LR          = 5e-4  # convergência mais estável; ReduceLROnPlateau reduz progressivamente
PATIENCE    = 12   # sinal financeiro é ruidoso — evita parada prematura
HIDDEN      = 16   # capacidade adequada para 54 features de entrada
N_LAYERS    = 1    # 2 camadas habilitam dropout interno do GRU (entre camadas)
DROPOUT     = 0.30  # regularização moderada para 2 camadas
SEED        = 42


# ── utilidades ────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sequences(
    X: np.ndarray,
    alpha_daily: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Janelas [t-lookback : t] → label binário do alpha acumulado t+1..t+horizon.
    alpha_daily = R_A - R_M (excesso de retorno diário sobre o Ibovespa).
    label = 1 se Σ alpha > 0 (ativo supera o mercado), 0 caso contrário.
    """
    xs, ys = [], []
    n = len(X)
    for i in range(lookback, n - horizon):
        xs.append(X[i - lookback : i])
        acum_alpha = float(np.sum(alpha_daily[i + 1 : i + 1 + horizon]))
        ys.append(1.0 if acum_alpha > 0.0 else 0.0)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


# ── modelo ────────────────────────────────────────────────────────────────────

class GRUClassifier(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),   # logit (sem sigmoid — BCEWithLogitsLoss cuida disso)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


# ── pipeline por ticker ───────────────────────────────────────────────────────

def carregar_df(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # verifica colunas obrigatórias
    colunas_obrigatorias = FEATURE_COLS + [PRICE_COL_A, PRICE_COL_M]
    faltando = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes — rode prepara_dados.py antes: {faltando}")

    # elimina linhas com NaN em features ou preços (warm-up dos lags máx={max(_WINDOWS)}d)
    n_antes = len(df)
    df = df.dropna(subset=colunas_obrigatorias).reset_index(drop=True)
    n_removidas = n_antes - len(df)
    if n_removidas > 0:
        print(f"  warm-up removido: {n_removidas} linhas (lag máx={max(_WINDOWS)}d)", flush=True)

    return df


def treinar(ticker: str, df: pd.DataFrame) -> dict:
    X_raw  = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    # alpha diário: R_A,t - R_M,t = ln(close_t/close_{t-1}) - ln(ibov_t/ibov_{t-1})
    r_a   = np.log(df[PRICE_COL_A] / df[PRICE_COL_A].shift(1))
    r_m   = np.log(df[PRICE_COL_M] / df[PRICE_COL_M].shift(1))
    y_raw = np.nan_to_num((r_a - r_m).to_numpy(dtype=np.float32), nan=0.0)
    dates  = df["date"].to_numpy()
    closes = df["close"].to_numpy(dtype=np.float32)

    split = int(len(df) * TRAIN_RATIO)
    if split <= LOOKBACK + HORIZON + 2:
        raise ValueError(f"{ticker}: série muito curta ({len(df)} linhas após limpeza).")

    # normaliza features sem look-ahead
    scaler_x = StandardScaler()
    X_sc = X_raw.copy()
    X_sc[:split] = scaler_x.fit_transform(X_raw[:split])
    X_sc[split:] = scaler_x.transform(X_raw[split:])

    # sequências com labels binários
    X_seq, y_seq = make_sequences(X_sc, y_raw, LOOKBACK, HORIZON)

    # índice do primeiro dia do horizonte para cada sequência
    target_idx = np.arange(LOOKBACK + 1, len(X_sc) - HORIZON + 1)
    train_mask = target_idx < split
    test_mask  = target_idx >= split

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError(f"{ticker}: split sem amostras suficientes.")

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_test,  y_test  = X_seq[test_mask],  y_seq[test_mask]
    t_test     = dates[target_idx[test_mask]]
    close_test = closes[target_idx[test_mask]]

    # peso de classe para lidar com desequilíbrio (mercado tende a subir mais)
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

    # ── treino ────────────────────────────────────────────────────────────
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = GRUClassifier(len(FEATURE_COLS), HIDDEN, N_LAYERS, DROPOUT).to(device)
    optim   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=4, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    best_loss, best_state, patience_cnt = math.inf, None, 0
    train_losses: list[float] = []

    for epoch in range(EPOCHS):
        model.train()
        ep_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_losses.append(float(loss.item()))
        ep_loss = float(np.mean(ep_losses))
        train_losses.append(ep_loss)
        sched.step(ep_loss)

        if ep_loss + 1e-9 < best_loss:
            best_loss   = ep_loss
            best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"    early stop época {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── previsão ──────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test).to(device)).cpu().numpy()

    prob_sobe = torch.sigmoid(torch.from_numpy(logits)).numpy()   # P(sobe)
    pred_cls  = (prob_sobe >= THRESHOLD).astype(int)
    real_cls  = y_test.astype(int)

    # ── métricas ──────────────────────────────────────────────────────────
    acc  = float(accuracy_score(real_cls, pred_cls) * 100)
    prec = float(precision_score(real_cls, pred_cls, zero_division=0) * 100)
    rec  = float(recall_score(real_cls, pred_cls, zero_division=0) * 100)
    f1   = float(f1_score(real_cls, pred_cls, zero_division=0) * 100)
    fpr, tpr, _ = roc_curve(real_cls, prob_sobe)
    roc_auc = float(auc(fpr, tpr) * 100)
    cm = confusion_matrix(real_cls, pred_cls)

    # proporção real de dias em que o ativo superou o Ibovespa (alpha > 0)
    taxa_alta_real = float(real_cls.mean() * 100)

    metrics = {
        "ticker":          ticker,
        "n_treino":        int(train_mask.sum()),
        "n_teste":         int(test_mask.sum()),
        "acuracia_pct":    acc,
        "precisao_pct":    prec,
        "recall_pct":      rec,
        "f1_pct":          f1,
        "auc_roc_pct":     roc_auc,
        "taxa_alta_real":  taxa_alta_real,
        "horizon_dias":    HORIZON,
        "lookback_dias":   LOOKBACK,
        "n_features":      len(FEATURE_COLS),
        "epochs_run":      len(train_losses),
    }

    mode_label = f"{'semanal' if MODE == 'weekly' else 'diário'} (t+1..t+{HORIZON})"

    # ── salva previsões ───────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "date":       pd.to_datetime(t_test),
        "ticker":     ticker,
        "close":      close_test,
        "real_supera_ibov":  real_cls,   # 1 = ativo superou o Ibovespa
        "prob_supera_ibov":  prob_sobe,  # P(alpha > 0)
        "pred_supera_ibov":  pred_cls,
        "acerto":            (pred_cls == real_cls).astype(int),
    }).sort_values("date").reset_index(drop=True)
    pred_df.to_csv(OUT_DIR / f"{ticker}_previsoes.csv", index=False)

    # ── plot 4 painéis ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 13))
    fig.suptitle(
        f"GRU Classificador — {ticker}  [{mode_label}]\n"
        f"Acurácia: {acc:.1f}%  |  F1: {f1:.1f}%  |  AUC-ROC: {roc_auc:.1f}%  |  "
        f"Precisão: {prec:.1f}%  |  Recall: {rec:.1f}%",
        fontsize=11,
    )

    gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.30)
    ax0 = fig.add_subplot(gs[0, :])   # probabilidade — largura total
    ax1 = fig.add_subplot(gs[1, :])   # preço close — largura total
    ax2 = fig.add_subplot(gs[2, 0])   # curva ROC
    ax3 = fig.add_subplot(gs[2, 1])   # matriz de confusão

    # ── painel 0: probabilidade prevista vs direção real ──────────────────
    datas = pred_df["date"].values
    ax0.fill_between(datas, 0.5, pred_df["prob_supera_ibov"],
                     where=pred_df["prob_supera_ibov"] >= 0.5,
                     color="tab:green", alpha=0.35, label="Previsto: supera Ibovespa")
    ax0.fill_between(datas, pred_df["prob_supera_ibov"], 0.5,
                     where=pred_df["prob_supera_ibov"] < 0.5,
                     color="tab:red", alpha=0.35, label="Previsto: abaixo do Ibovespa")
    ax0.plot(datas, pred_df["prob_supera_ibov"], color="tab:blue", lw=0.8, alpha=0.7)
    ax0.axhline(0.5, color="gray", lw=0.8, ls="--")

    # marcadores de erro
    erros_mask = pred_df["acerto"] == 0
    ax0.scatter(datas[erros_mask], pred_df["prob_supera_ibov"][erros_mask],
                color="black", s=12, zorder=5, label="Erro de classificação")
    ax0.set_ylabel("P(supera Ibovespa)")
    ax0.set_ylim(0, 1)
    ax0.legend(fontsize=8, loc="upper left")
    ax0.grid(True, alpha=0.20)

    # ── painel 1: preço de fechamento ─────────────────────────────────────
    ax1.plot(datas, pred_df["close"], color="tab:blue", lw=1)

    # fundo verde onde previu sobe e estava certo, vermelho onde errou
    for i, row in pred_df.iterrows():
        cor = "tab:green" if row["acerto"] == 1 else "tab:red"
        if i + 1 < len(pred_df):
            ax1.axvspan(row["date"], pred_df["date"].iloc[i + 1],
                        alpha=0.08, color=cor, lw=0)

    ax1.set_ylabel("Close (R$)")
    ax1.set_xlabel("Data")
    ax1.grid(True, alpha=0.20)

    # ── painel 2: curva ROC ───────────────────────────────────────────────
    ax2.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"AUC = {roc_auc:.1f}%")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="Aleatório (50%)")
    ax2.set_xlabel("Taxa de Falso Positivo")
    ax2.set_ylabel("Taxa de Verdadeiro Positivo")
    ax2.set_title("Curva ROC")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── painel 3: matriz de confusão ──────────────────────────────────────
    im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax3)
    classes = ["Cai (0)", "Sobe (1)"]
    tick_marks = np.arange(2)
    ax3.set_xticks(tick_marks)
    ax3.set_yticks(tick_marks)
    ax3.set_xticklabels(classes)
    ax3.set_yticklabels(classes)
    ax3.set_xlabel("Previsto")
    ax3.set_ylabel("Real")
    ax3.set_title("Matriz de Confusão")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=13, fontweight="bold")

    plt.savefig(OUT_DIR / f"{ticker}_plot.png", dpi=140, bbox_inches="tight")
    plt.close()

    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Device: {device} | mode={MODE} | horizon={HORIZON}d | lookback={LOOKBACK}d | "
        f"threshold={THRESHOLD} | features={len(FEATURE_COLS)}\n"
    )

    csvs = sorted(p for p in DATA_DIR.glob("*.csv") if p.stem != IBOV_STEM)
    if not csvs:
        print(f"Nenhum CSV em {DATA_DIR}. Rode novo_input_dados.py e prepara_dados.py primeiro.")
        sys.exit(1)

    if TICKERS != "all":
        tickers_set = set(TICKERS)
        csvs = [p for p in csvs if any(t in p.stem for t in tickers_set)]

    all_metrics: list[dict] = []
    erros: list[tuple[str, str]] = []

    for fp in csvs:
        ticker = fp.stem
        print(f"[{ticker}] carregando e treinando ...", flush=True)
        try:
            df = carregar_df(fp)
            r_a_h = np.log(df[PRICE_COL_A] / df[PRICE_COL_A].shift(1))
            r_m_h = np.log(df[PRICE_COL_M] / df[PRICE_COL_M].shift(1))
            pct_alpha_pos = float((r_a_h - r_m_h).dropna().gt(0).mean() * 100)
            print(f"  {len(df)} dias úteis | alpha>0 em {pct_alpha_pos:.1f}% dos dias", flush=True)
            m = treinar(ticker, df)
            all_metrics.append(m)
            print(
                f"  acurácia={m['acuracia_pct']:.1f}%  "
                f"F1={m['f1_pct']:.1f}%  "
                f"AUC-ROC={m['auc_roc_pct']:.1f}%  "
                f"precisão={m['precisao_pct']:.1f}%  "
                f"recall={m['recall_pct']:.1f}%"
            )
        except Exception as exc:
            erros.append((ticker, str(exc)))
            print(f"  ERRO: {exc}")

    if all_metrics:
        metrics_df = (
            pd.DataFrame(all_metrics)
            .sort_values("auc_roc_pct", ascending=False)
            .reset_index(drop=True)
        )
        metrics_df.to_csv(OUT_DIR / "metricas_todos.csv", index=False)
        print("\n── Métricas consolidadas (ordenado por AUC-ROC) ────────────────")
        cols = ["ticker", "acuracia_pct", "f1_pct", "auc_roc_pct",
                "precisao_pct", "recall_pct", "taxa_alta_real", "n_teste", "epochs_run"]
        print(metrics_df[cols].round(2).to_string(index=False))
        print(f"\nResultados em: {OUT_DIR}")

    if erros:
        print("\nErros:")
        for t, msg in erros:
            print(f"  {t}: {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro fatal: {exc}")
        sys.exit(1)