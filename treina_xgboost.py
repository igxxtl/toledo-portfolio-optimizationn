from __future__ import annotations

"""
treina_xgboost.py — Classificador XGBoost (supera / não supera Ibovespa)
-------------------------------------------------------------------------
Mesma lógica do treina_gru.py, mas usando XGBoost (árvores de decisão
com gradient boosting) em vez de redes neurais recorrentes.

Diferenças principais em relação ao GRU:
  - Entrada tabular (2D): sem janela de lookback — as 54 features já
    codificam múltiplos horizontes temporais (3d, 5d, 14d, 21d, 42d, 63d).
  - Sem normalização: XGBoost é invariante à escala das features.
  - Early stopping nativo via eval_set (sem loop manual de épocas).
  - Saída extra: importância das features por grupo e por janela.

Target: 1 se Σ alpha(t+1..t+HORIZON) > 0, senão 0
        alpha = R_A - R_M = log-retorno ativo − log-retorno Ibovespa
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)   # silencia logs de cada trial

ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "dados_diarios"
OUT_DIR   = ROOT / "resultados_xgboost"
OUT_DIR.mkdir(exist_ok=True)
IBOV_STEM = "IBOV"

# ── configuração ──────────────────────────────────────────────────────────────

TICKERS: list[str] | str = "all"   # "all" ou ex: ["ABEV3", "PETR4"]

_WINDOWS = [3, 5, 14, 21, 42, 63]
_FEATURE_BASE = [
    "spread_retorno",   # 1. Momentum Relativo
    "dist_sma",
    "spread_vol",       # 2. Sensibilidade e Risco
    "beta",
    "spread_volratio",  # 3. Microestrutura e Liquidez
    "z_alpha",          # 4. Excesso Estatístico
    "trend_mercado",    # 6. Regime de Mercado
    "risco_mercado",
    "z_mercado",
]
FEATURE_COLS: list[str] = [
    f"{feat}_{n}d"
    for n in _WINDOWS
    for feat in _FEATURE_BASE
]

PRICE_COL_A = "close"
PRICE_COL_M = "ibov_close"

MODE    = "daily"           # "daily" | "weekly"
HORIZON = {"daily": 1, "weekly": 5}[MODE]

THRESHOLD   = 0.50          # limiar de decisão
TRAIN_RATIO = 0.80
SEED        = 42

# ── configuração do Optuna ────────────────────────────────────────────────────
OPTUNA_TRIALS  = 50    # trials por ticker (↑ = mais preciso, ↑ = mais lento)
OPTUNA_CV_FOLDS = 5    # folds do TimeSeriesSplit dentro do conjunto de treino
# Espaço de busca dos hiperparâmetros
PARAM_SPACE = {
    "learning_rate":    (0.01,  0.20),   # (min, max)
    "max_depth":        (3,     7),      # árvores rasas para dados financeiros
    "subsample":        (0.60,  1.00),
    "colsample_bytree": (0.50,  1.00),
    "min_child_weight": (5,     50),
    "reg_alpha":        (0.0,   2.0),
    "reg_lambda":       (0.5,   5.0),
    "n_estimators":     500,             # fixo — early stopping controla
}

# Hiperparâmetros padrão usados se OPTUNA_TRIALS = 0 (modo rápido sem tuning)
XGB_DEFAULT = dict(
    n_estimators          = 1000,
    learning_rate         = 0.03,
    max_depth             = 4,
    subsample             = 0.80,
    colsample_bytree      = 0.80,
    min_child_weight      = 10,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    early_stopping_rounds = 40,
    eval_metric           = "auc",
    random_state          = SEED,
    n_jobs                = -1,
    verbosity             = 0,
)


# ── utilidades ────────────────────────────────────────────────────────────────

def make_labels(
    alpha_daily: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Para cada linha i (até n-horizon-1), calcula o label binário:
      1 se Σ alpha_daily[i+1..i+horizon] > 0 (ativo supera Ibovespa)
      0 caso contrário
    Retorna (índices válidos, labels).
    """
    n = len(alpha_daily)
    idxs, ys = [], []
    for i in range(n - horizon - 1):
        acum = float(np.sum(alpha_daily[i + 1 : i + 1 + horizon]))
        idxs.append(i)
        ys.append(1 if acum > 0.0 else 0)
    return np.array(idxs), np.array(ys, dtype=np.int32)


def carregar_df(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    colunas_obrigatorias = FEATURE_COLS + [PRICE_COL_A, PRICE_COL_M]
    faltando = [c for c in colunas_obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes — rode prepara_dados.py antes: {faltando}")

    n_antes = len(df)
    df = df.dropna(subset=colunas_obrigatorias).reset_index(drop=True)
    n_removidas = n_antes - len(df)
    if n_removidas > 0:
        print(f"  warm-up removido: {n_removidas} linhas (lag máx={max(_WINDOWS)}d)", flush=True)

    return df


# ── otimização de hiperparâmetros ────────────────────────────────────────────

def otimizar_params(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scale_pos_weight: float,
    n_trials: int,
) -> dict:
    """
    Usa Optuna com TimeSeriesSplit para encontrar os melhores hiperparâmetros
    sem look-ahead bias. A validação é feita exclusivamente dentro do treino.
    Retorna o dict de parâmetros que maximizou o AUC-ROC médio nos folds.
    """
    tscv = TimeSeriesSplit(n_splits=OPTUNA_CV_FOLDS)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators      = PARAM_SPACE["n_estimators"],
            learning_rate     = trial.suggest_float("learning_rate",    *PARAM_SPACE["learning_rate"],    log=True),
            max_depth         = trial.suggest_int("max_depth",          *PARAM_SPACE["max_depth"]),
            subsample         = trial.suggest_float("subsample",        *PARAM_SPACE["subsample"]),
            colsample_bytree  = trial.suggest_float("colsample_bytree", *PARAM_SPACE["colsample_bytree"]),
            min_child_weight  = trial.suggest_int("min_child_weight",   *PARAM_SPACE["min_child_weight"]),
            reg_alpha         = trial.suggest_float("reg_alpha",        *PARAM_SPACE["reg_alpha"]),
            reg_lambda        = trial.suggest_float("reg_lambda",       *PARAM_SPACE["reg_lambda"]),
            scale_pos_weight  = scale_pos_weight,
            eval_metric       = "auc",
            early_stopping_rounds = 30,
            random_state      = SEED,
            n_jobs            = -1,
            verbosity         = 0,
        )

        aucs = []
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue   # fold sem ambas as classes — pula

            model = XGBClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            prob = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, prob)
            aucs.append(auc(fpr, tpr))

            # pruning: cancela trial se fold inicial for muito ruim
            trial.report(float(np.mean(aucs)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(aucs)) if aucs else 0.5

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update(dict(
        n_estimators          = PARAM_SPACE["n_estimators"],
        scale_pos_weight      = scale_pos_weight,
        early_stopping_rounds = 40,
        eval_metric           = "auc",
        random_state          = SEED,
        n_jobs                = -1,
        verbosity             = 0,
    ))
    return best, study.best_value


# ── pipeline por ticker ───────────────────────────────────────────────────────

def treinar(ticker: str, df: pd.DataFrame) -> dict:
    X_all  = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    r_a    = np.log(df[PRICE_COL_A] / df[PRICE_COL_A].shift(1))
    r_m    = np.log(df[PRICE_COL_M] / df[PRICE_COL_M].shift(1))
    alpha  = np.nan_to_num((r_a - r_m).to_numpy(dtype=np.float32), nan=0.0)
    dates  = df["date"].to_numpy()
    closes = df["close"].to_numpy(dtype=np.float32)

    # labels para cada linha (sem lookback — features já têm memória temporal)
    idxs, y_all = make_labels(alpha, HORIZON)
    X_all = X_all[idxs]
    dates = dates[idxs]
    closes = closes[idxs]

    split = int(len(X_all) * TRAIN_RATIO)
    if split < 50:
        raise ValueError(f"{ticker}: série muito curta após limpeza.")

    X_train, y_train = X_all[:split], y_all[:split]
    X_test,  y_test  = X_all[split:], y_all[split:]
    t_test     = dates[split:]
    close_test = closes[split:]

    # peso de classe proporcional ao desequilíbrio no treino
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1.0)

    # ── otimização de hiperparâmetros (Optuna) ────────────────────────────
    if OPTUNA_TRIALS > 0:
        print(f"    Optuna: {OPTUNA_TRIALS} trials × {OPTUNA_CV_FOLDS} folds ...", flush=True)
        best_params, best_cv_auc = otimizar_params(
            X_train, y_train, scale_pos_weight, OPTUNA_TRIALS
        )
        print(f"    Melhor AUC-CV: {best_cv_auc:.4f} | "
              f"lr={best_params['learning_rate']:.4f} | "
              f"depth={best_params['max_depth']} | "
              f"alpha={best_params['reg_alpha']:.3f}", flush=True)
    else:
        best_params = dict(scale_pos_weight=scale_pos_weight, **XGB_DEFAULT)
        best_cv_auc = float("nan")

    # ── treino final com melhores parâmetros ──────────────────────────────
    model = XGBClassifier(**best_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    n_rounds = model.best_iteration + 1

    # ── previsão ──────────────────────────────────────────────────────────
    prob_sobe = model.predict_proba(X_test)[:, 1]
    pred_cls  = (prob_sobe >= THRESHOLD).astype(int)
    real_cls  = y_test

    # ── métricas ──────────────────────────────────────────────────────────
    acc  = float(accuracy_score(real_cls, pred_cls) * 100)
    prec = float(precision_score(real_cls, pred_cls, zero_division=0) * 100)
    rec  = float(recall_score(real_cls, pred_cls, zero_division=0) * 100)
    f1   = float(f1_score(real_cls, pred_cls, zero_division=0) * 100)
    fpr, tpr, _ = roc_curve(real_cls, prob_sobe)
    roc_auc = float(auc(fpr, tpr) * 100)
    cm = confusion_matrix(real_cls, pred_cls)
    taxa_alpha_pos = float(real_cls.mean() * 100)

    metrics = {
        "ticker":          ticker,
        "n_treino":        split,
        "n_teste":         len(X_test),
        "acuracia_pct":    acc,
        "precisao_pct":    prec,
        "recall_pct":      rec,
        "f1_pct":          f1,
        "auc_roc_pct":     roc_auc,
        "auc_cv_optuna":   round(best_cv_auc, 4),
        "taxa_alpha_pos":  taxa_alpha_pos,
        "horizon_dias":    HORIZON,
        "n_features":      len(FEATURE_COLS),
        "n_rounds":        n_rounds,
        "lr":              round(best_params.get("learning_rate", float("nan")), 4),
        "max_depth":       best_params.get("max_depth", "—"),
    }

    mode_label = f"{'semanal' if MODE == 'weekly' else 'diário'} (t+1..t+{HORIZON})"

    # ── salva previsões ───────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "date":              pd.to_datetime(t_test),
        "ticker":            ticker,
        "close":             close_test,
        "real_supera_ibov":  real_cls,
        "prob_supera_ibov":  prob_sobe,
        "pred_supera_ibov":  pred_cls,
        "acerto":            (pred_cls == real_cls).astype(int),
    }).sort_values("date").reset_index(drop=True)
    pred_df.to_csv(OUT_DIR / f"{ticker}_previsoes.csv", index=False)

    # ── salva modelo ──────────────────────────────────────────────────────
    joblib.dump(model, OUT_DIR / f"{ticker}_model.pkl")

    # ── importância das features ───────────────────────────────────────────
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)

    # agrega por grupo (remove sufixo _Nd, ex: "spread_retorno_21d" → "spread_retorno")
    grupo_idx = ["_".join(f.split("_")[:-1]) for f in FEATURE_COLS]
    imp_grupo = (
        pd.Series(importances.values, index=grupo_idx)
        .groupby(level=0).sum()
        .sort_values(ascending=False)
    )
    # agrega por janela temporal (ex: "spread_retorno_21d" → "21d")
    janela_idx = [f.split("_")[-1] for f in FEATURE_COLS]
    imp_janela = (
        pd.Series(importances.values, index=janela_idx)
        .groupby(level=0).sum()
        .sort_values(ascending=False)
    )

    # salva CSV de importância para análise posterior
    imp_df = pd.DataFrame({
        "feature":  FEATURE_COLS,
        "grupo":    grupo_idx,
        "janela":   janela_idx,
        "importance": importances.values,
    }).sort_values("importance", ascending=False)
    imp_df.to_csv(OUT_DIR / f"{ticker}_importancia.csv", index=False)

    # ── plot 4 painéis ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"XGBoost Classificador — {ticker}  [{mode_label}]\n"
        f"Acurácia: {acc:.1f}%  |  F1: {f1:.1f}%  |  AUC-ROC: {roc_auc:.1f}%  |  "
        f"Precisão: {prec:.1f}%  |  Recall: {rec:.1f}%  |  Rounds: {n_rounds}",
        fontsize=11,
    )

    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)
    ax0 = fig.add_subplot(gs[0, :])   # probabilidade — largura total
    ax1 = fig.add_subplot(gs[1, :])   # preço close — largura total
    ax2 = fig.add_subplot(gs[2, 0])   # curva ROC
    ax3 = fig.add_subplot(gs[2, 1])   # matriz de confusão

    # painel 0: probabilidade prevista
    datas = pred_df["date"].values
    ax0.fill_between(datas, 0.5, pred_df["prob_supera_ibov"],
                     where=pred_df["prob_supera_ibov"] >= 0.5,
                     color="tab:green", alpha=0.35, label="Previsto: supera Ibovespa")
    ax0.fill_between(datas, pred_df["prob_supera_ibov"], 0.5,
                     where=pred_df["prob_supera_ibov"] < 0.5,
                     color="tab:red", alpha=0.35, label="Previsto: abaixo do Ibovespa")
    ax0.plot(datas, pred_df["prob_supera_ibov"], color="tab:blue", lw=0.8, alpha=0.7)
    ax0.axhline(0.5, color="gray", lw=0.8, ls="--")
    erros_mask = pred_df["acerto"] == 0
    ax0.scatter(datas[erros_mask], pred_df["prob_supera_ibov"][erros_mask],
                color="black", s=12, zorder=5, label="Erro de classificação")
    ax0.set_ylabel("P(supera Ibovespa)")
    ax0.set_ylim(0, 1)
    ax0.legend(fontsize=8, loc="upper left")
    ax0.grid(True, alpha=0.20)

    # painel 1: preço com fundo de acerto/erro
    ax1.plot(datas, pred_df["close"], color="tab:blue", lw=1)
    for i, row in pred_df.iterrows():
        cor = "tab:green" if row["acerto"] == 1 else "tab:red"
        if i + 1 < len(pred_df):
            ax1.axvspan(row["date"], pred_df["date"].iloc[i + 1],
                        alpha=0.08, color=cor, lw=0)
    ax1.set_ylabel("Close (R$)")
    ax1.set_xlabel("Data")
    ax1.grid(True, alpha=0.20)

    # painel 2: curva ROC
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.1f}%")
    ax2.plot([0, 1], [0, 1], "k--", lw=0.8, label="Aleatório (50%)")
    ax2.set_xlabel("Taxa de Falso Positivo")
    ax2.set_ylabel("Taxa de Verdadeiro Positivo")
    ax2.set_title("Curva ROC")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # painel 3: matriz de confusão
    im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax3)
    classes = ["Cai (0)", "Sobe (1)"]
    ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
    ax3.set_xticklabels(classes); ax3.set_yticklabels(classes)
    ax3.set_xlabel("Previsto"); ax3.set_ylabel("Real")
    ax3.set_title("Matriz de Confusão")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black",
                     fontsize=13, fontweight="bold")

    plt.savefig(OUT_DIR / f"{ticker}_plot.png", dpi=140, bbox_inches="tight")
    plt.close()

    # ── plot importância (por grupo e por janela) ──────────────────────────
    fig2, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle(f"Importância das Features — {ticker}", fontsize=12)

    imp_grupo.plot(kind="barh", ax=axA, color="steelblue")
    axA.set_title("Por grupo de feature")
    axA.set_xlabel("Importância total (gain)")
    axA.invert_yaxis()
    axA.grid(True, alpha=0.25, axis="x")

    imp_janela.plot(kind="barh", ax=axB, color="darkorange")
    axB.set_title("Por janela temporal")
    axB.set_xlabel("Importância total (gain)")
    axB.invert_yaxis()
    axB.grid(True, alpha=0.25, axis="x")

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{ticker}_importancia.png", dpi=140, bbox_inches="tight")
    plt.close()

    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED)
    print(
        f"XGBoost | mode={MODE} | horizon={HORIZON}d | "
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
                f"recall={m['recall_pct']:.1f}%  "
                f"rounds={m['n_rounds']}"
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
        cols = ["ticker", "acuracia_pct", "f1_pct", "auc_roc_pct", "auc_cv_optuna",
                "precisao_pct", "recall_pct", "taxa_alpha_pos", "n_teste", "n_rounds",
                "lr", "max_depth"]
        print(metrics_df[cols].round(2).to_string(index=False))
        print(f"\nResultados em: {OUT_DIR}")

        # ── plot consolidado de feature importance ────────────────────────
        imp_csvs = sorted(OUT_DIR.glob("*_importancia.csv"))
        if imp_csvs:
            imp_all = pd.concat([pd.read_csv(f) for f in imp_csvs], ignore_index=True)

            imp_grupo_all = (
                imp_all.groupby("grupo")["importance"].mean()
                .sort_values(ascending=False)
            )
            imp_janela_all = (
                imp_all.groupby("janela")["importance"].mean()
                .sort_values(ascending=False)
            )
            imp_top_all = (
                imp_all.groupby("feature")["importance"].mean()
                .sort_values(ascending=False)
                .head(20)
            )

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle(
                f"Feature Importance Consolidada — {len(imp_csvs)} tickers "
                f"| mode={MODE} | horizon={HORIZON}d",
                fontsize=13,
            )

            imp_grupo_all.plot(kind="barh", ax=axes[0], color="steelblue")
            axes[0].set_title("Por grupo de feature\n(média entre tickers)")
            axes[0].set_xlabel("Importância média (gain)")
            axes[0].invert_yaxis()
            axes[0].grid(True, alpha=0.25, axis="x")

            imp_janela_all.plot(kind="barh", ax=axes[1], color="darkorange")
            axes[1].set_title("Por janela temporal\n(média entre tickers)")
            axes[1].set_xlabel("Importância média (gain)")
            axes[1].invert_yaxis()
            axes[1].grid(True, alpha=0.25, axis="x")

            imp_top_all.plot(kind="barh", ax=axes[2], color="seagreen")
            axes[2].set_title("Top 20 features individuais\n(média entre tickers)")
            axes[2].set_xlabel("Importância média (gain)")
            axes[2].invert_yaxis()
            axes[2].grid(True, alpha=0.25, axis="x")

            plt.tight_layout()
            plt.savefig(OUT_DIR / "importancia_consolidada.png", dpi=140, bbox_inches="tight")
            plt.close()
            print(f"Feature importance consolidada salva em: {OUT_DIR / 'importancia_consolidada.png'}")

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
