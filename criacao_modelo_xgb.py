from __future__ import annotations

"""
criacao_modelo_xgb.py — Regressor XGBoost para o vetor Q (Black-Litterman)
--------------------------------------------------------------------------
Objetivo: prever o log-retorno de cada ativo no período t+1 e usar o
output como vetor Q (opiniões) do modelo de Black-Litterman. A incerteza
"fria" da previsão (omega_base) é estimada a partir do MSE obtido na
Walk-Forward Validation.

Metodologia (resumo):
    1. Target: r_{t+1} = ln(P_{t+1} / P_t)            (log-retorno)
    2. Features:
        2.1 Autoregressivas:    lags do retorno (1, 2, 3, 5)
                                momentum acumulado (15, 30, 90 dias)
        2.2 Risco/Volatilidade: vol histórica (10, 20)  + ATR(14)
        2.3 Indicadores Téc.:   RSI(14), dist EMA(20, 50), Bollinger %B(20)
        2.4 Macro:              retorno Ibov, retorno USDBRL, var DI
                                (USDBRL e DI carregados se houver CSV
                                 correspondente em dados_diarios/)
        2.5 Sazonal:            day-of-week (one-hot), month (one-hot)
    3. Validação: Walk-Forward (expansiva), sem look-ahead.
    4. Output:
        - Q_i        = predição do log-retorno t+1 do ativo i
        - omega_base = MSE da Walk-Forward (incerteza estatística "fria")

Saídas em criacao_modelo_xgb/:
    <TICKER>_predicoes.csv        (real vs previsto por fold)
    <TICKER>_model.pkl            (modelo final treinado em todo o histórico)
    <TICKER>_importancia.csv      (importância por feature)
    <TICKER>_plot.png             (4 painéis: previsão, dispersão, ROC dir, MC)
    q_vetor.csv                   (vetor Q final: 1 linha por ticker)
    omega_base.csv                (MSE/RMSE da Walk-Forward por ticker)
    metricas_todos.csv            (consolidado de métricas)
    importancia_consolidada.png   (média de importância entre tickers)
"""

import io
import shutil
import sys
from pathlib import Path

# Força UTF-8 no stdout do Windows (evita erro com caracteres como → e ⚠)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dados_diarios"
OUT_DIR  = ROOT / "criacao_modelo_xgb"
OUT_DIR.mkdir(exist_ok=True)

IBOV_STEM    = "IBOV"
USDBRL_STEM  = "USDBRL"   # opcional — dados_diarios/USDBRL.csv
DI_STEM      = "DI"       # opcional — dados_diarios/DI.csv

# ── seleção de tickers ────────────────────────────────────────────────────────

TICKERS: list[str] | str = "all"   # "all" ou ex.: ["ABEV3", "PETR4"]

# ── parâmetros de feature engineering ────────────────────────────────────────
# Janelas calibradas para o horizonte de previsão de 1 mês (h=21 pregões).
# Princípio: features devem capturar a dinâmica relevante para o target horizon.

# § 2.1 — autoregressivas
# lag_ret: sinal de entrada da posição (reversão/continuação de curtíssimo prazo)
# momentum: fatores clássicos de momentum mensal, trimestral, semestral e anual
LAGS_RETORNO     = [1, 5, 21]              # ontem, semana passada, mês passado
MOMENTUM_JANELAS = [21, 63, 126, 252]      # 1m, 3m, 6m, 12m

# § 2.2 — risco/volatilidade
# Regime de vol mensal e trimestral — horizontes relevantes para clustering de volatilidade
VOL_JANELAS      = [21, 63]                # vol realizada mensal e trimestral
ATR_JANELA       = 21                      # ATR médio do último mês

# § 2.3 — indicadores técnicos
# Todos recalibrados para o ciclo mensal
RSI_JANELA       = 21                      # RSI mensal
EMA_JANELAS      = [21, 63]                # EMA 1 mês e 3 meses
BBANDS_JANELA    = 21                      # Bollinger mensal
BBANDS_K         = 2.0                     # k = 2 desvios

# § 2.2b — volume
# Janelas para atividade de volume relativo e VWAP rolling
VOL_VOLUME_JANELAS = [21, 63]   # volume relativo mensal e trimestral
VOL_VWAP_JANELAS   = [21]       # VWAP rolling mensal

# § 2.4 complementar — contexto de mercado (recalculado a partir de ibov_close)
# Beta e z-alpha em janelas mensais e trimestrais para capturar regime sistêmico
BETA_JANELAS          = [21, 63]   # beta móvel mensal e trimestral
Z_ALPHA_JANELAS       = [21, 63]   # z-score do alpha mensal e trimestral
RISCO_MERCADO_JANELAS = [21, 63]   # vol. Ibovespa mensal e trimestral
TREND_MERCADO_JANELAS = [21, 63]   # tendência acumulada do Ibovespa mensal e trimestral

# Tentar baixar USDBRL via yfinance se o CSV não existir.
AUTO_DOWNLOAD_USDBRL = True

# ── teste de horizonte de previsão ────────────────────────────────────────────
# Para cada ticker, o modelo é treinado 3 vezes (Direct Forecasting independente).
# Target: ln(P_{t+h} / P_t)  —  retorno acumulado nos próximos h dias úteis.
HORIZONS         = [21]                     # fixo: 1 mês (21 pregões úteis)
HORIZON_NOMES    = {1: "1d", 5: "1sem", 10: "2sem", 21: "1mes"}
HORIZON_CRITERIO = "dir_acc_pct"            # critério para eleger o vencedor por ticker
                                            # opções: "dir_acc_pct", "rmse_wf", "mae_wf"
                                            # (para rmse/mae, o vencedor é o de menor valor)
HORIZON_MENOR_MELHOR = {"rmse_wf", "mae_wf"}  # conjunto de métricas onde menor = melhor

# ── penalidade de omega por baixa acurácia direcional ────────────────────────
# Modelos com dir_acc_pct abaixo do limiar têm omega inflado exponencialmente,
# reduzindo o peso da opinião Q no Black-Litterman.
#
#   omega_ajustado = omega_base × exp(K × max(0, limiar − dir_acc) / 100)
#
# Exemplos com K=10 e limiar=52%:
#   dir_acc = 51% (1% abaixo) → ×e^0.10 ≈ ×1.11
#   dir_acc = 49% (3% abaixo) → ×e^0.30 ≈ ×1.35
#   dir_acc = 45% (7% abaixo) → ×e^0.70 ≈ ×2.01
DIR_ACC_PENALIDADE_LIMIAR = 52.0   # % — abaixo disso, omega recebe penalidade
DIR_ACC_PENALIDADE_K      = 10.0   # intensidade da penalidade (ver exemplos acima)

# ── walk-forward ──────────────────────────────────────────────────────────────
WF_TRAIN_INICIAL = 0.60   # 60% do histórico usados como treino inicial
WF_N_FOLDS       = 5      # nº de blocos cronológicos de teste
# Embargo: exclui as últimas (horizon × WF_EMBARGO_FATOR) linhas do treino de
# cada fold, evitando que labels sobrepostos (horizonte h=21 cria overlap de 20
# dias consecutivos) contaminem o período de teste.  1.0 = 1 horizonte completo.
WF_EMBARGO_FATOR = 1.0
SEED             = 42

# ── normalização do target por fold ──────────────────────────────────────────
# Se True, o target (log-retorno) é z-scored dentro de cada fold de treino.
# A predição é desnormalizada antes de calcular métricas e gerar o vetor Q.
# Efeito: força o modelo a aprender *desvios relativos* em vez de valores
# absolutos, contornando o encolhimento para zero causado pela regularização
# quando a série tem média ≈ 0 e alta volatilidade.
NORMALIZAR_TARGET = True

# ── Optuna (opcional, dentro do treino inicial) ───────────────────────────────
OPTUNA_TRIALS    = 30     # 0 = desliga (usa XGB_DEFAULT). Sugestão: 30–60.
                          # Nota: testes com 50 trials mostraram overfitting em dados
                          # financeiros não-estacionários — default conservador é superior.
OPTUNA_CV_FOLDS  = 4
PARAM_SPACE = {
    "learning_rate":    (0.01, 0.20),
    "max_depth":        (3,    7),
    "subsample":        (0.60, 1.00),
    "colsample_bytree": (0.50, 1.00),
    "min_child_weight": (5,    50),
    "reg_alpha":        (0.0,  2.0),
    "reg_lambda":       (0.5,  5.0),
    "n_estimators":     800,
}

XGB_DEFAULT = dict(
    n_estimators          = 1000,
    learning_rate         = 0.03,
    max_depth             = 4,
    subsample             = 0.80,
    colsample_bytree      = 0.80,
    min_child_weight      = 10,
    reg_alpha             = 0.1,
    reg_lambda            = 1.0,
    early_stopping_rounds = 50,
    eval_metric           = "rmse",
    random_state          = SEED,
    n_jobs                = -1,
    verbosity             = 0,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de indicadores técnicos
# ─────────────────────────────────────────────────────────────────────────────

def log_ret(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))


def rolling_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """Average True Range normalizado pelo close (ATR%): ATR_n / close."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n, min_periods=n).mean()
    return atr / close.replace(0, np.nan)


def rolling_rsi(close: pd.Series, n: int) -> pd.Series:
    """RSI clássico (Wilder), escala 0–100."""
    delta = close.diff()
    ganho = delta.clip(lower=0.0)
    perda = (-delta).clip(lower=0.0)
    avg_g = ganho.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_p = perda.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_g / avg_p.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def dist_ema(close: pd.Series, n: int) -> pd.Series:
    """Distância percentual do close à EMA_n: (close / EMA_n) - 1."""
    ema = close.ewm(span=n, adjust=False, min_periods=n).mean()
    return close / ema.replace(0, np.nan) - 1.0


def bollinger_pctb(close: pd.Series, n: int, k: float) -> pd.Series:
    """%B = (close - lower) / (upper - lower), com bandas em SMA ± k·σ."""
    sma = close.rolling(n, min_periods=n).mean()
    std = close.rolling(n, min_periods=n).std(ddof=1)
    upper = sma + k * std
    lower = sma - k * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de dados macro (IBOV + opcionais USDBRL e DI)
# ─────────────────────────────────────────────────────────────────────────────

def carregar_macro_csv(stem: str, calendario: pd.DatetimeIndex) -> pd.Series | None:
    """
    Carrega dados_diarios/<stem>.csv, alinha ao calendário do ativo via
    forward-fill e devolve a coluna 'close'. Retorna None se não existir.
    """
    fp = DATA_DIR / f"{stem}.csv"
    if not fp.exists():
        return None
    macro = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
    macro = macro.set_index("date")["close"].astype(float)
    return macro.reindex(calendario).ffill()


def garantir_usdbrl() -> bool:
    """Baixa dados_diarios/USDBRL.csv via yfinance se ainda não existir."""
    fp = DATA_DIR / f"{USDBRL_STEM}.csv"
    if fp.exists() or not AUTO_DOWNLOAD_USDBRL:
        return fp.exists()
    try:
        import yfinance as yf
        print(f"  baixando USDBRL via yfinance ...", flush=True)
        df = yf.download(
            "USDBRL=X", start="2018-01-01", interval="1d",
            auto_adjust=True, progress=False,
        )
        if df.empty:
            print("  [USDBRL] yfinance retornou vazio — feature será ignorada.")
            return False
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = USDBRL_STEM
        cols = ["date", "open", "high", "low", "close", "volume", "ticker"]
        cols = [c for c in cols if c in df.columns]
        df[cols].sort_values("date").to_csv(fp, index=False)
        print(f"  USDBRL salvo em {fp.name} ({len(df)} dias).")
        return True
    except Exception as exc:
        print(f"  [USDBRL] falha ao baixar ({exc}) — feature será ignorada.")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (cinco famílias da metodologia)
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Constrói matriz X com features estacionárias e devolve (DataFrame, cols).
    Espera colunas: date, open, high, low, close, volume, ibov_close;
    e opcionalmente usdbrl_close, di_close (já alinhados por data).

    Todas as features usam dados estritamente até t para prever r_{t+1}.
    """
    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"].values

    close      = df["close"].astype(float)
    high       = df["high"].astype(float)
    low        = df["low"].astype(float)
    ibov_close = df["ibov_close"].astype(float)

    r      = log_ret(close)
    r_ibov = log_ret(ibov_close)

    # ── § 2.1 — autoregressivas ──────────────────────────────────────────────

    for k in LAGS_RETORNO:
        out[f"lag_ret_{k}"] = r.shift(k)

    for n in MOMENTUM_JANELAS:
        out[f"mom_{n}"] = np.log(close / close.shift(n))

    # ── § 2.2 — risco e volatilidade ─────────────────────────────────────────

    for n in VOL_JANELAS:
        out[f"vol_{n}"] = r.rolling(n, min_periods=n).std(ddof=1)

    out[f"atr_{ATR_JANELA}"] = rolling_atr(high, low, close, ATR_JANELA)

    # ── § 2.2b — volume ──────────────────────────────────────────────────────

    volume = df["volume"].astype(float).replace(0, np.nan)

    # Volume relativo: desvio percentual em relação à média móvel.
    # Detecta picos de atividade que frequentemente antecedem movimentos relevantes.
    for n in VOL_VOLUME_JANELAS:
        vol_ma = volume.rolling(n, min_periods=n).mean()
        out[f"vol_rel_{n}"] = volume / vol_ma.replace(0, np.nan) - 1.0

    # On-Balance Volume z-score: acumulação/distribuição suavizada.
    # OBV = cumsum(sign(Δclose) × volume); z-score mensal remove tendência de escala.
    obv = (np.sign(close.diff()) * volume).cumsum()
    obv_ma  = obv.rolling(21, min_periods=21).mean()
    obv_std = obv.rolling(21, min_periods=21).std(ddof=1)
    out["obv_z21"] = (obv - obv_ma) / obv_std.replace(0, np.nan)

    # Force Index z-score: magnitude direcional (Δclose × volume), normalizada.
    # Forte quando grandes movimentos de preço são confirmados por alto volume.
    fi = close.diff() * volume
    fi_ma  = fi.rolling(21, min_periods=21).mean()
    fi_std = fi.rolling(21, min_periods=21).std(ddof=1)
    out["force_idx_z21"] = (fi - fi_ma) / fi_std.replace(0, np.nan)

    # VWAP rolling: preço médio ponderado pelo volume no período.
    # Distância do close ao VWAP indica pressão compradora/vendedora acumulada.
    for n in VOL_VWAP_JANELAS:
        vwap = (close * volume).rolling(n, min_periods=n).sum() / \
               volume.rolling(n, min_periods=n).sum()
        out[f"vwap_dist_{n}"] = close / vwap.replace(0, np.nan) - 1.0

    # ── § 2.3 — indicadores técnicos ─────────────────────────────────────────

    out[f"rsi_{RSI_JANELA}"] = rolling_rsi(close, RSI_JANELA)

    for n in EMA_JANELAS:
        out[f"dist_ema_{n}"] = dist_ema(close, n)

    out[f"bb_pctb_{BBANDS_JANELA}"] = bollinger_pctb(close, BBANDS_JANELA, BBANDS_K)

    # ── § 2.4 — contexto macroeconômico e de mercado ─────────────────────────

    # Retorno imediato do Ibovespa (t) como proxy de sentimento do dia
    out["macro_ibov_ret"] = r_ibov

    # Beta móvel: Cov(r_A, r_Ibov, N) / Var(r_Ibov, N)
    # Captura a sensibilidade sistêmica do ativo — ausente nos indicadores técnicos puros.
    for n in BETA_JANELAS:
        cov = r.rolling(n, min_periods=n).cov(r_ibov)
        var = r_ibov.rolling(n, min_periods=n).var(ddof=1)
        out[f"beta_{n}"] = cov / var.replace(0, np.nan)

    # Z-score do alpha diário: sinal de reversão à média relativa ao Ibovespa.
    # alpha_t = r_A_t - r_Ibov_t; z = (alpha - μ) / σ sobre janela N.
    for n in Z_ALPHA_JANELAS:
        alpha = r - r_ibov
        mu_a  = alpha.rolling(n, min_periods=n).mean()
        std_a = alpha.rolling(n, min_periods=n).std(ddof=1)
        out[f"z_alpha_{n}"] = (alpha - mu_a) / std_a.replace(0, np.nan)

    # Volatilidade realizada do Ibovespa: proxy do regime de risco do mercado.
    for n in RISCO_MERCADO_JANELAS:
        out[f"risco_mercado_{n}"] = r_ibov.rolling(n, min_periods=n).std(ddof=1)

    # Tendência acumulada do Ibovespa: contexto de tendência primária.
    for n in TREND_MERCADO_JANELAS:
        out[f"trend_mercado_{n}"] = np.log(ibov_close / ibov_close.shift(n))

    # Opcionais: câmbio e juros
    if "usdbrl_close" in df.columns:
        out["macro_usdbrl_ret"] = log_ret(df["usdbrl_close"].astype(float))
    if "di_close" in df.columns:
        out["macro_di_var"] = df["di_close"].astype(float).diff()

    # ── § 2.5 — sazonais (one-hot) ───────────────────────────────────────────

    dow = pd.to_datetime(df["date"]).dt.dayofweek
    for d, nome in enumerate(["seg", "ter", "qua", "qui", "sex"]):
        out[f"dow_{nome}"] = (dow == d).astype(np.int8)
    mes = pd.to_datetime(df["date"]).dt.month
    for m in range(1, 13):
        out[f"mes_{m:02d}"] = (mes == m).astype(np.int8)

    feature_cols = [c for c in out.columns if c != "date"]
    return out, feature_cols


def carregar_df(fp: Path) -> pd.DataFrame:
    """Lê CSV do ativo, anexa colunas macro opcionais (USDBRL, DI)."""
    df = pd.read_csv(fp, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    obrig = ["date", "open", "high", "low", "close", "volume", "ibov_close"]
    faltando = [c for c in obrig if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes — rode prepara_dados.py antes: {faltando}")

    cal = pd.to_datetime(df["date"])
    usdbrl = carregar_macro_csv(USDBRL_STEM, pd.DatetimeIndex(cal))
    if usdbrl is not None:
        df["usdbrl_close"] = usdbrl.values
    di = carregar_macro_csv(DI_STEM, pd.DatetimeIndex(cal))
    if di is not None:
        df["di_close"] = di.values

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_predict(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_folds: int,
    train_inicial: int,
    embargo: int = 0,
    normalize_target: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Janela expansiva: para cada fold k, treina em [0 : ini_te − embargo]
    e prediz em [ini_te : fim_te].

    O `embargo` (em número de linhas) exclui as últimas amostras do treino cujos
    labels se sobrepõem com o início do período de teste.  Para um horizonte h,
    usa-se embargo = h, garantindo que nenhum label de treino "veja" o futuro
    que o fold de teste pretende prever.

    Se `normalize_target=True`, o y de treino é z-scored (StandardScaler fitado
    apenas no treino de cada fold) antes de entrar no XGBoost.  A predição é
    desnormalizada para a escala original antes de calcular métricas, evitando
    que a regularização encolha sistematicamente as previsões para zero.

    Retorna: (índices testados, predições OOF correspondentes, métricas/fold).
    """
    n = len(X)
    if train_inicial >= n - n_folds:
        raise ValueError("Histórico insuficiente para walk-forward com esses parâmetros.")

    tam = (n - train_inicial) // n_folds
    if tam < 5:
        raise ValueError(f"Folds muito pequenos (tam={tam}). Reduza WF_N_FOLDS.")

    idx_oof  = np.empty(0, dtype=np.int64)
    pred_oof = np.empty(0, dtype=np.float64)
    folds_metrics: list[dict] = []

    for k in range(n_folds):
        ini_te = train_inicial + k * tam
        fim_te = ini_te + tam if k < n_folds - 1 else n   # último fold absorve resto

        # Embargo: remove as últimas `embargo` linhas do treino para evitar
        # que labels sobrepostos (horizonte h > 1) contaminem o período de teste.
        fim_tr = max(1, ini_te - embargo)
        X_tr, y_tr = X[:fim_tr], y[:fim_tr]
        X_te, y_te = X[ini_te:fim_te], y[ini_te:fim_te]

        if len(X_tr) < 50:
            raise ValueError(
                f"Fold {k+1}: treino muito curto após embargo ({len(X_tr)} amostras). "
                "Reduza WF_EMBARGO_FATOR ou aumente o histórico."
            )

        # validação interna (10% final do treino) para early stopping no fold
        cut = int(len(X_tr) * 0.90)
        X_fit, X_val = X_tr[:cut], X_tr[cut:]
        y_fit, y_val = y_tr[:cut], y_tr[cut:]

        # Normalização do target: z-score fitado APENAS no treino do fold.
        # Impede que a regularização do XGBoost encaminhe as predições para a
        # média global (≈ 0) quando o sinal-ruído é baixo.
        if normalize_target:
            scaler_y = StandardScaler()
            y_fit_in = scaler_y.fit_transform(y_fit.reshape(-1, 1)).ravel()
            y_val_in = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        else:
            scaler_y = None
            y_fit_in, y_val_in = y_fit, y_val

        model = XGBRegressor(**params)
        model.fit(X_fit, y_fit_in, eval_set=[(X_val, y_val_in)], verbose=False)
        pred_raw = model.predict(X_te)

        # Desnormaliza para escala original antes de métricas
        if scaler_y is not None:
            pred = scaler_y.inverse_transform(pred_raw.reshape(-1, 1)).ravel()
        else:
            pred = pred_raw

        idx_oof  = np.concatenate([idx_oof,  np.arange(ini_te, fim_te)])
        pred_oof = np.concatenate([pred_oof, pred])

        folds_metrics.append({
            "fold":      k + 1,
            "n_treino":  fim_tr,
            "n_embargo": embargo,
            "n_teste":   len(X_te),
            "rmse":      float(np.sqrt(mean_squared_error(y_te, pred))),
            "mae":       float(mean_absolute_error(y_te, pred)),
            "r2":        float(r2_score(y_te, pred)),
            "dir_acc":   float(np.mean(np.sign(pred) == np.sign(y_te))),
            "n_rounds":  int(getattr(model, "best_iteration", -1) + 1),
        })

    return idx_oof, pred_oof, folds_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Optuna — busca de hiperparâmetros (opcional)
# ─────────────────────────────────────────────────────────────────────────────

def otimizar_params(X_train: np.ndarray, y_train: np.ndarray, n_trials: int) -> tuple[dict, float]:
    tscv = TimeSeriesSplit(n_splits=OPTUNA_CV_FOLDS)

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators     = PARAM_SPACE["n_estimators"],
            learning_rate    = trial.suggest_float("learning_rate",    *PARAM_SPACE["learning_rate"], log=True),
            max_depth        = trial.suggest_int  ("max_depth",        *PARAM_SPACE["max_depth"]),
            subsample        = trial.suggest_float("subsample",        *PARAM_SPACE["subsample"]),
            colsample_bytree = trial.suggest_float("colsample_bytree", *PARAM_SPACE["colsample_bytree"]),
            min_child_weight = trial.suggest_int  ("min_child_weight", *PARAM_SPACE["min_child_weight"]),
            reg_alpha        = trial.suggest_float("reg_alpha",        *PARAM_SPACE["reg_alpha"]),
            reg_lambda       = trial.suggest_float("reg_lambda",       *PARAM_SPACE["reg_lambda"]),
            eval_metric      = "rmse",
            early_stopping_rounds = 30,
            random_state     = SEED,
            n_jobs           = -1,
            verbosity        = 0,
        )
        rmses = []
        for fold, (tr, va) in enumerate(tscv.split(X_train)):
            X_tr, X_va = X_train[tr], X_train[va]
            y_tr, y_va = y_train[tr], y_train[va]
            model = XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            pred = model.predict(X_va)
            rmses.append(float(np.sqrt(mean_squared_error(y_va, pred))))
            trial.report(float(np.mean(rmses)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses)) if rmses else 1.0

    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=2)
    study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update(dict(
        n_estimators          = PARAM_SPACE["n_estimators"],
        early_stopping_rounds = 50,
        eval_metric           = "rmse",
        random_state          = SEED,
        n_jobs                = -1,
        verbosity             = 0,
    ))
    return best, float(study.best_value)


# ─────────────────────────────────────────────────────────────────────────────
# Penalidade de incerteza por baixa acurácia direcional
# ─────────────────────────────────────────────────────────────────────────────

def penalizar_omega(omega_base: float, dir_acc_pct: float) -> tuple[float, float]:
    """
    Infla o omega_base quando a acurácia direcional fica abaixo de
    DIR_ACC_PENALIDADE_LIMIAR, reduzindo o peso da opinião Q no Black-Litterman.

    Fórmula:
        deficit  = max(0, limiar − dir_acc_pct) / 100
        fator    = exp(K × deficit)
        omega_aj = omega_base × fator

    Retorna (omega_ajustado, fator_penalidade).
    """
    deficit = max(0.0, DIR_ACC_PENALIDADE_LIMIAR - dir_acc_pct) / 100.0
    fator   = float(np.exp(DIR_ACC_PENALIDADE_K * deficit))
    return omega_base * fator, fator


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline por ticker
# ─────────────────────────────────────────────────────────────────────────────

def treinar(ticker: str, df: pd.DataFrame, horizon: int = 1) -> dict:
    """
    Treina o regressor XGBoost para o horizonte h (Direct Forecasting).
    Target: ln(P_{t+h} / P_t)  — retorno acumulado nos próximos h dias úteis.
    Salva arquivos com sufixo _{horizon}d (ex.: PETR4_h5d_predicoes.csv).
    """
    nome_h = HORIZON_NOMES.get(horizon, f"{horizon}d")
    prefixo = f"{ticker}_h{horizon}d"

    feat_df, feature_cols = build_features(df)

    # Target: retorno acumulado nos próximos h dias (sem look-ahead)
    close = df["close"].astype(float)
    feat_df["__y__"] = np.log(close.shift(-horizon) / close).values

    # Remove warmup (NaN nas features). As últimas h linhas têm y=NaN — usamos para Q.
    valid_X = feat_df[feature_cols].notna().all(axis=1)
    feat_df = feat_df[valid_X].reset_index(drop=True)

    # Linha de previsão de Q: a mais recente com X válido e y ainda desconhecido.
    pred_row = feat_df[feat_df["__y__"].isna()].tail(1).copy()

    # Conjunto supervisionado
    sup = feat_df[feat_df["__y__"].notna()].reset_index(drop=True)
    X = sup[feature_cols].to_numpy(dtype=np.float32)
    y = sup["__y__"].to_numpy(dtype=np.float64)
    dates  = pd.to_datetime(sup["date"]).to_numpy()
    closes = df.loc[df["date"].isin(sup["date"]), "close"].to_numpy(dtype=np.float32)

    n = len(X)
    if n < 250:
        raise ValueError(f"{ticker} h={horizon}d: histórico curto após limpeza (n={n}).")

    # ── tuning opcional (apenas dentro do treino inicial p/ evitar leakage) ──
    train_inicial = max(int(n * WF_TRAIN_INICIAL), 200)
    if OPTUNA_TRIALS > 0:
        print(f"      Optuna {OPTUNA_TRIALS} trials × {OPTUNA_CV_FOLDS} folds ...", end=" ", flush=True)
        params, cv_rmse = otimizar_params(X[:train_inicial], y[:train_inicial], OPTUNA_TRIALS)
        print(
            f"RMSE-CV={cv_rmse:.5f} | "
            f"lr={params['learning_rate']:.4f} | "
            f"depth={params['max_depth']} | "
            f"min_child={params['min_child_weight']}",
            flush=True,
        )
    else:
        params  = dict(XGB_DEFAULT)
        cv_rmse = float("nan")

    # ── walk-forward ──────────────────────────────────────────────────────
    embargo_dias = int(horizon * WF_EMBARGO_FATOR)
    idx_oof, pred_oof, folds_metrics = walk_forward_predict(
        X, y, params, n_folds=WF_N_FOLDS,
        train_inicial=train_inicial, embargo=embargo_dias,
        normalize_target=NORMALIZAR_TARGET,
    )
    y_oof = y[idx_oof]

    rmse_wf = float(np.sqrt(mean_squared_error(y_oof, pred_oof)))
    mae_wf  = float(mean_absolute_error(y_oof, pred_oof))
    r2_wf   = float(r2_score(y_oof, pred_oof))
    mse_wf  = float(mean_squared_error(y_oof, pred_oof))
    dir_acc = float(np.mean(np.sign(pred_oof) == np.sign(y_oof)) * 100)

    # ── modelo final em TODO o histórico supervisionado ───────────────────
    cut = int(n * 0.90)

    if NORMALIZAR_TARGET:
        # Scaler fitado apenas nos dados de treino (sem look-ahead sobre o val)
        scaler_y_final = StandardScaler()
        y_fit_final = scaler_y_final.fit_transform(y[:cut].reshape(-1, 1)).ravel()
        y_val_final  = scaler_y_final.transform(y[cut:].reshape(-1, 1)).ravel()
    else:
        scaler_y_final = None
        y_fit_final, y_val_final = y[:cut], y[cut:]

    final = XGBRegressor(**params)
    final.fit(X[:cut], y_fit_final, eval_set=[(X[cut:], y_val_final)], verbose=False)
    n_rounds_final = int(getattr(final, "best_iteration", -1) + 1)

    # ── predição Q ────────────────────────────────────────────────────────
    if not pred_row.empty:
        X_q = pred_row[feature_cols].to_numpy(dtype=np.float32)
        q_raw = float(final.predict(X_q)[0])
        q_data_base = pd.to_datetime(pred_row["date"].iloc[0]).date().isoformat()
    else:
        q_raw = float(final.predict(X[-1:])[0])
        q_data_base = pd.to_datetime(sup["date"].iloc[-1]).date().isoformat()

    # Desnormaliza a predição do Q para a escala de log-retorno original
    if scaler_y_final is not None:
        q_log = float(scaler_y_final.inverse_transform([[q_raw]])[0][0])
    else:
        q_log = q_raw

    q_simples = float(np.expm1(q_log))

    # ── salva predições walk-forward ──────────────────────────────────────
    pred_df = pd.DataFrame({
        "date":   dates[idx_oof],
        "ticker": ticker,
        "close":  closes[idx_oof],
        "y_real": y_oof,
        "y_pred": pred_oof,
        "erro":   y_oof - pred_oof,
        "fold":   np.concatenate([
            np.full(m["n_teste"], m["fold"], dtype=np.int32) for m in folds_metrics
        ]),
    }).sort_values("date").reset_index(drop=True)
    pred_df.to_csv(OUT_DIR / f"{prefixo}_predicoes.csv", index=False)

    # ── salva modelo e importância ────────────────────────────────────────
    joblib.dump(final, OUT_DIR / f"{prefixo}_model.pkl")
    if scaler_y_final is not None:
        joblib.dump(scaler_y_final, OUT_DIR / f"{prefixo}_scaler_y.pkl")

    importances = pd.Series(final.feature_importances_, index=feature_cols)
    imp_df = (
        importances.reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
        .sort_values("importance", ascending=False)
    )
    imp_df.columns = ["feature", "importance"]
    imp_df.to_csv(OUT_DIR / f"{prefixo}_importancia.csv", index=False)

    # ── plot 4 painéis ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"XGBoost — {ticker}  | horizonte: {nome_h}  (Walk-Forward)\n"
        f"RMSE: {rmse_wf:.5f}  |  MAE: {mae_wf:.5f}  |  R²: {r2_wf:.4f}  |  "
        f"Direção: {dir_acc:.1f}%  |  Q = {q_log:+.4%} (log)  ≈ {q_simples:+.4%} (simples)",
        fontsize=11,
    )
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])

    ax0.plot(pred_df["date"], pred_df["y_real"], color="black", lw=0.8, label="real")
    ax0.plot(pred_df["date"], pred_df["y_pred"], color="tab:orange", lw=0.8, label="previsto")
    ax0.axhline(0, color="gray", lw=0.6, ls="--")
    ax0.set_ylabel(f"log-retorno t+{horizon}")
    ax0.set_title("Walk-Forward: real × previsto")
    ax0.legend(fontsize=9, loc="upper left")
    ax0.grid(True, alpha=0.20)

    ax1.plot(pred_df["date"], pred_df["close"], color="tab:blue", lw=1)
    ax1.set_ylabel("Close (R$)")
    ax1.grid(True, alpha=0.20)
    pos = pred_df["y_pred"] > 0
    for d_ini, d_fim, p in zip(pred_df["date"][:-1], pred_df["date"][1:], pos[:-1]):
        ax1.axvspan(d_ini, d_fim, alpha=0.06,
                    color="tab:green" if p else "tab:red", lw=0)
    ax1.set_title("Sinal direcional (verde = prevê alta)")

    ax2.scatter(pred_df["y_real"], pred_df["y_pred"], s=8, alpha=0.45, color="tab:purple")
    lim = float(np.nanmax(np.abs(np.concatenate([pred_df["y_real"], pred_df["y_pred"]])))) * 1.05
    ax2.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    ax2.axhline(0, color="gray", lw=0.5, ls=":")
    ax2.axvline(0, color="gray", lw=0.5, ls=":")
    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
    ax2.set_xlabel("y real"); ax2.set_ylabel("y previsto")
    ax2.set_title("Dispersão (OOF)")
    ax2.grid(True, alpha=0.25)

    top = imp_df.head(15)[::-1]
    ax3.barh(top["feature"], top["importance"], color="steelblue")
    ax3.set_title("Top-15 importância")
    ax3.set_xlabel("Importância (gain)")
    ax3.grid(True, alpha=0.25, axis="x")

    plt.savefig(OUT_DIR / f"{prefixo}_plot.png", dpi=140, bbox_inches="tight")
    plt.close()

    return {
        "ticker":         ticker,
        "horizon":        horizon,
        "horizon_nome":   nome_h,
        "n_total":        n,
        "n_oof":          int(len(y_oof)),
        "rmse_wf":        rmse_wf,
        "mae_wf":         mae_wf,
        "r2_wf":          r2_wf,
        "mse_wf":         mse_wf,
        "dir_acc_pct":    dir_acc,
        "rmse_cv_optuna": round(cv_rmse, 6) if not np.isnan(cv_rmse) else None,
        "n_rounds_final": n_rounds_final,
        "q_log":          q_log,
        "q_simples":      q_simples,
        "q_data_base":    q_data_base,
        "omega_base":     mse_wf,
        "lr":             round(params.get("learning_rate", float("nan")), 4),
        "max_depth":      params.get("max_depth", "—"),
    }


def _plot_comparacao_horizontes(ticker: str, resultados: list[dict], h_vencedor: int) -> None:
    """
    Gráfico de barras comparando RMSE, MAE e acurácia direcional entre horizontes.
    O horizonte vencedor é destacado.
    """
    nomes  = [r["horizon_nome"] for r in resultados]
    rmses  = [r["rmse_wf"]     for r in resultados]
    maes   = [r["mae_wf"]      for r in resultados]
    dirs   = [r["dir_acc_pct"] for r in resultados]
    cores  = [
        "tab:green" if r["horizon"] == h_vencedor else "tab:blue"
        for r in resultados
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"Comparação de Horizontes — {ticker}\n"
        f"Vencedor: {HORIZON_NOMES.get(h_vencedor, f'{h_vencedor}d')} "
        f"(critério: {HORIZON_CRITERIO})",
        fontsize=12,
    )

    axes[0].bar(nomes, rmses, color=cores)
    axes[0].set_title("RMSE (Walk-Forward)")
    axes[0].set_ylabel("RMSE"); axes[0].grid(True, alpha=0.25, axis="y")

    axes[1].bar(nomes, maes, color=cores)
    axes[1].set_title("MAE (Walk-Forward)")
    axes[1].set_ylabel("MAE"); axes[1].grid(True, alpha=0.25, axis="y")

    axes[2].bar(nomes, dirs, color=cores)
    axes[2].axhline(50, color="red", ls="--", lw=0.8, label="aleatório (50%)")
    axes[2].set_title("Acurácia Direcional (%)")
    axes[2].set_ylabel("Dir. Acc. (%)"); axes[2].grid(True, alpha=0.25, axis="y")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{ticker}_horizontes_comparacao.png", dpi=140, bbox_inches="tight")
    plt.close()


def comparar_horizontes(ticker: str, df: pd.DataFrame) -> tuple[list[dict], dict]:
    """
    Executa treinar() para cada horizonte em HORIZONS (Direct Forecasting independente).
    Elege o vencedor pelo HORIZON_CRITERIO e copia seus artefatos como arquivos padrão
    (sem sufixo de horizonte), que são os insumos do Black-Litterman.

    Retorna: (lista de métricas por horizonte, métricas do vencedor).
    """
    resultados: list[dict] = []
    for h in HORIZONS:
        nome_h = HORIZON_NOMES.get(h, f"{h}d")
        print(f"    horizonte {nome_h} ...", end=" ", flush=True)
        try:
            m = treinar(ticker, df, horizon=h)
            resultados.append(m)
            print(
                f"RMSE={m['rmse_wf']:.5f}  DirAcc={m['dir_acc_pct']:.1f}%",
                flush=True,
            )
        except Exception as exc:
            print(f"ERRO: {exc}", flush=True)

    if not resultados:
        raise RuntimeError(f"{ticker}: falhou em todos os horizontes.")

    # Seleciona o vencedor
    if HORIZON_CRITERIO in HORIZON_MENOR_MELHOR:
        vencedor = min(resultados, key=lambda m: m[HORIZON_CRITERIO])
    else:
        vencedor = max(resultados, key=lambda m: m[HORIZON_CRITERIO])
    h_win = vencedor["horizon"]

    # Aplica penalidade de omega por baixa acurácia direcional
    omega_aj, fator = penalizar_omega(vencedor["omega_base"], vencedor["dir_acc_pct"])
    vencedor["omega_ajustado"]   = omega_aj
    vencedor["penalidade_fator"] = round(fator, 4)

    # Copia artefatos do vencedor como arquivos padrão (sem sufixo _h{h}d)
    for sufixo in ("_predicoes.csv", "_model.pkl", "_importancia.csv", "_plot.png"):
        src = OUT_DIR / f"{ticker}_h{h_win}d{sufixo}"
        dst = OUT_DIR / f"{ticker}{sufixo}"
        if src.exists():
            shutil.copy2(src, dst)

    # Plot de comparação entre horizontes
    if len(resultados) > 1:
        _plot_comparacao_horizontes(ticker, resultados, h_win)

    for m in resultados:
        m["vencedor"] = (m["horizon"] == h_win)

    return resultados, vencedor


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(SEED)

    garantir_usdbrl()

    csvs = sorted(p for p in DATA_DIR.glob("*.csv")
                  if p.stem not in {IBOV_STEM, USDBRL_STEM, DI_STEM})
    if not csvs:
        print(f"Nenhum CSV de ativo em {DATA_DIR}. Rode prepara_dados.py primeiro.")
        sys.exit(1)
    if TICKERS != "all":
        alvo = set(TICKERS)
        csvs = [p for p in csvs if p.stem in alvo]

    horizontes_str = " / ".join(HORIZON_NOMES.get(h, f"{h}d") for h in HORIZONS)
    print(
        f"Regressor XGBoost para vetor Q | tickers: {len(csvs)} | "
        f"horizontes: {horizontes_str} | critério: {HORIZON_CRITERIO}\n"
        f"WF: {WF_N_FOLDS} folds, treino inicial = {WF_TRAIN_INICIAL:.0%} | "
        f"Optuna: {OPTUNA_TRIALS} trials\n"
    )

    # all_h_metrics: todos os resultados (ticker × horizonte)
    # winner_metrics: apenas o vencedor por ticker → entra no vetor Q
    all_h_metrics: list[dict] = []
    winner_metrics: list[dict] = []
    erros: list[tuple[str, str]] = []

    for fp in csvs:
        ticker = fp.stem
        print(f"[{ticker}] {len(pd.read_csv(fp, usecols=['date']))} dias úteis", flush=True)
        try:
            df = carregar_df(fp)
            resultados, vencedor = comparar_horizontes(ticker, df)
            all_h_metrics.extend(resultados)
            winner_metrics.append(vencedor)
            nome_win = HORIZON_NOMES.get(vencedor["horizon"], f"{vencedor['horizon']}d")
            fator    = vencedor["penalidade_fator"]
            pen_str  = f"  [!] penalidade x{fator:.2f}" if fator > 1.0 else ""
            print(
                f"  -> vencedor: {nome_win} | "
                f"DirAcc={vencedor['dir_acc_pct']:.1f}%  "
                f"RMSE={vencedor['rmse_wf']:.5f}  "
                f"Q={vencedor['q_simples']:+.3%}  "
                f"omega={vencedor['omega_base']:.2e}  "
                f"omega_aj={vencedor['omega_ajustado']:.2e}"
                f"{pen_str}",
                flush=True,
            )
        except Exception as exc:
            erros.append((ticker, str(exc)))
            print(f"  ERRO: {exc}", flush=True)

    if not winner_metrics:
        print("\nNenhum modelo treinado com sucesso.")
        return

    # ── consolida métricas completas (todos os horizontes) ────────────────
    all_h_df = (
        pd.DataFrame(all_h_metrics)
        .sort_values(["ticker", "horizon"])
        .reset_index(drop=True)
    )
    all_h_df.to_csv(OUT_DIR / "metricas_todos_horizontes.csv", index=False)

    # métricas dos vencedores
    winner_df = (
        pd.DataFrame(winner_metrics)
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    # Guarda snapshot da rodada anterior para comparação (se existir)
    arq_anterior = OUT_DIR / "metricas_vencedores.csv"
    snapshot_anterior: pd.DataFrame | None = None
    if arq_anterior.exists():
        snapshot_anterior = pd.read_csv(arq_anterior)
        shutil.copy2(arq_anterior, OUT_DIR / "metricas_vencedores_anterior.csv")

    winner_df.to_csv(arq_anterior, index=False)

    # ── vetor Q (entrada do Black-Litterman) — apenas vencedores ─────────
    q_df = (
        winner_df[["ticker", "horizon_nome", "q_data_base", "q_log", "q_simples"]]
        .rename(columns={"horizon_nome": "horizonte_vencedor"})
    )
    q_df.to_csv(OUT_DIR / "q_vetor.csv", index=False)

    # ── omega_base — apenas vencedores ────────────────────────────────────
    omega_df = (
        winner_df[[
            "ticker", "horizon_nome", "dir_acc_pct",
            "mse_wf", "omega_ajustado", "penalidade_fator",
            "rmse_wf", "mae_wf",
        ]]
        .rename(columns={
            "mse_wf":       "omega_base",
            "horizon_nome": "horizonte_vencedor",
        })
    )
    omega_df.to_csv(OUT_DIR / "omega_base.csv", index=False)

    # ── resumo no terminal ────────────────────────────────────────────────
    print("\n── Vencedores por ticker (para o vetor Q) ─────────────────────")
    cols = ["ticker", "horizon_nome", "dir_acc_pct", "rmse_wf",
            "r2_wf", "q_simples", "omega_base", "omega_ajustado", "penalidade_fator"]
    print(winner_df[cols].round(6).to_string(index=False))

    print("\n── Comparação entre horizontes (todos os tickers) ─────────────")
    pivot = all_h_df.pivot_table(
        index="ticker", columns="horizon_nome",
        values="dir_acc_pct", aggfunc="first",
    )
    print(pivot.round(1).to_string())

    print(f"\nVetor Q    -> {OUT_DIR / 'q_vetor.csv'}")
    print(f"omega_base -> {OUT_DIR / 'omega_base.csv'}")
    print(f"Todos horizontes -> {OUT_DIR / 'metricas_todos_horizontes.csv'}")

    # ── comparação com rodada anterior (ex: sem Optuna vs com Optuna) ─────
    if snapshot_anterior is not None and "dir_acc_pct" in snapshot_anterior.columns:
        _comparar_rodadas(snapshot_anterior, winner_df)

    # ── plot consolidado de comparação entre horizontes ───────────────────
    _plot_consolidado_horizontes(all_h_df)

    # ── plot consolidado de feature importance (artefatos dos vencedores) ─
    imp_csvs = [
        OUT_DIR / f"{m['ticker']}_importancia.csv"
        for m in winner_metrics
        if (OUT_DIR / f"{m['ticker']}_importancia.csv").exists()
    ]
    if imp_csvs:
        imp_all = pd.concat(
            [pd.read_csv(f) for f in imp_csvs], ignore_index=True
        )
        imp_top = (
            imp_all.groupby("feature")["importance"].mean()
            .sort_values(ascending=False)
            .head(20)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        imp_top.iloc[::-1].plot(kind="barh", ax=ax, color="seagreen")
        ax.set_title(
            f"Top-20 features — modelos vencedores ({len(imp_csvs)} tickers)\n"
            f"critério de seleção: {HORIZON_CRITERIO}"
        )
        ax.set_xlabel("Importância média (gain)")
        ax.grid(True, alpha=0.25, axis="x")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "importancia_consolidada.png", dpi=140, bbox_inches="tight")
        plt.close()

    if erros:
        print("\nErros:")
        for t, msg in erros:
            print(f"  {t}: {msg}")


def _comparar_rodadas(anterior: pd.DataFrame, atual: pd.DataFrame) -> None:
    """
    Compara Dir Acc e RMSE entre a rodada anterior (ex: sem Optuna) e a atual.
    Salva comparacao_rodadas.csv e imprime diff no terminal.
    """
    cols_merge = ["ticker", "dir_acc_pct", "rmse_wf", "omega_ajustado"]
    cols_merge = [c for c in cols_merge if c in anterior.columns and c in atual.columns]

    comp = anterior[["ticker", "dir_acc_pct", "rmse_wf"]].merge(
        atual[["ticker", "dir_acc_pct", "rmse_wf"]],
        on="ticker", suffixes=("_ant", "_atual"),
    )
    comp["delta_dir_acc"] = comp["dir_acc_pct_atual"] - comp["dir_acc_pct_ant"]
    comp["delta_rmse"]    = comp["rmse_wf_atual"]     - comp["rmse_wf_ant"]
    comp = comp.sort_values("delta_dir_acc", ascending=False).reset_index(drop=True)
    comp.to_csv(OUT_DIR / "comparacao_rodadas.csv", index=False)

    print("\n-- Comparacao com rodada anterior ----------------------------------")
    print(f"  {'ticker':<8}  {'DirAcc ant':>10}  {'DirAcc atual':>12}  {'dDirAcc':>9}  {'dRMSE':>9}")
    print("  " + "-" * 56)
    ganhos = perdas = 0
    for _, r in comp.iterrows():
        sinal = "+" if r["delta_dir_acc"] > 0 else ("-" if r["delta_dir_acc"] < 0 else "=")
        if r["delta_dir_acc"] > 0:
            ganhos += 1
        elif r["delta_dir_acc"] < 0:
            perdas += 1
        print(
            f"  {r['ticker']:<8}  {r['dir_acc_pct_ant']:>10.2f}%  "
            f"{r['dir_acc_pct_atual']:>11.2f}%  "
            f"{sinal} {abs(r['delta_dir_acc']):>6.2f}pp  "
            f"{r['delta_rmse']:>+8.5f}"
        )
    media_delta = comp["delta_dir_acc"].mean()
    print("  " + "-" * 56)
    print(
        f"  Media: dDirAcc = {media_delta:+.2f}pp  |  "
        f"melhoraram: {ganhos}  |  pioraram: {perdas}  |  iguais: {len(comp)-ganhos-perdas}"
    )
    print(f"  Arquivo: {OUT_DIR / 'comparacao_rodadas.csv'}")

    # plot visual
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Comparação: rodada anterior × atual\n(ordenado por Δ Dir Acc)", fontsize=12)

    cores = ["tab:green" if v > 0 else ("tab:red" if v < 0 else "tab:gray")
             for v in comp["delta_dir_acc"]]
    ax1.barh(comp["ticker"], comp["delta_dir_acc"], color=cores)
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_title("Δ Acurácia Direcional (pp)")
    ax1.set_xlabel("Δ Dir Acc (pp)")
    ax1.grid(True, alpha=0.25, axis="x")

    cores2 = ["tab:green" if v < 0 else ("tab:red" if v > 0 else "tab:gray")
              for v in comp["delta_rmse"]]
    ax2.barh(comp["ticker"], comp["delta_rmse"], color=cores2)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_title("Δ RMSE (verde = melhorou)")
    ax2.set_xlabel("Δ RMSE")
    ax2.grid(True, alpha=0.25, axis="x")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparacao_rodadas.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Plot:    {OUT_DIR / 'comparacao_rodadas.png'}")


def _plot_consolidado_horizontes(all_h_df: pd.DataFrame) -> None:
    """
    Plot consolidado: acurácia direcional média por horizonte (todos os tickers).
    Ajuda a identificar qual horizonte tem sinal mais forte para a carteira.
    """
    horizons_presentes = sorted(all_h_df["horizon"].unique())
    nomes   = [HORIZON_NOMES.get(h, f"{h}d") for h in horizons_presentes]
    dir_med = [
        all_h_df[all_h_df["horizon"] == h]["dir_acc_pct"].mean()
        for h in horizons_presentes
    ]
    rmse_med = [
        all_h_df[all_h_df["horizon"] == h]["rmse_wf"].mean()
        for h in horizons_presentes
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Comparação de Horizontes — Consolidado ({all_h_df['ticker'].nunique()} tickers)\n"
        f"critério de seleção por ticker: {HORIZON_CRITERIO}",
        fontsize=12,
    )

    cores = ["tab:blue", "tab:orange", "tab:green"][: len(horizons_presentes)]

    bars1 = ax1.bar(nomes, dir_med, color=cores)
    ax1.axhline(50, color="red", ls="--", lw=0.8, label="aleatório (50%)")
    ax1.set_title("Acurácia Direcional Média (%)")
    ax1.set_ylabel("Dir. Acc. (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars1, dir_med):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    bars2 = ax2.bar(nomes, rmse_med, color=cores)
    ax2.set_title("RMSE Médio (Walk-Forward)")
    ax2.set_ylabel("RMSE")
    ax2.grid(True, alpha=0.25, axis="y")
    for bar, val in zip(bars2, rmse_med):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.00001,
                 f"{val:.5f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "horizontes_consolidado.png", dpi=140, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro fatal: {exc}")
        sys.exit(1)
