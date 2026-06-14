from __future__ import annotations

"""
criacao_modelo_xgb.py — Regressor XGBoost para o vetor Q (Black-Litterman)
--------------------------------------------------------------------------
Objetivo: prever o log-retorno de cada ativo no período t+h e usar o
output como vetor Q (opiniões) do modelo de Black-Litterman.

Melhorias implementadas:
    A. Validação:
        - WF_INTERNAL_SPLIT = 0.80 (era 0.90): mais amostras na validação interna
        - early_stopping_rounds = 20 (era 50): menos overfitting no early stopping
        - HORIZONS = [1, 5, 10, 21]: testa múltiplos horizontes
        - USE_SAMPLE_WEIGHTS: pesos exponenciais por tempo (obs recentes > antigas)
    B. Target:
        - USE_CLASSIFICATION: modo 3 classes (down/neutral/up) paralelo ao regressor;
          o Q é escolhido pela melhor acurácia direcional OOF
        - USE_QUANTILE: regressão por quantil (mediana) — mais robusta a outliers
    C. Features:
        - MACRO_SETORIAL: macro específica por ativo (brent, iron_ore, etc.)
        - CS_RANK_JANELAS: rank percentil cross-seccional dos retornos entre tickers
    D. Ensemble:
        - USE_LGB_ENSEMBLE: média simples XGBoost + LightGBM (requer lightgbm)
    E. Análise:
        - USE_SHAP: valores SHAP por feature salvo em <prefixo>_shap.csv
"""

import shutil
import sys
from pathlib import Path

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
from xgboost import XGBClassifier, XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import shap as _shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dados_diarios"
OUT_DIR  = ROOT / "criacao_modelo_xgb"
OUT_DIR.mkdir(exist_ok=True)

IBOV_STEM   = "IBOV"
USDBRL_STEM = "USDBRL"
DI_STEM     = "DI"

# ── seleção de tickers ─────────────────────────────────────────────────────────
TICKERS: list[str] | str = "all"

# ── § 2.1 — autoregressivas ────────────────────────────────────────────────────
LAGS_RETORNO     = [1, 5, 21]
MOMENTUM_JANELAS = [21, 63, 126, 252]

# ── § 2.2 — risco/volatilidade ─────────────────────────────────────────────────
VOL_JANELAS = [21, 63]
ATR_JANELA  = 21

# ── § 2.3 — volume ─────────────────────────────────────────────────────────────
VOL_VOLUME_JANELAS = [21, 63]
VOL_VWAP_JANELAS   = [21]

# ── § 2.4 — indicadores técnicos ──────────────────────────────────────────────
RSI_JANELA    = 21
EMA_JANELAS   = [21, 63]
BBANDS_JANELA = 21
BBANDS_K      = 2.0

# ── § 2.5 — contexto de mercado ────────────────────────────────────────────────
BETA_JANELAS          = [21, 63]
Z_ALPHA_JANELAS       = [21, 63]
RISCO_MERCADO_JANELAS = [21, 63]
TREND_MERCADO_JANELAS = [21, 63]

# ── § 2.7 — macro setorial ────────────────────────────────────────────────────
# Para cada ticker, lista de stems de CSV em dados_diarios/ a incluir como features.
# Esses arquivos são opcionais — ignorados se não existirem.
# Formato do CSV: date, close (colunas mínimas).
MACRO_SETORIAL: dict[str, list[str]] = {
    "PETR3":  ["brent"],
    "PETR4":  ["brent"],
    "PRIO3":  ["brent"],
    "VALE3":  ["iron_ore"],
    "GGBR4":  ["iron_ore"],
    "USIM5":  ["iron_ore"],
    "SUZB3":  ["cellulose"],
    "CSAN3":  ["brent", "sugar"],
}

# ── § 2.8 — rank cross-seccional ──────────────────────────────────────────────
# Janelas para o percentil de retorno do ativo entre todos os ativos do universo.
CS_RANK_JANELAS = [21, 63, 126]

AUTO_DOWNLOAD_USDBRL = True

# ── horizontes de previsão ─────────────────────────────────────────────────────
HORIZONS         = [21]
HORIZON_NOMES    = {1: "1d", 5: "1sem", 10: "2sem", 21: "1mes"}
HORIZON_CRITERIO = "dir_acc_pct"
HORIZON_MENOR_MELHOR = {"rmse_wf", "mae_wf"}

# ── penalidade de omega por baixa acurácia direcional ─────────────────────────
DIR_ACC_PENALIDADE_LIMIAR = 52.0
DIR_ACC_PENALIDADE_K      = 10.0

# ── walk-forward ───────────────────────────────────────────────────────────────
WF_TRAIN_INICIAL  = 0.60
WF_INTERNAL_SPLIT = 0.80   # treino/val interno para early stopping (era 0.90)
WF_N_FOLDS        = 5
WF_EMBARGO_FATOR  = 1.0
SEED              = 42

# ── normalização do target ─────────────────────────────────────────────────────
NORMALIZAR_TARGET = True

# ── pesos amostrais com decaimento temporal ────────────────────────────────────
# w_t = exp(λ × t/n), normalizado pela média.
# λ=5 → razão último/primeiro obs ≈ e^5 ≈ 148×. λ=0 desativa (pesos iguais).
USE_SAMPLE_WEIGHTS   = True
SAMPLE_WEIGHT_LAMBDA = 5.0

# ── modo classificação 3 classes (opcional) ────────────────────────────────────
# Treina XGBClassifier (down / neutral / up) em walk-forward paralelo ao regressor.
# Q_clf = P(up)×μ_up + P(neutral)×μ_neutral + P(down)×μ_down (retorno esperado).
# O modelo com maior dir_acc OOF vence e seus artefatos vão para o Black-Litterman.
USE_CLASSIFICATION = False

# ── regressão por quantil (opcional) ──────────────────────────────────────────
# Usa objective='reg:quantileerror' (mediana, α=0.5) em vez de MSE.
# Mais robusto a outliers de retorno — recomendado quando R² << 0.
USE_QUANTILE = False

# ── ensemble LightGBM (opcional) ──────────────────────────────────────────────
# Treina LightGBM em cada fold e no modelo final; predição = média(XGB, LGB).
# Requer: pip install lightgbm
USE_LGB_ENSEMBLE = False

# ── SHAP ───────────────────────────────────────────────────────────────────────
# Salva <prefixo>_shap.csv com importância média |SHAP| por feature.
# Requer: pip install shap
USE_SHAP = True

# ── Optuna ─────────────────────────────────────────────────────────────────────
OPTUNA_TRIALS   = 0
OPTUNA_CV_FOLDS = 4
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
    early_stopping_rounds = 20,    # reduzido de 50 → menos risco de early stop prematuro
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
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean() / close.replace(0, np.nan)


def rolling_rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    avg_g = delta.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_p = (-delta).clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_g / avg_p.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def dist_ema(close: pd.Series, n: int) -> pd.Series:
    ema = close.ewm(span=n, adjust=False, min_periods=n).mean()
    return close / ema.replace(0, np.nan) - 1.0


def bollinger_pctb(close: pd.Series, n: int, k: float) -> pd.Series:
    sma   = close.rolling(n, min_periods=n).mean()
    std   = close.rolling(n, min_periods=n).std(ddof=1)
    upper = sma + k * std
    lower = sma - k * std
    return (close - lower) / (upper - lower).replace(0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de dados macro
# ─────────────────────────────────────────────────────────────────────────────

def carregar_macro_csv(stem: str, calendario: pd.DatetimeIndex) -> pd.Series | None:
    fp = DATA_DIR / f"{stem}.csv"
    if not fp.exists():
        return None
    macro = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
    macro = macro.set_index("date")["close"].astype(float)
    return macro.reindex(calendario).ffill()


def garantir_usdbrl() -> bool:
    fp = DATA_DIR / f"{USDBRL_STEM}.csv"
    if fp.exists() or not AUTO_DOWNLOAD_USDBRL:
        return fp.exists()
    try:
        import yfinance as yf
        print("  baixando USDBRL via yfinance ...", end=" ", flush=True)
        df = yf.download("USDBRL=X", start="2018-01-01", interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            print("retornou vazio — feature ignorada.")
            return False
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["ticker"] = USDBRL_STEM
        cols = [c for c in ["date", "open", "high", "low", "close", "volume", "ticker"]
                if c in df.columns]
        df[cols].sort_values("date").to_csv(fp, index=False)
        print(f"salvo ({len(df)} dias).")
        return True
    except Exception as exc:
        print(f"falha ({exc}) — feature ignorada.")
        return False


def carregar_df(fp: Path) -> pd.DataFrame:
    """Lê CSV do ativo; anexa macro global (USDBRL, DI) e macro setorial."""
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

    # Macro setorial: carrega CSVs específicos do ticker (ex: brent para PETR3)
    ticker = fp.stem
    for stem in MACRO_SETORIAL.get(ticker, []):
        s = carregar_macro_csv(stem, pd.DatetimeIndex(cal))
        if s is not None:
            df[f"{stem}_close"] = s.values

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Constrói matriz X com features estacionárias.
    Aceita colunas opcionais no df:
      usdbrl_close, di_close         — macro global
      <stem>_close                   — macro setorial (brent, iron_ore, etc.)
      cs_rank_<h>                    — rank cross-seccional pré-calculado
    """
    out = pd.DataFrame(index=df.index)
    out["date"] = df["date"].values

    close      = df["close"].astype(float)
    high       = df["high"].astype(float)
    low        = df["low"].astype(float)
    ibov_close = df["ibov_close"].astype(float)
    r          = log_ret(close)
    r_ibov     = log_ret(ibov_close)

    # § 2.1 — autoregressivas
    for k in LAGS_RETORNO:
        out[f"lag_ret_{k}"] = r.shift(k)
    for n in MOMENTUM_JANELAS:
        out[f"mom_{n}"] = np.log(close / close.shift(n))

    # § 2.2 — risco/volatilidade
    for n in VOL_JANELAS:
        out[f"vol_{n}"] = r.rolling(n, min_periods=n).std(ddof=1)
    out[f"atr_{ATR_JANELA}"] = rolling_atr(high, low, close, ATR_JANELA)

    # § 2.3 — volume
    volume = df["volume"].astype(float).replace(0, np.nan)
    for n in VOL_VOLUME_JANELAS:
        vol_ma = volume.rolling(n, min_periods=n).mean()
        out[f"vol_rel_{n}"] = volume / vol_ma.replace(0, np.nan) - 1.0
    obv     = (np.sign(close.diff()) * volume).cumsum()
    obv_ma  = obv.rolling(21, min_periods=21).mean()
    obv_std = obv.rolling(21, min_periods=21).std(ddof=1)
    out["obv_z21"] = (obv - obv_ma) / obv_std.replace(0, np.nan)
    fi     = close.diff() * volume
    fi_ma  = fi.rolling(21, min_periods=21).mean()
    fi_std = fi.rolling(21, min_periods=21).std(ddof=1)
    out["force_idx_z21"] = (fi - fi_ma) / fi_std.replace(0, np.nan)
    for n in VOL_VWAP_JANELAS:
        vwap = (close * volume).rolling(n, min_periods=n).sum() / \
               volume.rolling(n, min_periods=n).sum()
        out[f"vwap_dist_{n}"] = close / vwap.replace(0, np.nan) - 1.0

    # § 2.4 — indicadores técnicos
    out[f"rsi_{RSI_JANELA}"] = rolling_rsi(close, RSI_JANELA)
    for n in EMA_JANELAS:
        out[f"dist_ema_{n}"] = dist_ema(close, n)
    out[f"bb_pctb_{BBANDS_JANELA}"] = bollinger_pctb(close, BBANDS_JANELA, BBANDS_K)

    # § 2.5 — contexto de mercado
    out["macro_ibov_ret"] = r_ibov
    for n in BETA_JANELAS:
        cov = r.rolling(n, min_periods=n).cov(r_ibov)
        var = r_ibov.rolling(n, min_periods=n).var(ddof=1)
        out[f"beta_{n}"] = cov / var.replace(0, np.nan)
    for n in Z_ALPHA_JANELAS:
        alpha = r - r_ibov
        mu_a  = alpha.rolling(n, min_periods=n).mean()
        std_a = alpha.rolling(n, min_periods=n).std(ddof=1)
        out[f"z_alpha_{n}"] = (alpha - mu_a) / std_a.replace(0, np.nan)
    for n in RISCO_MERCADO_JANELAS:
        out[f"risco_mercado_{n}"] = r_ibov.rolling(n, min_periods=n).std(ddof=1)
    for n in TREND_MERCADO_JANELAS:
        out[f"trend_mercado_{n}"] = np.log(ibov_close / ibov_close.shift(n))

    # § 2.6 — macro global opcional
    if "usdbrl_close" in df.columns:
        out["macro_usdbrl_ret"] = log_ret(df["usdbrl_close"].astype(float))
    if "di_close" in df.columns:
        out["macro_di_var"] = df["di_close"].astype(float).diff()

    # § 2.7 — macro setorial: colunas no formato <stem>_close (ex: brent_close)
    _reservadas = {"ibov_close", "usdbrl_close", "di_close", "close"}
    for col in df.columns:
        if col.endswith("_close") and col not in _reservadas:
            stem_name = col[: -len("_close")]
            out[f"macro_{stem_name}_ret"] = log_ret(df[col].astype(float))

    # § 2.8 — rank cross-seccional pré-calculado (colunas cs_rank_*)
    for col in df.columns:
        if col.startswith("cs_rank_"):
            out[col] = df[col].values

    # § 2.9 — sazonais (one-hot)
    dow = pd.to_datetime(df["date"]).dt.dayofweek
    for d, nome in enumerate(["seg", "ter", "qua", "qui", "sex"]):
        out[f"dow_{nome}"] = (dow == d).astype(np.int8)
    mes = pd.to_datetime(df["date"]).dt.month
    for m in range(1, 13):
        out[f"mes_{m:02d}"] = (mes == m).astype(np.int8)

    feature_cols = [c for c in out.columns if c != "date"]
    return out, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Rank cross-seccional (pré-computado em main antes do loop de tickers)
# ─────────────────────────────────────────────────────────────────────────────

def build_cross_sectional_ranks(
    csvs: list[Path],
    janelas: list[int] = CS_RANK_JANELAS,
) -> dict[str, pd.DataFrame]:
    """
    Para cada horizonte h e data t, calcula o percentil do retorno
    ln(P_t / P_{t-h}) do ativo dentro do universo de tickers.
    Retorna {ticker -> DataFrame(cs_rank_h, ...) indexado por Timestamp}.
    """
    closes: dict[str, pd.Series] = {}
    for fp in csvs:
        try:
            tmp = pd.read_csv(fp, parse_dates=["date"], usecols=["date", "close"])
            closes[fp.stem] = tmp.set_index("date")["close"].astype(float)
        except Exception:
            pass
    if len(closes) < 2:
        return {}

    prices = pd.DataFrame(closes).sort_index()

    result: dict[str, pd.DataFrame] = {}
    for ticker in closes:
        rk_df = pd.DataFrame(index=prices.index)
        for h in janelas:
            ret  = np.log(prices / prices.shift(h))
            rank = ret.rank(axis=1, pct=True)
            col  = rank.get(ticker) if hasattr(rank, "get") else rank[ticker]
            rk_df[f"cs_rank_{h}"] = col if col is not None else np.nan
        result[ticker] = rk_df

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Pesos amostrais por decaimento temporal
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(n: int, lam: float = SAMPLE_WEIGHT_LAMBDA) -> np.ndarray:
    """w_t = exp(λ × t/n), normalizado para média 1."""
    t = np.arange(n, dtype=np.float64)
    w = np.exp(lam * t / n)
    return w / w.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers ensemble LightGBM
# ─────────────────────────────────────────────────────────────────────────────

def _make_lgb_params(xgb_params: dict) -> dict:
    """Converte params XGBoost para equivalentes LightGBM."""
    return dict(
        n_estimators     = xgb_params.get("n_estimators", 1000),
        learning_rate    = xgb_params.get("learning_rate", 0.03),
        max_depth        = xgb_params.get("max_depth", 4),
        subsample        = xgb_params.get("subsample", 0.80),
        colsample_bytree = xgb_params.get("colsample_bytree", 0.80),
        min_child_samples= max(5, xgb_params.get("min_child_weight", 10)),
        reg_alpha        = xgb_params.get("reg_alpha", 0.1),
        reg_lambda       = xgb_params.get("reg_lambda", 1.0),
        random_state     = SEED,
        n_jobs           = -1,
        verbose          = -1,
    )


def _fit_predict_ensemble(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    params: dict,
    sw: np.ndarray | None,
) -> tuple[np.ndarray, object]:
    """Treina XGB (+ LGB se ativo) e retorna (predições, modelo_xgb)."""
    model = XGBRegressor(**params)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)],
              sample_weight=sw, verbose=False)
    pred = model.predict(X_te)

    if USE_LGB_ENSEMBLE and HAS_LGB:
        try:
            lgb_model = lgb.LGBMRegressor(**_make_lgb_params(params))
            lgb_model.fit(
                X_fit, y_fit,
                eval_set=[(X_val, y_val)],
                sample_weight=sw,
                callbacks=[
                    lgb.early_stopping(20, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            lgb_pred = lgb_model.predict(
                X_te, num_iteration=lgb_model.best_iteration_
            )
            pred = 0.5 * pred + 0.5 * lgb_pred
        except Exception:
            pass

    return pred, model


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation — Regressor
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward_predict(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_folds: int,
    train_inicial: int,
    embargo: int = 0,
    normalize_target: bool = False,
    use_sample_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Janela expansiva com embargo.
    Fold k: treino = [0 : ini_te − embargo], teste = [ini_te : fim_te].
    Inclui pesos temporais opcionais e ensemble LGB via _fit_predict_ensemble.
    """
    n = len(X)
    if train_inicial >= n - n_folds:
        raise ValueError("Histórico insuficiente para walk-forward.")
    tam = (n - train_inicial) // n_folds
    if tam < 5:
        raise ValueError(f"Folds muito pequenos (tam={tam}). Reduza WF_N_FOLDS.")

    idx_oof  = np.empty(0, dtype=np.int64)
    pred_oof = np.empty(0, dtype=np.float64)
    folds_metrics: list[dict] = []

    for k in range(n_folds):
        ini_te = train_inicial + k * tam
        fim_te = ini_te + tam if k < n_folds - 1 else n
        fim_tr = max(1, ini_te - embargo)

        X_tr, y_tr = X[:fim_tr], y[:fim_tr]
        X_te, y_te = X[ini_te:fim_te], y[ini_te:fim_te]

        if len(X_tr) < 50:
            raise ValueError(
                f"Fold {k+1}: treino muito curto após embargo ({len(X_tr)} obs)."
            )

        cut = int(len(X_tr) * WF_INTERNAL_SPLIT)
        X_fit, X_val = X_tr[:cut], X_tr[cut:]
        y_fit, y_val = y_tr[:cut], y_tr[cut:]

        if normalize_target:
            scaler_y = StandardScaler()
            y_fit_in = scaler_y.fit_transform(y_fit.reshape(-1, 1)).ravel()
            y_val_in = scaler_y.transform(y_val.reshape(-1, 1)).ravel()
        else:
            scaler_y = None
            y_fit_in, y_val_in = y_fit, y_val

        sw = compute_sample_weights(len(y_fit)) if use_sample_weights else None

        pred_raw, model = _fit_predict_ensemble(
            X_fit, y_fit_in, X_val, y_val_in, X_te, params, sw
        )

        pred = scaler_y.inverse_transform(pred_raw.reshape(-1, 1)).ravel() \
               if scaler_y is not None else pred_raw

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
# Walk-Forward Validation — Classificador 3 classes
# ─────────────────────────────────────────────────────────────────────────────

def _label_3class(y_arr: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    """0 = down (<q_lo), 1 = neutral, 2 = up (>q_hi). Tertis calculados no treino."""
    labels = np.ones(len(y_arr), dtype=np.int32)
    labels[y_arr < q_lo] = 0
    labels[y_arr > q_hi] = 2
    return labels


def walk_forward_classify(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    n_folds: int,
    train_inicial: int,
    embargo: int = 0,
    use_sample_weights: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Walk-forward com XGBClassifier 3-classes (down/neutral/up).
    Tertis calculados dinamicamente dentro de cada fold de treino.
    Retorna (idx_oof, expected_return_oof, fold_metrics).
    O retorno esperado = P(up)×μ_up + P(neutral)×μ_neutral + P(down)×μ_down.
    """
    n   = len(X)
    tam = (n - train_inicial) // n_folds

    idx_oof  = np.empty(0, dtype=np.int64)
    exp_oof  = np.empty(0, dtype=np.float64)
    folds_metrics: list[dict] = []

    clf_base = dict(
        objective             = "multi:softprob",
        num_class             = 3,
        n_estimators          = params.get("n_estimators", 800),
        learning_rate         = params.get("learning_rate", 0.03),
        max_depth             = params.get("max_depth", 4),
        subsample             = params.get("subsample", 0.80),
        colsample_bytree      = params.get("colsample_bytree", 0.80),
        min_child_weight      = params.get("min_child_weight", 10),
        reg_alpha             = params.get("reg_alpha", 0.1),
        reg_lambda            = params.get("reg_lambda", 1.0),
        early_stopping_rounds = 20,
        eval_metric           = "mlogloss",
        random_state          = SEED,
        n_jobs                = -1,
        verbosity             = 0,
    )

    for k in range(n_folds):
        ini_te = train_inicial + k * tam
        fim_te = ini_te + tam if k < n_folds - 1 else n
        fim_tr = max(1, ini_te - embargo)

        X_tr, y_tr = X[:fim_tr], y[:fim_tr]
        X_te, y_te = X[ini_te:fim_te], y[ini_te:fim_te]

        if len(X_tr) < 50:
            continue

        q33 = float(np.percentile(y_tr, 33.33))
        q67 = float(np.percentile(y_tr, 66.67))
        y_tr_cls = _label_3class(y_tr, q33, q67)

        cut = int(len(X_tr) * WF_INTERNAL_SPLIT)
        X_fit, X_val = X_tr[:cut], X_tr[cut:]
        y_fit_cls = y_tr_cls[:cut]
        y_val_cls = y_tr_cls[cut:]
        y_fit_raw = y_tr[:cut]

        # Médias de retorno por classe (no treino do fold)
        mu = np.array([
            float(np.mean(y_fit_raw[y_fit_cls == c])) if (y_fit_cls == c).any() else 0.0
            for c in range(3)
        ])

        sw = compute_sample_weights(len(y_fit_cls)) if use_sample_weights else None

        clf = XGBClassifier(**clf_base)
        clf.fit(X_fit, y_fit_cls, eval_set=[(X_val, y_val_cls)],
                sample_weight=sw, verbose=False)

        proba   = clf.predict_proba(X_te)   # (n_te, 3)
        exp_ret = proba @ mu                # retorno esperado

        y_te_cls = _label_3class(y_te, q33, q67)
        dir_acc  = float(np.mean(np.sign(exp_ret) == np.sign(y_te)))

        idx_oof  = np.concatenate([idx_oof,  np.arange(ini_te, fim_te)])
        exp_oof  = np.concatenate([exp_oof,  exp_ret])

        folds_metrics.append({
            "fold":    k + 1,
            "dir_acc": dir_acc,
            "n_teste": len(X_te),
        })

    return idx_oof, exp_oof, folds_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Optuna — busca de hiperparâmetros (opcional)
# ─────────────────────────────────────────────────────────────────────────────

def otimizar_params(
    X_train: np.ndarray, y_train: np.ndarray, n_trials: int
) -> tuple[dict, float]:
    tscv = TimeSeriesSplit(n_splits=OPTUNA_CV_FOLDS)

    def objective(trial: optuna.Trial) -> float:
        p = dict(
            n_estimators          = PARAM_SPACE["n_estimators"],
            learning_rate         = trial.suggest_float("learning_rate", *PARAM_SPACE["learning_rate"], log=True),
            max_depth             = trial.suggest_int  ("max_depth",     *PARAM_SPACE["max_depth"]),
            subsample             = trial.suggest_float("subsample",     *PARAM_SPACE["subsample"]),
            colsample_bytree      = trial.suggest_float("colsample_bytree", *PARAM_SPACE["colsample_bytree"]),
            min_child_weight      = trial.suggest_int  ("min_child_weight", *PARAM_SPACE["min_child_weight"]),
            reg_alpha             = trial.suggest_float("reg_alpha",     *PARAM_SPACE["reg_alpha"]),
            reg_lambda            = trial.suggest_float("reg_lambda",    *PARAM_SPACE["reg_lambda"]),
            eval_metric           = "rmse",
            early_stopping_rounds = 20,
            random_state          = SEED,
            n_jobs                = -1,
            verbosity             = 0,
        )
        rmses = []
        for fold, (tr, va) in enumerate(tscv.split(X_train)):
            X_tr, X_va = X_train[tr], X_train[va]
            y_tr, y_va = y_train[tr], y_train[va]
            sw = compute_sample_weights(len(y_tr)) if USE_SAMPLE_WEIGHTS else None
            model = XGBRegressor(**p)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      sample_weight=sw, verbose=False)
            rmses.append(float(np.sqrt(mean_squared_error(y_va, model.predict(X_va)))))
            trial.report(float(np.mean(rmses)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(rmses)) if rmses else 1.0

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best.update(dict(n_estimators=PARAM_SPACE["n_estimators"],
                     early_stopping_rounds=20, eval_metric="rmse",
                     random_state=SEED, n_jobs=-1, verbosity=0))
    return best, float(study.best_value)


# ─────────────────────────────────────────────────────────────────────────────
# Penalidade de incerteza por baixa acurácia direcional
# ─────────────────────────────────────────────────────────────────────────────

def penalizar_omega(omega_base: float, dir_acc_pct: float) -> tuple[float, float]:
    deficit = max(0.0, DIR_ACC_PENALIDADE_LIMIAR - dir_acc_pct) / 100.0
    fator   = float(np.exp(DIR_ACC_PENALIDADE_K * deficit))
    return omega_base * fator, fator


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline por ticker
# ─────────────────────────────────────────────────────────────────────────────

def treinar(
    ticker: str,
    df: pd.DataFrame,
    horizon: int = 1,
    cs_rank_df: pd.DataFrame | None = None,
) -> dict:
    """
    Treina regressor XGBoost (e opcionalmente classificador/LGB) para horizonte h.
    cs_rank_df: DataFrame indexado por Timestamp com colunas cs_rank_<h>.
    """
    nome_h  = HORIZON_NOMES.get(horizon, f"{horizon}d")
    prefixo = f"{ticker}_h{horizon}d"

    # Injeta features cross-sectionais alinhadas por data
    if cs_rank_df is not None and not cs_rank_df.empty:
        df = df.copy()
        dates_ts = pd.to_datetime(df["date"])
        for col in [c for c in cs_rank_df.columns if c.startswith("cs_rank_")]:
            aligned = cs_rank_df[col].reindex(dates_ts.values).values
            if not np.all(np.isnan(aligned.astype(float))):
                df[col] = aligned

    feat_df, feature_cols = build_features(df)

    close = df["close"].astype(float)
    feat_df["__y__"] = np.log(close.shift(-horizon) / close).values

    valid_X = feat_df[feature_cols].notna().all(axis=1)
    feat_df = feat_df[valid_X].reset_index(drop=True)

    pred_row = feat_df[feat_df["__y__"].isna()].tail(1).copy()
    sup      = feat_df[feat_df["__y__"].notna()].reset_index(drop=True)

    X      = sup[feature_cols].to_numpy(dtype=np.float32)
    y      = sup["__y__"].to_numpy(dtype=np.float64)
    dates  = pd.to_datetime(sup["date"]).to_numpy()
    closes = df.loc[df["date"].isin(sup["date"]), "close"].to_numpy(dtype=np.float32)

    n = len(X)
    if n < 250:
        raise ValueError(f"{ticker} h={horizon}d: histórico curto após limpeza (n={n}).")

    train_inicial = max(int(n * WF_TRAIN_INICIAL), 200)

    # ── Optuna ──────────────────────────────────────────────────────────────
    if OPTUNA_TRIALS > 0:
        print(f"      Optuna {OPTUNA_TRIALS} trials × {OPTUNA_CV_FOLDS} folds ...",
              end=" ", flush=True)
        params, cv_rmse = otimizar_params(X[:train_inicial], y[:train_inicial], OPTUNA_TRIALS)
        print(f"RMSE-CV={cv_rmse:.5f} | lr={params['learning_rate']:.4f} | "
              f"depth={params['max_depth']}", flush=True)
    else:
        params  = dict(XGB_DEFAULT)
        cv_rmse = float("nan")

    # Modo quantil: troca o objetivo do XGBoost por mediana robusta
    if USE_QUANTILE:
        params = {k: v for k, v in params.items()
                  if k not in {"eval_metric"}}
        params["objective"]   = "reg:quantileerror"
        params["quantile_alpha"] = 0.5
        params["eval_metric"] = "quantile"

    # ── Walk-Forward Regressor ───────────────────────────────────────────────
    embargo_dias = int(horizon * WF_EMBARGO_FATOR)
    idx_oof, pred_oof, folds_metrics = walk_forward_predict(
        X, y, params,
        n_folds          = WF_N_FOLDS,
        train_inicial    = train_inicial,
        embargo          = embargo_dias,
        normalize_target = NORMALIZAR_TARGET,
        use_sample_weights = USE_SAMPLE_WEIGHTS,
    )
    y_oof = y[idx_oof]

    rmse_wf     = float(np.sqrt(mean_squared_error(y_oof, pred_oof)))
    mae_wf      = float(mean_absolute_error(y_oof, pred_oof))
    r2_wf       = float(r2_score(y_oof, pred_oof))
    mse_wf      = float(mean_squared_error(y_oof, pred_oof))
    dir_acc_reg = float(np.mean(np.sign(pred_oof) == np.sign(y_oof)) * 100)

    # ── Walk-Forward Classificador (opcional) ───────────────────────────────
    dir_acc_clf: float | None = None
    q_source = "regressor"
    final_pred_oof = pred_oof
    final_idx_oof  = idx_oof
    final_dir_acc  = dir_acc_reg

    if USE_CLASSIFICATION:
        try:
            idx_clf, exp_clf, fm_clf = walk_forward_classify(
                X, y, params,
                n_folds          = WF_N_FOLDS,
                train_inicial    = train_inicial,
                embargo          = embargo_dias,
                use_sample_weights = USE_SAMPLE_WEIGHTS,
            )
            y_clf      = y[idx_clf]
            dir_acc_clf = float(np.mean(np.sign(exp_clf) == np.sign(y_clf)) * 100)
            if dir_acc_clf > dir_acc_reg:
                final_pred_oof = exp_clf
                final_idx_oof  = idx_clf
                final_dir_acc  = dir_acc_clf
                q_source       = "classificador"
                print(f"      [CLF] dir_acc {dir_acc_clf:.1f}% > reg {dir_acc_reg:.1f}% → usando classificador")
        except Exception as exc:
            print(f"      [CLF] erro no walk-forward: {exc}")

    y_final_oof = y[final_idx_oof]

    # ── Modelo final em todo o histórico supervisionado ──────────────────────
    cut = int(n * WF_INTERNAL_SPLIT)

    if NORMALIZAR_TARGET and not USE_QUANTILE:
        scaler_y_final = StandardScaler()
        y_fit_final = scaler_y_final.fit_transform(y[:cut].reshape(-1, 1)).ravel()
        y_val_final  = scaler_y_final.transform(y[cut:].reshape(-1, 1)).ravel()
    else:
        scaler_y_final = None
        y_fit_final, y_val_final = y[:cut], y[cut:]

    sw_final = compute_sample_weights(cut) if USE_SAMPLE_WEIGHTS else None

    # Determina o ponto de predição Q antes de treinar o modelo final
    if not pred_row.empty:
        X_q         = pred_row[feature_cols].to_numpy(dtype=np.float32)
        q_data_base = pd.to_datetime(pred_row["date"].iloc[0]).date().isoformat()
    else:
        X_q         = X[-1:]
        q_data_base = pd.to_datetime(sup["date"].iloc[-1]).date().isoformat()

    # Treina modelo final e prediz Q em uma única chamada (evita retreino)
    q_pred_arr, final_model = _fit_predict_ensemble(
        X[:cut], y_fit_final, X[cut:], y_val_final, X_q, params, sw_final
    )
    n_rounds_final = int(getattr(final_model, "best_iteration", -1) + 1)
    q_raw = float(q_pred_arr[0])

    q_log     = float(scaler_y_final.inverse_transform([[q_raw]])[0][0]) \
                if scaler_y_final is not None else q_raw
    q_simples = float(np.expm1(q_log))

    # Se classificador venceu: recalcula Q usando retorno esperado do classificador final
    if q_source == "classificador":
        try:
            q33_all = float(np.percentile(y, 33.33))
            q67_all = float(np.percentile(y, 66.67))
            mu_all  = np.array([
                float(np.mean(y[y < q33_all])),
                float(np.mean(y[(y >= q33_all) & (y <= q67_all)])),
                float(np.mean(y[y > q67_all])),
            ])
            y_cls_fit = _label_3class(y[:cut], q33_all, q67_all)
            y_cls_val = _label_3class(y[cut:],  q33_all, q67_all)
            clf_base = dict(
                objective="multi:softprob", num_class=3,
                n_estimators=params.get("n_estimators", 800),
                learning_rate=params.get("learning_rate", 0.03),
                max_depth=params.get("max_depth", 4),
                subsample=params.get("subsample", 0.80),
                colsample_bytree=params.get("colsample_bytree", 0.80),
                min_child_weight=params.get("min_child_weight", 10),
                reg_alpha=params.get("reg_alpha", 0.1),
                reg_lambda=params.get("reg_lambda", 1.0),
                early_stopping_rounds=20, eval_metric="mlogloss",
                random_state=SEED, n_jobs=-1, verbosity=0,
            )
            clf_final = XGBClassifier(**clf_base)
            clf_final.fit(X[:cut], y_cls_fit,
                          eval_set=[(X[cut:], y_cls_val)],
                          sample_weight=sw_final, verbose=False)
            proba_q   = clf_final.predict_proba(X_q)[0]
            q_log     = float(proba_q @ mu_all)
            q_simples = float(np.expm1(q_log))
            joblib.dump(clf_final, OUT_DIR / f"{prefixo}_clf.pkl")
        except Exception as exc:
            print(f"      [CLF] falha no Q final: {exc}")
            q_source = "regressor"

    # ── SHAP ────────────────────────────────────────────────────────────────
    if USE_SHAP and HAS_SHAP:
        try:
            explainer = _shap.TreeExplainer(final_model)
            shap_vals = explainer.shap_values(X)
            shap_mean = np.abs(shap_vals).mean(axis=0)
            shap_df   = pd.DataFrame({"feature": feature_cols,
                                      "shap_mean_abs": shap_mean})
            shap_df   = shap_df.sort_values("shap_mean_abs", ascending=False)
            shap_df.to_csv(OUT_DIR / f"{prefixo}_shap.csv", index=False)
        except Exception as exc:
            print(f"      [SHAP] falha: {exc}")

    # ── Salva predições walk-forward ─────────────────────────────────────────
    pred_df = pd.DataFrame({
        "date":   dates[final_idx_oof],
        "ticker": ticker,
        "close":  closes[final_idx_oof],
        "y_real": y_final_oof,
        "y_pred": final_pred_oof,
        "erro":   y_final_oof - final_pred_oof,
        "fold":   np.concatenate([
            np.full(m["n_teste"], m["fold"], dtype=np.int32) for m in folds_metrics
        ]),
    }).sort_values("date").reset_index(drop=True)
    pred_df.to_csv(OUT_DIR / f"{prefixo}_predicoes.csv", index=False)

    # ── Salva modelo e importância ────────────────────────────────────────────
    joblib.dump(final_model, OUT_DIR / f"{prefixo}_model.pkl")
    if scaler_y_final is not None:
        joblib.dump(scaler_y_final, OUT_DIR / f"{prefixo}_scaler_y.pkl")

    importances = pd.Series(final_model.feature_importances_, index=feature_cols)
    imp_df = (
        importances.reset_index()
        .rename(columns={"index": "feature", 0: "importance"})
        .sort_values("importance", ascending=False)
    )
    imp_df.columns = ["feature", "importance"]
    imp_df.to_csv(OUT_DIR / f"{prefixo}_importancia.csv", index=False)

    # ── Plot 4 painéis ────────────────────────────────────────────────────────
    _plot_ticker(ticker, nome_h, horizon, prefixo, pred_df, imp_df,
                 rmse_wf, mae_wf, r2_wf, final_dir_acc, q_log, q_simples, q_source)

    return {
        "ticker":          ticker,
        "horizon":         horizon,
        "horizon_nome":    nome_h,
        "n_total":         n,
        "n_oof":           int(len(y_final_oof)),
        "rmse_wf":         rmse_wf,
        "mae_wf":          mae_wf,
        "r2_wf":           r2_wf,
        "mse_wf":          mse_wf,
        "dir_acc_pct":     final_dir_acc,
        "dir_acc_reg_pct": dir_acc_reg,
        "dir_acc_clf_pct": dir_acc_clf,
        "q_source":        q_source,
        "rmse_cv_optuna":  round(cv_rmse, 6) if not np.isnan(cv_rmse) else None,
        "n_rounds_final":  n_rounds_final,
        "q_log":           q_log,
        "q_simples":       q_simples,
        "q_data_base":     q_data_base,
        "omega_base":      mse_wf,
        "lr":              round(params.get("learning_rate", float("nan")), 4),
        "max_depth":       params.get("max_depth", "—"),
    }


def _plot_ticker(
    ticker: str,
    nome_h: str,
    horizon: int,
    prefixo: str,
    pred_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    rmse_wf: float,
    mae_wf: float,
    r2_wf: float,
    dir_acc: float,
    q_log: float,
    q_simples: float,
    q_source: str,
) -> None:
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"XGBoost — {ticker}  | horizonte: {nome_h}  (Walk-Forward)\n"
        f"RMSE: {rmse_wf:.5f}  |  MAE: {mae_wf:.5f}  |  R²: {r2_wf:.4f}  |  "
        f"Direção: {dir_acc:.1f}%  |  Q = {q_log:+.4%} (log)  ≈ {q_simples:+.4%}  [{q_source}]",
        fontsize=11,
    )
    gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, :])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])

    ax0.plot(pred_df["date"], pred_df["y_real"],  color="black",      lw=0.8, label="real")
    ax0.plot(pred_df["date"], pred_df["y_pred"],  color="tab:orange", lw=0.8, label="previsto")
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

    lim = float(np.nanmax(np.abs(np.concatenate(
        [pred_df["y_real"].values, pred_df["y_pred"].values]
    )))) * 1.05
    ax2.scatter(pred_df["y_real"], pred_df["y_pred"], s=8, alpha=0.45, color="tab:purple")
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


def _plot_comparacao_horizontes(
    ticker: str, resultados: list[dict], h_vencedor: int
) -> None:
    nomes = [r["horizon_nome"] for r in resultados]
    rmses = [r["rmse_wf"]     for r in resultados]
    maes  = [r["mae_wf"]      for r in resultados]
    dirs  = [r["dir_acc_pct"] for r in resultados]
    cores = ["tab:green" if r["horizon"] == h_vencedor else "tab:blue"
             for r in resultados]

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


def comparar_horizontes(
    ticker: str,
    df: pd.DataFrame,
    cs_rank_df: pd.DataFrame | None = None,
) -> tuple[list[dict], dict]:
    """
    Executa treinar() para cada horizonte em HORIZONS (Direct Forecasting).
    Elege o vencedor pelo HORIZON_CRITERIO e copia seus artefatos como padrão.
    """
    resultados: list[dict] = []
    for h in HORIZONS:
        nome_h = HORIZON_NOMES.get(h, f"{h}d")
        print(f"    horizonte {nome_h} ...", end=" ", flush=True)
        try:
            m = treinar(ticker, df, horizon=h, cs_rank_df=cs_rank_df)
            resultados.append(m)
            print(
                f"RMSE={m['rmse_wf']:.5f}  DirAcc={m['dir_acc_pct']:.1f}%"
                f"  [{m['q_source']}]",
                flush=True,
            )
        except Exception as exc:
            print(f"ERRO: {exc}", flush=True)

    if not resultados:
        raise RuntimeError(f"{ticker}: falhou em todos os horizontes.")

    if HORIZON_CRITERIO in HORIZON_MENOR_MELHOR:
        vencedor = min(resultados, key=lambda m: m[HORIZON_CRITERIO])
    else:
        vencedor = max(resultados, key=lambda m: m[HORIZON_CRITERIO])
    h_win = vencedor["horizon"]

    omega_aj, fator = penalizar_omega(vencedor["omega_base"], vencedor["dir_acc_pct"])
    vencedor["omega_ajustado"]   = omega_aj
    vencedor["penalidade_fator"] = round(fator, 4)

    for sufixo in ("_predicoes.csv", "_model.pkl", "_importancia.csv",
                   "_plot.png", "_shap.csv", "_clf.pkl", "_scaler_y.pkl"):
        src = OUT_DIR / f"{ticker}_h{h_win}d{sufixo}"
        dst = OUT_DIR / f"{ticker}{sufixo}"
        if src.exists():
            shutil.copy2(src, dst)

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
    flags = []
    if USE_SAMPLE_WEIGHTS: flags.append(f"sw_λ={SAMPLE_WEIGHT_LAMBDA}")
    if USE_CLASSIFICATION:  flags.append("clf3")
    if USE_QUANTILE:        flags.append("quantile")
    if USE_LGB_ENSEMBLE and HAS_LGB: flags.append("lgb_ens")
    if USE_SHAP and HAS_SHAP:        flags.append("shap")
    flags_str = " | ".join(flags) if flags else "—"

    print(
        f"Regressor XGBoost para vetor Q | tickers: {len(csvs)} | "
        f"horizontes: {horizontes_str} | critério: {HORIZON_CRITERIO}\n"
        f"WF: {WF_N_FOLDS} folds (split_interno={WF_INTERNAL_SPLIT:.0%}), "
        f"treino inicial = {WF_TRAIN_INICIAL:.0%} | Optuna: {OPTUNA_TRIALS} trials\n"
        f"Extras: {flags_str}\n"
    )

    # Pré-computa ranks cross-sectionais para todos os tickers
    print("  Computando ranks cross-sectionais ...", end=" ", flush=True)
    cs_ranks = build_cross_sectional_ranks(csvs, janelas=CS_RANK_JANELAS)
    if cs_ranks:
        print(f"OK ({len(cs_ranks)} tickers, janelas={CS_RANK_JANELAS})")
    else:
        print("ignorado (menos de 2 tickers)")

    all_h_metrics:  list[dict] = []
    winner_metrics: list[dict] = []
    erros: list[tuple[str, str]] = []

    for fp in csvs:
        ticker = fp.stem
        print(f"[{ticker}] {len(pd.read_csv(fp, usecols=['date']))} dias úteis", flush=True)
        try:
            df  = carregar_df(fp)
            resultados, vencedor = comparar_horizontes(
                ticker, df, cs_rank_df=cs_ranks.get(ticker)
            )
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

    # ── consolida métricas completas ──────────────────────────────────────
    all_h_df = (
        pd.DataFrame(all_h_metrics)
        .sort_values(["ticker", "horizon"])
        .reset_index(drop=True)
    )
    all_h_df.to_csv(OUT_DIR / "metricas_todos_horizontes.csv", index=False)

    winner_df = (
        pd.DataFrame(winner_metrics)
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    arq_anterior = OUT_DIR / "metricas_vencedores.csv"
    snapshot_anterior: pd.DataFrame | None = None
    if arq_anterior.exists():
        snapshot_anterior = pd.read_csv(arq_anterior)
        shutil.copy2(arq_anterior, OUT_DIR / "metricas_vencedores_anterior.csv")

    winner_df.to_csv(arq_anterior, index=False)

    # ── vetor Q ────────────────────────────────────────────────────────────
    q_df = (
        winner_df[["ticker", "horizon_nome", "q_data_base", "q_log",
                   "q_simples", "q_source"]]
        .rename(columns={"horizon_nome": "horizonte_vencedor"})
    )
    q_df.to_csv(OUT_DIR / "q_vetor.csv", index=False)

    # ── omega_base ────────────────────────────────────────────────────────
    omega_df = (
        winner_df[[
            "ticker", "horizon_nome", "dir_acc_pct",
            "mse_wf", "omega_ajustado", "penalidade_fator",
            "rmse_wf", "mae_wf",
        ]]
        .rename(columns={"mse_wf": "omega_base", "horizon_nome": "horizonte_vencedor"})
    )
    omega_df.to_csv(OUT_DIR / "omega_base.csv", index=False)

    # ── resumo no terminal ────────────────────────────────────────────────
    print("\n── Vencedores por ticker (para o vetor Q) ─────────────────────")
    cols_show = ["ticker", "horizon_nome", "dir_acc_pct", "rmse_wf",
                 "r2_wf", "q_simples", "omega_base", "omega_ajustado",
                 "penalidade_fator", "q_source"]
    cols_show = [c for c in cols_show if c in winner_df.columns]
    print(winner_df[cols_show].round(6).to_string(index=False))

    print("\n── Comparação entre horizontes (todos os tickers) ─────────────")
    pivot = all_h_df.pivot_table(
        index="ticker", columns="horizon_nome",
        values="dir_acc_pct", aggfunc="first",
    )
    print(pivot.round(1).to_string())

    print(f"\nVetor Q    -> {OUT_DIR / 'q_vetor.csv'}")
    print(f"omega_base -> {OUT_DIR / 'omega_base.csv'}")
    print(f"Todos horizontes -> {OUT_DIR / 'metricas_todos_horizontes.csv'}")

    if snapshot_anterior is not None and "dir_acc_pct" in snapshot_anterior.columns:
        _comparar_rodadas(snapshot_anterior, winner_df)

    _plot_consolidado_horizontes(all_h_df)

    # ── plot consolidado de feature importance ───────────────────────────
    imp_csvs = [
        OUT_DIR / f"{m['ticker']}_importancia.csv"
        for m in winner_metrics
        if (OUT_DIR / f"{m['ticker']}_importancia.csv").exists()
    ]
    if imp_csvs:
        imp_all = pd.concat([pd.read_csv(f) for f in imp_csvs], ignore_index=True)
        imp_top = (
            imp_all.groupby("feature")["importance"].mean()
            .sort_values(ascending=False)
            .head(20)
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        imp_top.iloc[::-1].plot(kind="barh", ax=ax, color="seagreen")
        ax.set_title(
            f"Top-20 features — modelos vencedores ({len(imp_csvs)} tickers)\n"
            f"critério: {HORIZON_CRITERIO}"
        )
        ax.set_xlabel("Importância média (gain)")
        ax.grid(True, alpha=0.25, axis="x")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "importancia_consolidada.png", dpi=140, bbox_inches="tight")
        plt.close()

    # ── plot consolidado SHAP (se disponível) ────────────────────────────
    if USE_SHAP and HAS_SHAP:
        shap_csvs = [
            OUT_DIR / f"{m['ticker']}_shap.csv"
            for m in winner_metrics
            if (OUT_DIR / f"{m['ticker']}_shap.csv").exists()
        ]
        if shap_csvs:
            shap_all = pd.concat([pd.read_csv(f) for f in shap_csvs], ignore_index=True)
            shap_top = (
                shap_all.groupby("feature")["shap_mean_abs"].mean()
                .sort_values(ascending=False)
                .head(20)
            )
            fig, ax = plt.subplots(figsize=(10, 7))
            shap_top.iloc[::-1].plot(kind="barh", ax=ax, color="darkorange")
            ax.set_title(
                f"Top-20 features por |SHAP| médio — modelos vencedores ({len(shap_csvs)} tickers)"
            )
            ax.set_xlabel("|SHAP| médio")
            ax.grid(True, alpha=0.25, axis="x")
            plt.tight_layout()
            plt.savefig(OUT_DIR / "shap_consolidado.png", dpi=140, bbox_inches="tight")
            plt.close()
            print(f"SHAP consolidado -> {OUT_DIR / 'shap_consolidado.png'}")

    if erros:
        print("\nErros:")
        for t, msg in erros:
            print(f"  {t}: {msg}")


def _comparar_rodadas(anterior: pd.DataFrame, atual: pd.DataFrame) -> None:
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
        f"melhoraram: {ganhos}  |  pioraram: {perdas}  |  "
        f"iguais: {len(comp)-ganhos-perdas}"
    )
    print(f"  Arquivo: {OUT_DIR / 'comparacao_rodadas.csv'}")

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
    horizons_presentes = sorted(all_h_df["horizon"].unique())
    nomes    = [HORIZON_NOMES.get(h, f"{h}d") for h in horizons_presentes]
    dir_med  = [all_h_df[all_h_df["horizon"] == h]["dir_acc_pct"].mean()
                for h in horizons_presentes]
    rmse_med = [all_h_df[all_h_df["horizon"] == h]["rmse_wf"].mean()
                for h in horizons_presentes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Comparação de Horizontes — Consolidado ({all_h_df['ticker'].nunique()} tickers)\n"
        f"critério de seleção por ticker: {HORIZON_CRITERIO}",
        fontsize=12,
    )

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    cores   = palette[: len(horizons_presentes)]

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
