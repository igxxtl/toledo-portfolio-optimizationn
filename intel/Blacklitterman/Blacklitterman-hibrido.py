"""
Pipeline Black-Litterman hibrido (prior de mercado + views de IA/sentimento).

Integra em um unico script as ideias de:
- PI.py / PI_incerteza.py  -> prior (PI) e incerteza do prior (tau * Sigma)
- Q.py                     -> views diarias Q via previsao + sentimento
- omega.py                 -> incerteza das views (Omega) dinamica por dia

Saidas principais:
- bl_hibrido_q_long.csv
- bl_hibrido_omega_long.csv
- bl_hibrido_q_omega_long.csv
- bl_hibrido_pi.csv
- bl_hibrido_sigma.csv
- bl_hibrido_prior_cov_tau_sigma.csv
- bl_hibrido_posterior_mu.csv
- bl_hibrido_posterior_weights.csv
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.model.predictor import Predictor

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BL_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]

NEWS_PATH = REPO_ROOT / "intel" / "data acquisition" / "noticias_b3_sem_duplicados_sentimento.json"
TICKERS_DIR = REPO_ROOT / "tickers_data"
MODEL_DIR = REPO_ROOT / "modelo_7h_vwap"
LAG_LLAMA_DIR = REPO_ROOT / "lag-Llama"

# -----------------------------------------------------------------------------
# Config Q (views)
# -----------------------------------------------------------------------------
SENTIMENT_MAP = {"positivo": 1.0, "negativo": -1.0, "neutro": 0.0}
PREDICTION_LENGTH = 7
PERIOD_START = "202501"
PERIOD_END = "202512"
ALPHA_Q = 0.50
B3_SESSION_HOURS = set(range(13, 20))

# -----------------------------------------------------------------------------
# Config prior PI
# -----------------------------------------------------------------------------
ATIVOS = ["VALE3.SA", "BBAS3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA"]
PRIOR_START_DATE = "2023-01-01"
PRIOR_END_DATE = "2025-12-31"
RISK_FREE_ANNUAL = 0.15
TAU = 0.025

# -----------------------------------------------------------------------------
# Config Omega (dinamico)
# -----------------------------------------------------------------------------
ERR_WINDOW = 14
ERR_MIN_PERIODS = 5
EPS = 1e-8
CONFIDENCE_FLOOR = 0.05
CONFIDENCE_CAP = 0.95
W_MODEL_CONF = 0.7
W_NEWS_CONF = 0.3

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
OUT_Q_LONG = BL_DIR / "bl_hibrido_q_long.csv"
OUT_OMEGA_LONG = BL_DIR / "bl_hibrido_omega_long.csv"
OUT_Q_OMEGA_LONG = BL_DIR / "bl_hibrido_q_omega_long.csv"
OUT_PI = BL_DIR / "bl_hibrido_pi.csv"
OUT_SIGMA = BL_DIR / "bl_hibrido_sigma.csv"
OUT_PRIOR_COV = BL_DIR / "bl_hibrido_prior_cov_tau_sigma.csv"
OUT_POSTERIOR_MU = BL_DIR / "bl_hibrido_posterior_mu.csv"
OUT_POSTERIOR_W = BL_DIR / "bl_hibrido_posterior_weights.csv"
OUT_POSTERIOR_W_CONTROLLED = BL_DIR / "bl_hibrido_posterior_weights_controlled.csv"
OUT_POSTERIOR_W_CTRL_DAILY = BL_DIR / "bl_hibrido_posterior_weights_controlled_daily.csv"
OUT_POSTERIOR_W_CTRL_WEEKLY = BL_DIR / "bl_hibrido_posterior_weights_controlled_weekly.csv"
OUT_POSTERIOR_W_CTRL_MONTHLY = BL_DIR / "bl_hibrido_posterior_weights_controlled_monthly.csv"

# -----------------------------------------------------------------------------
# Config de risco para pesos (portfolio controlado)
# -----------------------------------------------------------------------------
LONG_ONLY = True
MAX_WEIGHT_PER_ASSET = 0.35

# -----------------------------------------------------------------------------
# Config de prior sem vazamento temporal
# -----------------------------------------------------------------------------
USE_ROLLING_PRIOR = True
ROLLING_MIN_OBS = 60


# -----------------------------------------------------------------------------
# Q helpers
# -----------------------------------------------------------------------------
def parse_news_datetime(data_str: str) -> datetime | None:
    if not data_str or not data_str.strip():
        return None
    try:
        return datetime.strptime(data_str.replace(" GMT", "").strip(), "%a, %d %b %Y %H:%M:%S")
    except ValueError:
        return None


def ticker_to_sa(ticker: str) -> str:
    return ticker if ticker.endswith(".SA") else f"{ticker}.SA"


def ticker_from_sa(ticker: str) -> str:
    return ticker[:-3] if ticker.endswith(".SA") else ticker


def ticker_company_key(ticker: str) -> str:
    """
    Normaliza ticker para chave da empresa (ex.: VALE3/VALE4 -> VALE).
    """
    clean = ticker.upper().replace(".SA", "")
    letters = "".join(re.findall(r"[A-Z]", clean))
    return letters[:4] if len(letters) >= 4 else letters


def month_bounds(period_start: str, period_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=int(period_start[:4]), month=int(period_start[4:]), day=1)
    end_month = pd.Timestamp(year=int(period_end[:4]), month=int(period_end[4:]), day=1)
    end = end_month + pd.offsets.MonthEnd(1)
    return start.normalize(), end.normalize()


def load_daily_sentiment_for_ticker(path: Path, ticker: str) -> pd.DataFrame:
    rows = json.loads(path.read_text(encoding="utf-8"))
    parsed: list[dict[str, object]] = []
    target_key = ticker_company_key(ticker)

    for item in rows:
        item_ticker = (item.get("ticker") or "").strip().upper()
        if ticker_company_key(item_ticker) != target_key:
            continue

        dt = parse_news_datetime(item.get("data") or "")
        if dt is None:
            janela_inicio = item.get("janela_inicio") or ""
            try:
                dt = datetime.strptime(janela_inicio, "%Y-%m-%d")
            except ValueError:
                continue

        sentimento = (item.get("sentimento") or "neutro").strip().lower()
        parsed.append(
            {
                "ticker": ticker,
                "view_date": pd.Timestamp(dt.date()),
                "sentimento_score": SENTIMENT_MAP.get(sentimento, 0.0),
            }
        )

    if not parsed:
        return pd.DataFrame(
            columns=["ticker", "view_date", "sentiment_signal", "news_count"]
        )

    df = pd.DataFrame(parsed)
    grouped = (
        df.groupby(["ticker", "view_date"], as_index=False)
        .agg(sentiment_raw=("sentimento_score", "mean"), news_count=("sentimento_score", "size"))
    )
    grouped["sentiment_signal"] = np.tanh(grouped["sentiment_raw"] * np.log1p(grouped["news_count"]))
    return grouped[["ticker", "view_date", "sentiment_signal", "news_count"]]


def load_hourly_series_for_ticker(tickers_dir: Path, ticker: str) -> pd.DataFrame:
    file_path = tickers_dir / f"BMFBOVESPA_DLY_{ticker}, 60.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"CSV do ticker {ticker} nao encontrado em: {file_path}")

    df = pd.read_csv(file_path, usecols=["time", "close", "Volume"])
    df["ticker"] = ticker
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("float32")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("float32")
    return df.dropna(subset=["time", "close", "Volume"]).sort_values("time").reset_index(drop=True)


def filter_b3_session(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["weekday"] = out["time"].dt.weekday
    out["hour"] = out["time"].dt.hour
    out = out[(out["weekday"] <= 4) & (out["hour"].isin(B3_SESSION_HOURS))]
    out = out.sort_values("time").reset_index(drop=True)
    return out.drop(columns=["weekday", "hour"])


def add_vwap_target(df: pd.DataFrame, window_vwap: int = 7) -> pd.DataFrame:
    out = df.copy()
    out["pv"] = out["close"] * out["Volume"]
    out["vwap"] = out["pv"].rolling(window=window_vwap).sum() / out["Volume"].rolling(window=window_vwap).sum()
    out["vwap"] = out["vwap"].fillna(out["close"])
    out["log_ret_vwap"] = np.log(out["vwap"] / out["vwap"].shift(1))
    out = out.dropna(subset=["log_ret_vwap"]).copy()
    # Mantem o dtype coerente com o modelo treinado (evita Double vs Float no backend torch).
    out["pv"] = out["pv"].astype("float32")
    out["vwap"] = out["vwap"].astype("float32")
    out["log_ret_vwap"] = out["log_ret_vwap"].astype("float32")
    out["time_fake"] = pd.date_range(start=out["time"].min(), periods=len(out), freq="h")
    out["view_date"] = out["time"].dt.normalize()
    return out


def load_predictor(model_dir: Path) -> Predictor:
    if LAG_LLAMA_DIR.exists():
        sys.path.insert(0, str(LAG_LLAMA_DIR))
    return Predictor.deserialize(model_dir)


def walk_forward_predict_7h(
    df: pd.DataFrame,
    predictor: Predictor,
    ticker: str,
    period_start: str,
    period_end: str,
) -> pd.DataFrame:
    start_date, end_date = month_bounds(period_start, period_end)
    daily_groups = df.groupby("view_date", sort=True)
    all_days = [d for d in sorted(daily_groups.groups.keys()) if start_date <= d <= end_date]

    rows: list[dict[str, object]] = []
    for day in all_days:
        day_df = daily_groups.get_group(day).sort_values("time")
        if len(day_df) != PREDICTION_LENGTH:
            continue
        if set(day_df["time"].dt.hour.tolist()) != B3_SESSION_HOURS:
            continue

        first_idx = int(day_df.index.min())
        context_df = df.iloc[:first_idx].copy()
        if len(context_df) < 60:
            continue
        context_df["log_ret_vwap"] = context_df["log_ret_vwap"].astype("float32")

        dataset_loop = PandasDataset.from_long_dataframe(
            context_df[["ticker", "time_fake", "log_ret_vwap"]],
            item_id="ticker",
            timestamp="time_fake",
            target="log_ret_vwap",
            freq="h",
        )
        forecast_it, _ = make_evaluation_predictions(dataset=dataset_loop, predictor=predictor, num_samples=50)
        forecast = list(forecast_it)[0]

        rows.append(
            {
                "view_date": day,
                "ticker": ticker,
                "ret_7h_pred": float(np.sum(forecast.mean[:PREDICTION_LENGTH])),
                "ret_7h_real": float(day_df["log_ret_vwap"].sum()),
            }
        )

    if not rows:
        raise RuntimeError("Nenhuma view Q foi gerada no periodo informado.")

    out = pd.DataFrame(rows).sort_values("view_date").reset_index(drop=True)
    out["ret_scale"] = out["ret_7h_real"].abs().rolling(14, min_periods=5).mean().shift(1)
    fallback = float(out["ret_7h_real"].abs().median())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 0.005
    out["ret_scale"] = out["ret_scale"].fillna(fallback)
    return out


def combine_ai_and_sentiment(views: pd.DataFrame, sentiment: pd.DataFrame, alpha: float) -> pd.DataFrame:
    merged = views.merge(sentiment, on=["ticker", "view_date"], how="left")
    merged["sentiment_signal"] = merged["sentiment_signal"].fillna(0.0)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)
    merged["sentiment_component"] = merged["sentiment_signal"] * merged["ret_scale"]
    merged["Q"] = alpha * merged["ret_7h_pred"] + (1.0 - alpha) * merged["sentiment_component"]
    return merged[
        [
            "view_date",
            "ticker",
            "Q",
            "ret_7h_pred",
            "ret_7h_real",
            "sentiment_signal",
            "sentiment_component",
            "news_count",
            "ret_scale",
        ]
    ].sort_values(["view_date", "ticker"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Omega helpers
# -----------------------------------------------------------------------------
def _safe_sample_var(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return np.nan
    var_value = float(clean.var(ddof=1))
    if not np.isfinite(var_value) or var_value <= 0:
        return np.nan
    return var_value


def _compute_confidence_per_ticker(group: pd.DataFrame) -> pd.DataFrame:
    out = group.sort_values("view_date").copy()
    out["pred_abs_error"] = (out["ret_7h_pred"] - out["ret_7h_real"]).abs()

    out["error_scale"] = out["pred_abs_error"].rolling(ERR_WINDOW, min_periods=ERR_MIN_PERIODS).median().shift(1)
    fallback = float(out["pred_abs_error"].median())
    if not np.isfinite(fallback) or fallback <= 0:
        fallback = 0.005
    out["error_scale"] = out["error_scale"].fillna(fallback).clip(lower=EPS)

    out["model_confidence"] = np.exp(-out["pred_abs_error"] / out["error_scale"])
    out["news_confidence"] = 1.0 - np.exp(-out["news_count"] / 3.0)
    out["view_confidence"] = (
        W_MODEL_CONF * out["model_confidence"] + W_NEWS_CONF * out["news_confidence"]
    ).clip(lower=CONFIDENCE_FLOOR, upper=CONFIDENCE_CAP)
    return out


def calculate_omega(q_long: pd.DataFrame, tau: float) -> pd.DataFrame:
    sigma_by_ticker: dict[str, float] = {}
    for ticker, group in q_long.groupby("ticker", sort=True):
        sigma_ii = _safe_sample_var(group["ret_7h_real"])
        if not np.isfinite(sigma_ii):
            sigma_ii = _safe_sample_var(group["ret_7h_pred"])
        if not np.isfinite(sigma_ii):
            sigma_ii = 2.5e-5
        sigma_by_ticker[ticker] = sigma_ii

    parts = [_compute_confidence_per_ticker(g) for _, g in q_long.groupby("ticker", sort=True)]
    with_conf = pd.concat(parts, axis=0, ignore_index=True)

    with_conf["sigma_ii"] = with_conf["ticker"].map(sigma_by_ticker)
    with_conf["omega_base"] = tau * with_conf["sigma_ii"]
    with_conf["omega"] = with_conf["omega_base"] * (1.0 - with_conf["view_confidence"]) / with_conf["view_confidence"]
    with_conf["omega"] = with_conf["omega"].clip(lower=EPS)
    with_conf["tau"] = tau

    return with_conf[
        [
            "view_date",
            "ticker",
            "omega",
            "omega_base",
            "view_confidence",
            "model_confidence",
            "news_confidence",
            "pred_abs_error",
            "error_scale",
            "sigma_ii",
            "tau",
        ]
    ].sort_values(["view_date", "ticker"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# PI/prior helpers
# -----------------------------------------------------------------------------
def get_market_cap_weights(tickers: list[str]) -> pd.Series:
    caps: dict[str, float] = {}
    for t in tickers:
        info = yf.Ticker(t).info
        cap = info.get("marketCap")
        if cap is None or not np.isfinite(cap) or cap <= 0:
            raise ValueError(f"marketCap invalido para {t}")
        caps[t] = float(cap)
    cap_series = pd.Series(caps, dtype="float64")
    return cap_series / cap_series.sum()


def get_vwap_returns_for_prior(
    tickers: list[str],
    tickers_dir: Path,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    series_list: list[pd.Series] = []
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    for ticker_sa in tickers:
        base_ticker = ticker_sa[:-3] if ticker_sa.endswith(".SA") else ticker_sa
        df = load_hourly_series_for_ticker(tickers_dir, base_ticker)
        df = filter_b3_session(df)
        df = add_vwap_target(df, window_vwap=PREDICTION_LENGTH)

        grouped = df.groupby("view_date", sort=True)
        rows: list[tuple[pd.Timestamp, float]] = []
        for day, day_df in grouped:
            if len(day_df) != PREDICTION_LENGTH:
                continue
            if set(day_df["time"].dt.hour.tolist()) != B3_SESSION_HOURS:
                continue
            rows.append((pd.Timestamp(day), float(day_df["log_ret_vwap"].sum())))

        if not rows:
            raise ValueError(f"Sem retornos VWAP validos para prior em {ticker_sa}.")

        series = pd.Series({d: r for d, r in rows}, name=ticker_sa).sort_index()
        series = series[(series.index >= start_ts) & (series.index <= end_ts)]
        series_list.append(series)

    returns = pd.concat(series_list, axis=1).dropna(how="any")
    if returns.empty:
        raise ValueError("Sem retornos VWAP validos para calcular Sigma do prior.")
    returns.index.name = "view_date"
    return returns


def compute_pi_and_prior_uncertainty(
    returns: pd.DataFrame,
    market_weights: pd.Series,
    risk_free_annual: float,
    tau: float,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, float]:
    sigma = returns.cov()
    risk_free_daily = (1.0 + risk_free_annual) ** (1.0 / 252.0) - 1.0
    market_returns = returns @ market_weights
    delta = (market_returns.mean() - risk_free_daily) / market_returns.var()
    pi = delta * (sigma @ market_weights)
    pi.name = "pi"
    prior_cov = tau * sigma
    return pi, sigma, prior_cov, float(delta)


def compute_daily_bl_posterior(
    pi: pd.Series,
    sigma: pd.DataFrame,
    tau: float,
    q_omega_long: pd.DataFrame,
    delta: float,
    market_weights: pd.Series | None = None,
    returns_history: pd.DataFrame | None = None,
    use_rolling_prior: bool = False,
    rolling_min_obs: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = list(pi.index)
    n = len(tickers)
    static_tau_sigma = tau * sigma.values
    static_inv_tau_sigma = np.linalg.pinv(static_tau_sigma)
    static_pi_vec = pi.reindex(tickers).values.reshape(n, 1)
    static_delta = float(delta)

    mu_rows: list[dict[str, object]] = []
    w_rows: list[dict[str, object]] = []

    for view_date, day_df in q_omega_long.sort_values(["view_date", "ticker"]).groupby("view_date", sort=True):
        inv_tau_sigma = static_inv_tau_sigma
        pi_vec = static_pi_vec
        delta_used = static_delta
        sigma_used = sigma.values
        prior_obs = 0
        prior_end_date = ""

        if use_rolling_prior and returns_history is not None and market_weights is not None:
            vd = pd.Timestamp(view_date)
            hist = returns_history[returns_history.index < vd].copy()
            hist = hist[[c for c in tickers if c in hist.columns]].dropna(how="any")
            prior_obs = int(len(hist))
            if not hist.empty:
                prior_end_date = pd.Timestamp(hist.index.max()).strftime("%Y-%m-%d")

            if len(hist) >= rolling_min_obs:
                aligned_weights = market_weights.reindex(hist.columns)
                pi_roll, sigma_roll, _, delta_roll = compute_pi_and_prior_uncertainty(
                    returns=hist,
                    market_weights=aligned_weights,
                    risk_free_annual=RISK_FREE_ANNUAL,
                    tau=tau,
                )
                sigma_roll = sigma_roll.reindex(index=tickers, columns=tickers)
                pi_roll = pi_roll.reindex(tickers)
                sigma_used = sigma_roll.values
                inv_tau_sigma = np.linalg.pinv(tau * sigma_used)
                pi_vec = pi_roll.values.reshape(n, 1)
                delta_used = float(delta_roll)

        valid_rows = []
        for _, row in day_df.iterrows():
            view_ticker_sa = ticker_to_sa(str(row["ticker"]))
            if view_ticker_sa not in tickers:
                continue
            omega = max(float(row["omega"]), EPS)
            valid_rows.append((view_ticker_sa, float(row["Q"]), omega))

        if not valid_rows:
            continue

        k = len(valid_rows)
        p = np.zeros((k, n), dtype=float)
        q_vec = np.zeros((k, 1), dtype=float)
        omega_diag = np.zeros(k, dtype=float)

        view_tickers_used: list[str] = []
        for i, (view_ticker_sa, q_val, omega_val) in enumerate(valid_rows):
            p[i, tickers.index(view_ticker_sa)] = 1.0
            q_vec[i, 0] = q_val
            omega_diag[i] = omega_val
            view_tickers_used.append(view_ticker_sa)

        inv_omega = np.diag(1.0 / omega_diag)
        a_mat = inv_tau_sigma + p.T @ inv_omega @ p
        b_vec = inv_tau_sigma @ pi_vec + p.T @ inv_omega @ q_vec
        mu_post = np.linalg.pinv(a_mat) @ b_vec

        # Pesos implicitos: w = inv(delta*Sigma) * mu_post
        w_post = np.linalg.pinv(delta_used * sigma_used) @ mu_post
        if np.isfinite(w_post).all() and abs(float(w_post.sum())) > EPS:
            w_post = w_post / float(w_post.sum())

        mu_entry: dict[str, object] = {"view_date": pd.Timestamp(view_date).strftime("%Y-%m-%d")}
        w_entry: dict[str, object] = {"view_date": pd.Timestamp(view_date).strftime("%Y-%m-%d")}
        for i, tk in enumerate(tickers):
            mu_entry[tk] = float(mu_post[i, 0])
            w_entry[tk] = float(w_post[i, 0])

        mu_entry["views_count"] = k
        mu_entry["view_tickers"] = ";".join(view_tickers_used)
        mu_entry["Q_mean"] = float(np.mean(q_vec))
        mu_entry["Omega_mean"] = float(np.mean(omega_diag))
        mu_entry["prior_obs"] = int(prior_obs)
        mu_entry["prior_end_date"] = prior_end_date

        w_entry["views_count"] = k
        w_entry["view_tickers"] = ";".join(view_tickers_used)
        w_entry["Q_mean"] = float(np.mean(q_vec))
        w_entry["Omega_mean"] = float(np.mean(omega_diag))
        w_entry["prior_obs"] = int(prior_obs)
        w_entry["prior_end_date"] = prior_end_date
        mu_rows.append(mu_entry)
        w_rows.append(w_entry)

    return pd.DataFrame(mu_rows), pd.DataFrame(w_rows)


def _long_only_capped_weights(raw_w: np.ndarray, max_w: float) -> np.ndarray:
    """
    Projeta pesos para:
    - long-only (w >= 0)
    - limite por ativo (w <= max_w)
    - soma = 1
    """
    n = len(raw_w)
    if n == 0:
        return raw_w

    if max_w <= 0 or max_w * n < 1.0:
        raise ValueError(
            f"MAX_WEIGHT_PER_ASSET={max_w} inviavel para {n} ativos (precisa max_w*n >= 1)."
        )

    scores = np.maximum(raw_w.astype(float), 0.0)
    if not np.isfinite(scores).all() or scores.sum() <= EPS:
        return np.full(n, 1.0 / n, dtype=float)

    w = np.zeros(n, dtype=float)
    remaining = np.ones(n, dtype=bool)
    budget = 1.0

    # Alocacao iterativa com saturacao no teto.
    while budget > EPS and np.any(remaining):
        idx = np.where(remaining)[0]
        s = scores[idx]
        s_sum = float(s.sum())
        if s_sum <= EPS:
            w[idx] += budget / len(idx)
            break

        proposal = budget * (s / s_sum)
        saturated = proposal >= max_w - EPS

        if not np.any(saturated):
            w[idx] += proposal
            break

        sat_idx = idx[saturated]
        for j in sat_idx:
            alloc = max_w - w[j]
            if alloc > 0:
                w[j] += alloc
                budget -= alloc
            remaining[j] = False

    # Ajuste numerico final.
    w = np.clip(w, 0.0, max_w)
    s = float(w.sum())
    if s > EPS:
        w = w / s
    else:
        w = np.full(n, 1.0 / n, dtype=float)
    return w


def apply_weight_controls(
    posterior_w: pd.DataFrame,
    long_only: bool,
    max_weight_per_asset: float,
) -> pd.DataFrame:
    out = posterior_w.copy()
    asset_cols = [c for c in out.columns if c.endswith(".SA")]
    if not asset_cols:
        return out

    raw_mat = out[asset_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    controlled_mat = np.zeros_like(raw_mat)

    for i in range(raw_mat.shape[0]):
        raw = raw_mat[i]
        if long_only:
            ctrl = _long_only_capped_weights(raw, max_weight_per_asset)
        else:
            # Fallback: somente normaliza soma para 1.
            s = float(raw.sum())
            ctrl = raw / s if abs(s) > EPS else np.full_like(raw, 1.0 / len(raw))
        controlled_mat[i] = ctrl

    out[asset_cols] = controlled_mat
    out["gross_exposure"] = np.abs(controlled_mat).sum(axis=1)
    out["max_weight_used"] = np.max(controlled_mat, axis=1)
    return out


def build_rebalanced_weights(weights: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Gera pesos com holding por frequencia:
    - daily: atualiza todo dia
    - weekly: atualiza no primeiro dia util de cada semana observada
    - monthly: atualiza no primeiro dia util de cada mes observado
    """
    valid_modes = {"daily", "weekly", "monthly"}
    if mode not in valid_modes:
        raise ValueError(f"Modo invalido: {mode}. Use one of {sorted(valid_modes)}")

    out = weights.copy()
    out["view_date"] = pd.to_datetime(out["view_date"], errors="coerce")
    out = out.dropna(subset=["view_date"]).sort_values("view_date").reset_index(drop=True)
    asset_cols = [c for c in out.columns if c.endswith(".SA")]
    if not asset_cols:
        return out

    if mode == "daily":
        out["rebalance_mode"] = "daily"
        out["rebalance_flag"] = 1
        out["rebalance_source_date"] = out["view_date"].dt.strftime("%Y-%m-%d")
        return out

    if mode == "weekly":
        period_key = out["view_date"].dt.to_period("W-FRI").astype(str)
    else:
        period_key = out["view_date"].dt.to_period("M").astype(str)

    first_idx = out.groupby(period_key, sort=False).head(1).index
    rebalance_mask = out.index.isin(first_idx)

    held = out[asset_cols].to_numpy(dtype=float).copy()
    source_dates = np.empty(len(out), dtype=object)
    current_weights: np.ndarray | None = None
    current_source_date = None

    for i in range(len(out)):
        if rebalance_mask[i] or current_weights is None:
            current_weights = held[i].copy()
            current_source_date = out.loc[i, "view_date"]
        else:
            held[i] = current_weights
        source_dates[i] = pd.Timestamp(current_source_date).strftime("%Y-%m-%d")

    out[asset_cols] = held
    out["rebalance_mode"] = mode
    out["rebalance_flag"] = rebalance_mask.astype(int)
    out["rebalance_source_date"] = source_dates
    out["gross_exposure"] = np.abs(held).sum(axis=1)
    out["max_weight_used"] = np.max(np.abs(held), axis=1)
    return out


def main() -> None:
    predictor = load_predictor(MODEL_DIR)
    view_tickers = [ticker_from_sa(t) for t in ATIVOS]

    print("1) Gerando Q diario para todos os ativos de ATIVOS...")
    q_parts: list[pd.DataFrame] = []
    for ticker in view_tickers:
        try:
            sentiment = load_daily_sentiment_for_ticker(NEWS_PATH, ticker=ticker)
            prices = load_hourly_series_for_ticker(TICKERS_DIR, ticker=ticker)
            prices = filter_b3_session(prices)
            prices = add_vwap_target(prices, window_vwap=PREDICTION_LENGTH)
            views = walk_forward_predict_7h(
                df=prices,
                predictor=predictor,
                ticker=ticker,
                period_start=PERIOD_START,
                period_end=PERIOD_END,
            )
            q_parts.append(combine_ai_and_sentiment(views, sentiment, alpha=ALPHA_Q))
            print(f"  - {ticker}: OK ({len(views)} dias)")
        except Exception as exc:
            print(f"  - {ticker}: ignorado ({exc})")

    if not q_parts:
        raise RuntimeError("Nenhuma view Q foi gerada para os ativos configurados.")
    q_long = pd.concat(q_parts, axis=0, ignore_index=True).sort_values(["view_date", "ticker"]).reset_index(drop=True)

    print("2) Gerando Omega diario...")
    omega_long = calculate_omega(q_long, tau=TAU)
    q_omega_long = q_long.merge(
        omega_long[["view_date", "ticker", "omega", "omega_base", "view_confidence"]],
        on=["view_date", "ticker"],
        how="inner",
    ).sort_values(["view_date", "ticker"]).reset_index(drop=True)

    print("3) Gerando PI e incerteza do prior (tau*Sigma) com retornos VWAP...")
    market_weights = get_market_cap_weights(ATIVOS)
    returns = get_vwap_returns_for_prior(
        tickers=ATIVOS,
        tickers_dir=TICKERS_DIR,
        start_date=PRIOR_START_DATE,
        end_date=PRIOR_END_DATE,
    )
    common = [c for c in returns.columns if c in market_weights.index]
    returns = returns[common].copy()
    market_weights = market_weights[common].copy()
    pi, sigma, prior_cov, delta = compute_pi_and_prior_uncertainty(
        returns=returns,
        market_weights=market_weights,
        risk_free_annual=RISK_FREE_ANNUAL,
        tau=TAU,
    )

    print("4) Calculando posterior diario do Black-Litterman...")
    posterior_mu, posterior_w = compute_daily_bl_posterior(
        pi=pi,
        sigma=sigma,
        tau=TAU,
        q_omega_long=q_omega_long,
        delta=delta,
        market_weights=market_weights,
        returns_history=returns,
        use_rolling_prior=USE_ROLLING_PRIOR,
        rolling_min_obs=ROLLING_MIN_OBS,
    )
    posterior_w_controlled = apply_weight_controls(
        posterior_w=posterior_w,
        long_only=LONG_ONLY,
        max_weight_per_asset=MAX_WEIGHT_PER_ASSET,
    )
    posterior_w_ctrl_daily = build_rebalanced_weights(posterior_w_controlled, mode="daily")
    posterior_w_ctrl_weekly = build_rebalanced_weights(posterior_w_controlled, mode="weekly")
    posterior_w_ctrl_monthly = build_rebalanced_weights(posterior_w_controlled, mode="monthly")

    print("5) Salvando resultados...")
    q_out = q_long.copy()
    q_out["view_date"] = q_out["view_date"].dt.strftime("%Y-%m-%d")
    o_out = omega_long.copy()
    o_out["view_date"] = o_out["view_date"].dt.strftime("%Y-%m-%d")
    qo_out = q_omega_long.copy()
    qo_out["view_date"] = qo_out["view_date"].dt.strftime("%Y-%m-%d")

    q_out.to_csv(OUT_Q_LONG, index=False)
    o_out.to_csv(OUT_OMEGA_LONG, index=False)
    qo_out.to_csv(OUT_Q_OMEGA_LONG, index=False)
    pi.to_frame().reset_index().rename(columns={"index": "ticker"}).to_csv(OUT_PI, index=False)
    sigma.to_csv(OUT_SIGMA)
    prior_cov.to_csv(OUT_PRIOR_COV)
    posterior_mu.to_csv(OUT_POSTERIOR_MU, index=False)
    posterior_w.to_csv(OUT_POSTERIOR_W, index=False)
    posterior_w_controlled.to_csv(OUT_POSTERIOR_W_CONTROLLED, index=False)
    posterior_w_ctrl_daily.to_csv(OUT_POSTERIOR_W_CTRL_DAILY, index=False)
    posterior_w_ctrl_weekly.to_csv(OUT_POSTERIOR_W_CTRL_WEEKLY, index=False)
    posterior_w_ctrl_monthly.to_csv(OUT_POSTERIOR_W_CTRL_MONTHLY, index=False)

    print(f"Q diario: {OUT_Q_LONG}")
    print(f"Omega diario: {OUT_OMEGA_LONG}")
    print(f"Q+Omega diario: {OUT_Q_OMEGA_LONG}")
    print(f"PI: {OUT_PI}")
    print(f"Sigma: {OUT_SIGMA}")
    print(f"tau*Sigma: {OUT_PRIOR_COV}")
    print(f"Posterior mu: {OUT_POSTERIOR_MU}")
    print(f"Posterior pesos: {OUT_POSTERIOR_W}")
    print(f"Posterior pesos controlados: {OUT_POSTERIOR_W_CONTROLLED}")
    print(f"Pesos controlados daily: {OUT_POSTERIOR_W_CTRL_DAILY}")
    print(f"Pesos controlados weekly: {OUT_POSTERIOR_W_CTRL_WEEKLY}")
    print(f"Pesos controlados monthly: {OUT_POSTERIOR_W_CTRL_MONTHLY}")
    print(f"Controles: LONG_ONLY={LONG_ONLY} | MAX_WEIGHT_PER_ASSET={MAX_WEIGHT_PER_ASSET:.2f}")
    print(f"Prior temporal: USE_ROLLING_PRIOR={USE_ROLLING_PRIOR} | ROLLING_MIN_OBS={ROLLING_MIN_OBS}")
    print(f"Views processadas (linhas Q/Omega): {len(q_omega_long)}")
    print(f"Dias no posterior: {len(posterior_mu)}")
    print(f"Delta: {delta:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro no pipeline hibrido: {exc}")
        sys.exit(1)
