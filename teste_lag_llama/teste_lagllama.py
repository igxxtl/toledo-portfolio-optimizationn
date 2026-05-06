import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.model.predictor import Predictor

ROOT = Path(__file__).resolve().parent
TICKERS_DIR = ROOT / "tickers_data"
MODEL_DIR = ROOT / "modelo_7h_vwap"
LAG_LLAMA_DIR = ROOT / "lag-Llama"

OUT_RESULTS = ROOT / "teste_lagllama_resultados_diario.csv"
OUT_SAMPLES = ROOT / "teste_lagllama_amostras_diarias.csv"
OUT_METHODS = ROOT / "teste_lagllama_metodos_diario.csv"
OUT_METRICS = ROOT / "teste_lagllama_metricas_diario.csv"
OUT_PLOT = ROOT / "teste_lagllama_plot_diario_amostras_vs_real.png"

TICKER = "ABEV3"
PERIOD_START = "202506"
PERIOD_END = "202512"
PRED_LEN = 1
MIN_CTX_DAYS = 60
MAX_CONTEXT_DAYS: int | None = 150
NUM_SAMPLES = 100
EPS = 1e-6

TARGET_PRICE_COL = "vwap"
TARGET_RET_COL = "ret_vwap_day"


def month_bounds(start: str, end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = pd.Timestamp(year=int(start[:4]), month=int(start[4:]), day=1)
    b = pd.Timestamp(year=int(end[:4]), month=int(end[4:]), day=1) + pd.offsets.MonthEnd(1)
    return a.normalize(), b.normalize()


def load_daily_returns(ticker: str) -> pd.DataFrame:
    fp = TICKERS_DIR / f"BMFBOVESPA_DLY_{ticker}, 60.csv"
    if not fp.exists():
        raise FileNotFoundError(f"CSV nao encontrado: {fp}")

    df = pd.read_csv(fp)
    if "time" not in df.columns:
        raise ValueError("CSV sem coluna 'time'.")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None)
    col_map = {c.lower(): c for c in df.columns}

    # Cria VWAP caso o CSV nao tenha a coluna pronta.
    if TARGET_PRICE_COL not in df.columns:
        high_col = col_map.get("high")
        low_col = col_map.get("low")
        close_col = col_map.get("close")
        vol_col = col_map.get("volume")
        if high_col is None or low_col is None or close_col is None:
            raise ValueError("Nao foi possivel criar VWAP: faltam colunas high/low/close.")

        high = pd.to_numeric(df[high_col], errors="coerce")
        low = pd.to_numeric(df[low_col], errors="coerce")
        close = pd.to_numeric(df[close_col], errors="coerce")
        typical_price = (high + low + close) / 3.0

        if vol_col is not None:
            volume = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
            day_key = df["time"].dt.normalize()
            cum_pv = (typical_price * volume).groupby(day_key).cumsum()
            cum_v = volume.groupby(day_key).cumsum().replace(0.0, np.nan)
            df[TARGET_PRICE_COL] = (cum_pv / cum_v).astype("float32")
        else:
            df[TARGET_PRICE_COL] = typical_price.astype("float32")

    df[TARGET_PRICE_COL] = pd.to_numeric(df[TARGET_PRICE_COL], errors="coerce").astype("float32")
    df = df.dropna(subset=["time", TARGET_PRICE_COL]).sort_values("time").reset_index(drop=True)

    df["ret_vwap"] = df[TARGET_PRICE_COL].pct_change().astype("float32")
    df = df.dropna(subset=["ret_vwap"]).copy()
    df["view_date"] = df["time"].dt.normalize()

    daily = (
        df.groupby("view_date", as_index=False)
        .agg(**{TARGET_RET_COL: ("ret_vwap", lambda x: float(np.prod(1.0 + x.to_numpy(dtype=float)) - 1.0))})
        .sort_values("view_date")
        .reset_index(drop=True)
    )
    daily["ticker"] = ticker
    daily["time"] = pd.to_datetime(daily["view_date"])
    daily["time_fake"] = pd.date_range(start=daily["time"].min(), periods=len(daily), freq="D")
    daily[TARGET_RET_COL] = pd.to_numeric(daily[TARGET_RET_COL], errors="coerce").astype("float32")
    return daily[["ticker", "time", "time_fake", "view_date", TARGET_RET_COL]]


def load_predictor() -> Predictor:
    if LAG_LLAMA_DIR.exists():
        sys.path.insert(0, str(LAG_LLAMA_DIR))
    return Predictor.deserialize(MODEL_DIR)


def run_daily(df: pd.DataFrame, predictor: Predictor) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end = month_bounds(PERIOD_START, PERIOD_END)
    groups = df.groupby("view_date", sort=True)
    days = [d for d in sorted(groups.groups.keys()) if start <= d <= end]

    results: list[dict] = []
    samples: list[dict] = []
    for d in days:
        y = groups.get_group(d).sort_values("time")
        if len(y) != PRED_LEN:
            continue

        ctx = df.iloc[: int(y.index.min())].copy()
        if len(ctx) < MIN_CTX_DAYS:
            continue
        if MAX_CONTEXT_DAYS is not None:
            ctx = ctx.tail(MAX_CONTEXT_DAYS)

        ds = PandasDataset.from_long_dataframe(
            ctx[["ticker", "time_fake", TARGET_RET_COL]],
            item_id="ticker",
            timestamp="time_fake",
            target=TARGET_RET_COL,
            freq="D",
        )
        forecasts = make_evaluation_predictions(dataset=ds, predictor=predictor, num_samples=NUM_SAMPLES)[0]
        fc = list(forecasts)[0]

        p = np.asarray(fc.mean[:PRED_LEN], dtype=float)
        r = y[TARGET_RET_COL].to_numpy(dtype=float)
        smp = np.asarray(fc.samples[:, :PRED_LEN], dtype=float)

        results.append(
            {
                "view_date": d,
                "ticker": TICKER,
                "ret_pred_model_mean": float(np.prod(1.0 + p) - 1.0),
                "ret_real": float(np.prod(1.0 + r) - 1.0),
                "context_days_used": int(len(ctx)),
            }
        )
        for sid in range(smp.shape[0]):
            samples.append(
                {
                    "time_hour": y["time"].iloc[0],
                    "sample_id": sid,
                    "pred_sample": float(smp[sid, 0]),
                    "real_value": float(r[0]),
                }
            )

    return pd.DataFrame(results), pd.DataFrame(samples)


def build_methods(samples_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for t, g in samples_df.groupby("time_hour", sort=True):
        s = g["pred_sample"].to_numpy(dtype=float)
        real = float(g["real_value"].iloc[0])

        q10, q25, q50, q75, q90 = np.quantile(s, [0.10, 0.25, 0.50, 0.75, 0.90])
        std = float(np.std(s, ddof=0))
        pod = float(np.mean(s > 0.0))
        abs_mean = float(np.mean(np.abs(s)))

        pred_median = float(q50)
        pred_conf_filter = float(q25) if q25 > 0 else (float(q75) if q75 < 0 else 0.0)
        pred_pod = float((pod - 0.5) * 2.0 * abs_mean)
        trimmed = s[(s >= q10) & (s <= q90)]
        pred_trimmed_mean = float(np.mean(trimmed)) if len(trimmed) > 0 else float(np.mean(s))
        pred_zscore = float(q50 / (std + EPS))

        rows.append(
            {
                "time_hour": t,
                "real_value": real,
                "pred_median": pred_median,
                "pred_conf_filter": pred_conf_filter,
                "pred_pod": pred_pod,
                "pred_trimmed_mean": pred_trimmed_mean,
                "pred_zscore": pred_zscore,
            }
        )

    return pd.DataFrame(rows).sort_values("time_hour").reset_index(drop=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    e = y_pred - y_true
    eps = 1e-12
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": len(y_true),
        "hit_ratio_pct": float(np.mean(np.sign(y_pred) == np.sign(y_true)) * 100.0),
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mape_pct": float(np.mean(np.abs(e) / (np.abs(y_true) + eps)) * 100.0),
        "smape_pct": float(np.mean(2 * np.abs(e) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0),
        "r2": float(1.0 - ss_res / (ss_tot + eps)),
        "corr": float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_true) > 1 else np.nan,
        "bias": float(np.mean(e)),
    }


def evaluate_methods(methods_df: pd.DataFrame) -> pd.DataFrame:
    y = methods_df["real_value"].to_numpy(dtype=float)
    cols = ["pred_median", "pred_conf_filter", "pred_pod", "pred_trimmed_mean", "pred_zscore"]
    out = []
    for c in cols:
        out.append({"method": c, **metrics(y, methods_df[c].to_numpy(dtype=float))})
    return pd.DataFrame(out).sort_values(["r2", "rmse"], ascending=[False, True]).reset_index(drop=True)


def plot_fan(samples_df: pd.DataFrame) -> None:
    if samples_df.empty:
        return

    g = (
        samples_df.groupby("time_hour")
        .agg(
            pred_q10=("pred_sample", lambda x: float(np.quantile(x, 0.10))),
            pred_q25=("pred_sample", lambda x: float(np.quantile(x, 0.25))),
            pred_med=("pred_sample", "median"),
            pred_q75=("pred_sample", lambda x: float(np.quantile(x, 0.75))),
            pred_q90=("pred_sample", lambda x: float(np.quantile(x, 0.90))),
            real=("real_value", "first"),
        )
        .reset_index()
    )
    g["time_hour"] = pd.to_datetime(g["time_hour"])

    plt.figure(figsize=(14, 6))
    plt.fill_between(g["time_hour"], g["pred_q10"], g["pred_q90"], alpha=0.12, color="tab:blue", label="P10-P90")
    plt.fill_between(g["time_hour"], g["pred_q25"], g["pred_q75"], alpha=0.22, color="tab:blue", label="P25-P75")
    plt.plot(g["time_hour"], g["pred_med"], color="tab:orange", label="Mediana prevista")
    plt.plot(g["time_hour"], g["real"], color="tab:green", label="Real")
    plt.title("Lag-Llama diario: fan chart previsto vs real")
    plt.xlabel("Dia previsto")
    plt.ylabel("Retorno diario")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=140)
    plt.close()


def main() -> None:
    print(f"Teste Lag-Llama diario | {TICKER} | {PERIOD_START}-{PERIOD_END} | target={TARGET_RET_COL}")
    df = load_daily_returns(TICKER)
    predictor = load_predictor()
    results_df, samples_df = run_daily(df, predictor)
    if results_df.empty or samples_df.empty:
        raise RuntimeError("Nenhuma previsao diaria gerada.")

    out_r = results_df.copy()
    out_r["view_date"] = pd.to_datetime(out_r["view_date"]).dt.strftime("%Y-%m-%d")
    out_r.to_csv(OUT_RESULTS, index=False)

    out_s = samples_df.copy()
    out_s["time_hour"] = pd.to_datetime(out_s["time_hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out_s.to_csv(OUT_SAMPLES, index=False)

    methods_df = build_methods(samples_df)
    out_m = methods_df.copy()
    out_m["time_hour"] = pd.to_datetime(out_m["time_hour"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out_m.to_csv(OUT_METHODS, index=False)

    metrics_df = evaluate_methods(methods_df)
    metrics_df.to_csv(OUT_METRICS, index=False)

    plot_fan(samples_df)
    print(metrics_df.round(6).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro no teste Lag-Llama: {exc}")
        sys.exit(1)