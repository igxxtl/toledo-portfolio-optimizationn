"""
Calcula um score composto (0-100) para o pipeline Black-Litterman hibrido.

Componentes:
- Predicao (S_pred): direcional + erro absoluto
- Portfolio (S_port): retorno acumulado + Sharpe - drawdown
- Risco (S_risk): alavancagem/exposicao e concentracao de pesos

Saidas:
- bl_hibrido_score_summary.csv
- bl_hibrido_score_mensal.csv
- bl_hibrido_rebalance_resumo.csv
- bl_hibrido_score_previsao_metricas.csv
- bl_hibrido_score_previsao_por_ticker.csv
- bl_hibrido_score_previsao_por_mes.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    from .config import (
        INITIAL_CAPITAL_BRL,
        OUT_GAIN_SERIES_DAILY,
        OUT_GAIN_SERIES_MONTHLY,
        OUT_GAIN_SERIES_WEEKLY,
        OUT_POSTERIOR_W,
        OUT_POSTERIOR_W_CONTROLLED,
        OUT_POSTERIOR_W_CTRL_DAILY,
        OUT_POSTERIOR_W_CTRL_MONTHLY,
        OUT_POSTERIOR_W_CTRL_WEEKLY,
        OUT_Q_LONG,
        OUT_SCORE_MONTHLY,
        OUT_SCORE_PRED_METRICS,
        OUT_SCORE_PRED_MONTH,
        OUT_SCORE_PRED_TICKER,
        OUT_SCORE_REBALANCE,
        OUT_GAIN_FALLBACK,
        OUT_SCORE_SUMMARY,
        TRADING_DAYS_YEAR,
    )
    from .io_utils import read_csv_with_date
    from .xgb_views import normalize_return_columns
except ImportError:
    from config import (
        INITIAL_CAPITAL_BRL,
        OUT_GAIN_SERIES_DAILY,
        OUT_GAIN_SERIES_MONTHLY,
        OUT_GAIN_SERIES_WEEKLY,
        OUT_POSTERIOR_W,
        OUT_POSTERIOR_W_CONTROLLED,
        OUT_POSTERIOR_W_CTRL_DAILY,
        OUT_POSTERIOR_W_CTRL_MONTHLY,
        OUT_POSTERIOR_W_CTRL_WEEKLY,
        OUT_Q_LONG,
        OUT_SCORE_MONTHLY,
        OUT_SCORE_PRED_METRICS,
        OUT_SCORE_PRED_MONTH,
        OUT_SCORE_PRED_TICKER,
        OUT_SCORE_REBALANCE,
        OUT_GAIN_FALLBACK,
        OUT_SCORE_SUMMARY,
        TRADING_DAYS_YEAR,
    )
    from io_utils import read_csv_with_date
    from xgb_views import normalize_return_columns

W_PRED = 0.35
W_PORT = 0.45
W_RISK = 0.20


def clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return default if abs(b) < 1e-12 else float(a / b)


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    q = normalize_return_columns(pd.read_csv(OUT_Q_LONG))
    q["view_date"] = pd.to_datetime(q["view_date"], errors="coerce")
    q = q.dropna(subset=["view_date"]).sort_values("view_date").reset_index(drop=True)

    gains_by_mode: dict[str, pd.DataFrame] = {}
    if OUT_GAIN_SERIES_DAILY.exists() and OUT_GAIN_SERIES_WEEKLY.exists() and OUT_GAIN_SERIES_MONTHLY.exists():
        gains_by_mode["daily"] = read_csv_with_date(OUT_GAIN_SERIES_DAILY)
        gains_by_mode["weekly"] = read_csv_with_date(OUT_GAIN_SERIES_WEEKLY)
        gains_by_mode["monthly"] = read_csv_with_date(OUT_GAIN_SERIES_MONTHLY)
    elif OUT_GAIN_FALLBACK.exists():
        gains_by_mode["daily"] = read_csv_with_date(OUT_GAIN_FALLBACK)
    else:
        raise FileNotFoundError("Não encontrei séries de ganho. Execute plot_ganho_estimado.py antes.")

    weights_by_mode: dict[str, pd.DataFrame] = {}
    if OUT_POSTERIOR_W_CTRL_DAILY.exists() and OUT_POSTERIOR_W_CTRL_WEEKLY.exists() and OUT_POSTERIOR_W_CTRL_MONTHLY.exists():
        weights_by_mode["daily"] = read_csv_with_date(OUT_POSTERIOR_W_CTRL_DAILY)
        weights_by_mode["weekly"] = read_csv_with_date(OUT_POSTERIOR_W_CTRL_WEEKLY)
        weights_by_mode["monthly"] = read_csv_with_date(OUT_POSTERIOR_W_CTRL_MONTHLY)
    else:
        w_path = OUT_POSTERIOR_W_CONTROLLED if OUT_POSTERIOR_W_CONTROLLED.exists() else OUT_POSTERIOR_W
        weights_by_mode["daily"] = read_csv_with_date(w_path)

    return q, gains_by_mode, weights_by_mode


def score_predicao(q: pd.DataFrame) -> tuple[float, dict[str, float]]:
    q2 = normalize_return_columns(q.copy())
    q2["ret_pred"] = pd.to_numeric(q2["ret_pred"], errors="coerce")
    q2["ret_real"] = pd.to_numeric(q2["ret_real"], errors="coerce")
    q2 = q2.dropna(subset=["ret_pred", "ret_real"])
    if q2.empty:
        return 0.0, {"hit_ratio": 0.0, "mae": np.nan, "mae_score": 0.0}

    hit_ratio = float((np.sign(q2["ret_pred"]) == np.sign(q2["ret_real"])).mean())
    mae = float((q2["ret_pred"] - q2["ret_real"]).abs().mean())

    # Referencia robusta: mediana de |ret_real| para escalar o erro.
    mae_ref = float(q2["ret_real"].abs().median())
    if not np.isfinite(mae_ref) or mae_ref <= 0:
        mae_ref = 0.01
    mae_score = clamp01(1.0 - (mae / mae_ref))

    s_pred = clamp01(0.6 * hit_ratio + 0.4 * mae_score)
    details = {"hit_ratio": hit_ratio, "mae": mae, "mae_score": mae_score}
    return s_pred, details


def forecast_metrics(q: pd.DataFrame) -> dict[str, float]:
    q2 = normalize_return_columns(q.copy())
    q2["ret_pred"] = pd.to_numeric(q2["ret_pred"], errors="coerce")
    q2["ret_real"] = pd.to_numeric(q2["ret_real"], errors="coerce")
    q2 = q2.dropna(subset=["ret_pred", "ret_real"])
    if q2.empty:
        return {
            "n": 0.0,
            "hit_ratio": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mape_pct": np.nan,
            "smape_pct": np.nan,
            "r2": np.nan,
            "score_pred_metricas": 0.0,
        }

    y_true = q2["ret_real"].to_numpy(dtype=float)
    y_pred = q2["ret_pred"].to_numpy(dtype=float)
    err = y_pred - y_true

    hit_ratio = float((np.sign(y_pred) == np.sign(y_true)).mean())
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))

    # MAPE com filtro para evitar divisao por ~0.
    denom = np.abs(y_true)
    mask = denom > 1e-9
    if mask.any():
        mape = float(np.mean(np.abs(err[mask]) / denom[mask]) * 100.0)
    else:
        mape = np.nan

    smape = float(
        np.mean(2.0 * np.abs(err) / (np.abs(y_true) + np.abs(y_pred) + 1e-12)) * 100.0
    )

    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    # Score de previsao por metricas (0-100):
    # mistura direcional + erro relativo + explicabilidade.
    hr_score = clamp01(hit_ratio)
    smape_score = clamp01(1.0 - (smape / 200.0))  # sMAPE max teorico ~200%
    r2_score = clamp01((r2 + 1.0) / 2.0) if np.isfinite(r2) else 0.0
    score_pred_metricas = 100.0 * (0.4 * hr_score + 0.4 * smape_score + 0.2 * r2_score)

    return {
        "n": float(len(q2)),
        "hit_ratio": hit_ratio,
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "smape_pct": smape,
        "r2": r2,
        "score_pred_metricas": score_pred_metricas,
    }


def build_forecast_metric_tables(q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame([forecast_metrics(q)])

    by_ticker_rows = []
    for ticker, g in q.groupby("ticker", sort=True):
        row = forecast_metrics(g)
        row["ticker"] = str(ticker)
        by_ticker_rows.append(row)
    by_ticker = pd.DataFrame(by_ticker_rows)
    if not by_ticker.empty:
        by_ticker = by_ticker[
            ["ticker", "n", "hit_ratio", "mae", "rmse", "mape_pct", "smape_pct", "r2", "score_pred_metricas"]
        ].sort_values("ticker")

    q_month = q.copy()
    q_month["mes"] = q_month["view_date"].dt.to_period("M").astype(str)
    by_month_rows = []
    for month, g in q_month.groupby("mes", sort=True):
        row = forecast_metrics(g)
        row["mes"] = month
        by_month_rows.append(row)
    by_month = pd.DataFrame(by_month_rows)
    if not by_month.empty:
        by_month = by_month[
            ["mes", "n", "hit_ratio", "mae", "rmse", "mape_pct", "smape_pct", "r2", "score_pred_metricas"]
        ].sort_values("mes")

    return overall, by_ticker, by_month


def score_portfolio(gain: pd.DataFrame) -> tuple[float, dict[str, float]]:
    g = gain.copy()
    g["ret_est_portfolio"] = pd.to_numeric(g["ret_est_portfolio"], errors="coerce")
    g["capital_est"] = pd.to_numeric(g["capital_est"], errors="coerce")
    g = g.dropna(subset=["ret_est_portfolio", "capital_est"])
    if g.empty:
        return 0.0, {"total_return": np.nan, "sharpe": np.nan, "max_drawdown": np.nan}

    ret = g["ret_est_portfolio"]
    total_return = float(g["capital_est"].iloc[-1] - 1.0)

    mu = float(ret.mean())
    sigma = float(ret.std(ddof=1))
    sharpe = safe_div(mu, sigma, default=0.0) * np.sqrt(TRADING_DAYS_YEAR) if sigma > 0 else 0.0

    running_max = g["capital_est"].cummax()
    drawdown = (g["capital_est"] / running_max) - 1.0
    max_drawdown = float(drawdown.min())  # negativo
    mdd_abs = abs(max_drawdown)

    # Normalizacoes simples/robustas
    ret_score = clamp01((total_return + 0.20) / 0.40)  # -20% -> 0 ; +20% -> 1
    sharpe_score = clamp01((sharpe + 1.0) / 2.0)       # -1 -> 0 ; +1 -> 1
    dd_score = clamp01(1.0 - (mdd_abs / 0.30))         # 30% de DD zera score

    s_port = clamp01(0.45 * ret_score + 0.35 * sharpe_score + 0.20 * dd_score)
    details = {
        "total_return": total_return,
        "vol_annual": sigma * np.sqrt(TRADING_DAYS_YEAR),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "ret_score": ret_score,
        "sharpe_score": sharpe_score,
        "dd_score": dd_score,
    }
    return s_port, details


def score_risco(weights: pd.DataFrame) -> tuple[float, dict[str, float]]:
    w = weights.copy()
    asset_cols = [c for c in w.columns if c.endswith(".SA")]
    if not asset_cols:
        return 0.0, {"avg_gross_exposure": np.nan, "avg_max_weight": np.nan}

    wmat = w[asset_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    gross = np.abs(wmat).sum(axis=1)
    maxw = np.max(np.abs(wmat), axis=1)

    avg_gross = float(np.mean(gross))
    avg_maxw = float(np.mean(maxw))

    # 1.0 de gross e 0.35 de max_weight sao "bons" no setup controlado.
    gross_score = clamp01(1.0 - max(avg_gross - 1.0, 0.0) / 1.0)
    concentration_score = clamp01(1.0 - max(avg_maxw - 0.35, 0.0) / 0.35)

    s_risk = clamp01(0.6 * gross_score + 0.4 * concentration_score)
    details = {
        "avg_gross_exposure": avg_gross,
        "avg_max_weight": avg_maxw,
        "gross_score": gross_score,
        "concentration_score": concentration_score,
    }
    return s_risk, details


def score_mensal(q: pd.DataFrame, gain: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    months = sorted(set(q["view_date"].dt.to_period("M")).intersection(gain["view_date"].dt.to_period("M")))
    rows: list[dict[str, float | str]] = []
    for m in months:
        q_m = q[q["view_date"].dt.to_period("M") == m]
        g_m = gain[gain["view_date"].dt.to_period("M") == m]
        w_m = w[w["view_date"].dt.to_period("M") == m]

        sp, _ = score_predicao(q_m)
        so, _ = score_portfolio(g_m)
        sr, _ = score_risco(w_m)
        total = 100.0 * (W_PRED * sp + W_PORT * so + W_RISK * sr)
        rows.append(
            {
                "mes": str(m),
                "score_pred": 100.0 * sp,
                "score_port": 100.0 * so,
                "score_risk": 100.0 * sr,
                "score_total": total,
                "dias": int(len(g_m)),
            }
        )
    return pd.DataFrame(rows)


def summarize_rebalance_modes(
    q: pd.DataFrame,
    gains_by_mode: dict[str, pd.DataFrame],
    weights_by_mode: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    s_pred, _ = score_predicao(q)
    rows: list[dict[str, float | str]] = []

    for mode, gain_df in gains_by_mode.items():
        w_df = weights_by_mode.get(mode, weights_by_mode.get("daily"))
        s_port, d_port = score_portfolio(gain_df)
        s_risk, d_risk = score_risco(w_df if w_df is not None else pd.DataFrame())
        s_total = 100.0 * (W_PRED * s_pred + W_PORT * s_port + W_RISK * s_risk)

        final_cap_brl = np.nan
        if "capital_est_brl" in gain_df.columns and not gain_df.empty:
            final_cap_brl = float(pd.to_numeric(gain_df["capital_est_brl"], errors="coerce").dropna().iloc[-1])

        rows.append(
            {
                "mode": mode,
                "score_total": s_total,
                "score_pred": 100.0 * s_pred,
                "score_port": 100.0 * s_port,
                "score_risk": 100.0 * s_risk,
                "total_return_est": d_port.get("total_return"),
                "vol_annual_est": d_port.get("vol_annual"),
                "sharpe_est": d_port.get("sharpe"),
                "max_drawdown_est": d_port.get("max_drawdown"),
                "final_capital_est_brl": final_cap_brl,
                "avg_gross_exposure": d_risk.get("avg_gross_exposure"),
                "avg_max_weight": d_risk.get("avg_max_weight"),
                "dias": int(len(gain_df)),
            }
        )

    out = pd.DataFrame(rows).sort_values("score_total", ascending=False).reset_index(drop=True)
    winner = str(out["mode"].iloc[0]) if not out.empty else "n/a"
    return out, winner


def main() -> None:
    q, gains_by_mode, weights_by_mode = load_inputs()
    base_mode = "daily" if "daily" in gains_by_mode else sorted(gains_by_mode.keys())[0]
    gain = gains_by_mode[base_mode]
    w = weights_by_mode.get(base_mode, list(weights_by_mode.values())[0])

    s_pred, d_pred = score_predicao(q)
    s_port, d_port = score_portfolio(gain)
    s_risk, d_risk = score_risco(w)
    s_total = 100.0 * (W_PRED * s_pred + W_PORT * s_port + W_RISK * s_risk)

    summary = pd.DataFrame(
        [
            {
                "score_total": s_total,
                "score_pred": 100.0 * s_pred,
                "score_port": 100.0 * s_port,
                "score_risk": 100.0 * s_risk,
                "hit_ratio": d_pred.get("hit_ratio"),
                "mae": d_pred.get("mae"),
                "total_return_est": d_port.get("total_return"),
                "sharpe_est": d_port.get("sharpe"),
                "max_drawdown_est": d_port.get("max_drawdown"),
                "avg_gross_exposure": d_risk.get("avg_gross_exposure"),
                "avg_max_weight": d_risk.get("avg_max_weight"),
            }
        ]
    )
    summary.to_csv(OUT_SCORE_SUMMARY, index=False)

    pred_overall, pred_by_ticker, pred_by_month = build_forecast_metric_tables(q)
    pred_overall.to_csv(OUT_SCORE_PRED_METRICS, index=False)
    pred_by_ticker.to_csv(OUT_SCORE_PRED_TICKER, index=False)
    pred_by_month.to_csv(OUT_SCORE_PRED_MONTH, index=False)

    monthly = score_mensal(q, gain, w)
    monthly.to_csv(OUT_SCORE_MONTHLY, index=False)

    rebalance, winner = summarize_rebalance_modes(q, gains_by_mode, weights_by_mode)
    rebalance.to_csv(OUT_SCORE_REBALANCE, index=False)

    print(f"Score total: {s_total:.2f}/100")
    print(f"- Predicao: {100.0 * s_pred:.2f}")
    print(f"- Portfolio: {100.0 * s_port:.2f}")
    print(f"- Risco: {100.0 * s_risk:.2f}")
    print(f"Modo base do score: {base_mode}")
    print(f"Vencedor entre rebalanceamentos: {winner}")
    if not pred_overall.empty:
        p = pred_overall.iloc[0]
        print(
            "Predicao (metricas) | hit_ratio={hr:.2f}% | RMSE={rmse:.6f} | "
            "sMAPE={smape:.2f}% | R2={r2:.4f} | score={score:.2f}/100".format(
                hr=float(p["hit_ratio"]) * 100.0 if np.isfinite(p["hit_ratio"]) else float("nan"),
                rmse=float(p["rmse"]) if np.isfinite(p["rmse"]) else float("nan"),
                smape=float(p["smape_pct"]) if np.isfinite(p["smape_pct"]) else float("nan"),
                r2=float(p["r2"]) if np.isfinite(p["r2"]) else float("nan"),
                score=float(p["score_pred_metricas"]) if np.isfinite(p["score_pred_metricas"]) else float("nan"),
            )
        )
    print(f"Resumo salvo em: {OUT_SCORE_SUMMARY}")
    print(f"Score mensal salvo em: {OUT_SCORE_MONTHLY}")
    print(f"Resumo por modo salvo em: {OUT_SCORE_REBALANCE}")
    print(f"Predição (geral) salva em: {OUT_SCORE_PRED_METRICS}")
    print(f"Predição por ticker salva em: {OUT_SCORE_PRED_TICKER}")
    print(f"Predição por mês salva em: {OUT_SCORE_PRED_MONTH}")


if __name__ == "__main__":
    main()
