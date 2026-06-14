"""
Compara Black-Litterman hibrido vs Markowitz puro.

Markowitz puro neste script:
- esperado: media historica dos retornos (rolling)
- risco: covariancia historica dos retornos (rolling)
- pesos base: inv(Sigma) * mu
- projecao para long-only com teto por ativo (para comparacao realista)

Saidas:
- bl_vs_markowitz_comparativo.csv
- bl_vs_markowitz_resumo.csv
- bl_vs_markowitz_plot.png
- markowitz_weights_daily.csv
- markowitz_weights_weekly.csv
- markowitz_weights_monthly.csv
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .config import (
        ATIVOS,
        EXECUTION_LAG_DAYS,
        INITIAL_CAPITAL_BRL,
        MAX_WEIGHT_PER_ASSET,
        OUT_MKZ_COMP,
        OUT_MKZ_PLOT,
        OUT_MKZ_SUMMARY,
        OUT_MKZ_W_DAILY,
        OUT_MKZ_W_MONTHLY,
        OUT_MKZ_W_WEEKLY,
        OUT_GAIN_SERIES_DAILY,
        OUT_GAIN_SERIES_MONTHLY,
        OUT_GAIN_SERIES_WEEKLY,
        PERIOD_END,
        PERIOD_START,
        SELIC_ANNUAL,
    )
    from .prior import get_daily_returns_for_prior
    from .weights import long_only_capped_weights
    from .xgb_views import month_bounds
except ImportError:
    from config import (
        ATIVOS,
        EXECUTION_LAG_DAYS,
        INITIAL_CAPITAL_BRL,
        MAX_WEIGHT_PER_ASSET,
        OUT_MKZ_COMP,
        OUT_MKZ_PLOT,
        OUT_MKZ_SUMMARY,
        OUT_MKZ_W_DAILY,
        OUT_MKZ_W_MONTHLY,
        OUT_MKZ_W_WEEKLY,
        OUT_GAIN_SERIES_DAILY,
        OUT_GAIN_SERIES_MONTHLY,
        OUT_GAIN_SERIES_WEEKLY,
        PERIOD_END,
        PERIOD_START,
        SELIC_ANNUAL,
    )
    from prior import get_daily_returns_for_prior
    from weights import long_only_capped_weights
    from xgb_views import month_bounds

ROLLING_WINDOW = 60
MIN_OBS = 40
RIDGE = 1e-6
MODES = ("daily", "weekly", "monthly")


def _mode_period_key(dates: pd.Series, mode: str) -> pd.Series:
    if mode == "daily":
        return dates.dt.strftime("%Y-%m-%d")
    if mode == "weekly":
        return dates.dt.to_period("W-FRI").astype(str)
    if mode == "monthly":
        return dates.dt.to_period("M").astype(str)
    raise ValueError(f"Modo invalido: {mode}")


def load_returns_matrix() -> pd.DataFrame:
    """Log-retornos diários reais de ``dados_diarios/`` (não usar ``y_real`` do XGB)."""
    start_date, end_date = month_bounds(PERIOD_START, PERIOD_END)
    returns = get_daily_returns_for_prior(
        tickers=ATIVOS,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    returns.index = pd.to_datetime(returns.index)
    if returns.empty:
        raise ValueError("Nao foi possivel montar matriz de retornos diarios para Markowitz.")
    return returns.sort_index()


def compute_markowitz_curve(returns_mat: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = returns_mat.index.to_series().reset_index(drop=True)
    assets = list(returns_mat.columns)
    n = len(assets)

    reb_key = _mode_period_key(dates, mode)
    rebalance_flag = ~reb_key.duplicated()

    weights = np.zeros((len(dates), n), dtype=float)
    current_w = np.full(n, 1.0 / n, dtype=float)

    for i in range(len(dates)):
        if bool(rebalance_flag.iloc[i]):
            hist = returns_mat.iloc[max(0, i - ROLLING_WINDOW):i].copy()
            hist = hist.dropna(how="any")
            if len(hist) >= MIN_OBS:
                mu = hist.mean().to_numpy(dtype=float)
                sigma = hist.cov().to_numpy(dtype=float)
                sigma = sigma + RIDGE * np.eye(n)
                raw_w = np.linalg.pinv(sigma) @ mu
                if np.isfinite(raw_w).all() and abs(float(raw_w.sum())) > 1e-12:
                    raw_w = raw_w / float(raw_w.sum())
                current_w = long_only_capped_weights(raw_w, MAX_WEIGHT_PER_ASSET)
        weights[i] = current_w

    w_exec = np.roll(weights, shift=EXECUTION_LAG_DAYS, axis=0)
    w_exec[:EXECUTION_LAG_DAYS, :] = np.nan
    ret_log = returns_mat.to_numpy(dtype=float)
    ret_simple = np.expm1(ret_log)
    ret_port = np.nansum(w_exec * ret_simple, axis=1)
    ret_port[:EXECUTION_LAG_DAYS] = np.nan

    out = pd.DataFrame(
        {
            "view_date": dates.values,
            "mode": mode,
            "ret_mkv": ret_port,
        }
    )
    out = out.dropna(subset=["ret_mkv"]).reset_index(drop=True)
    out["capital_mkv"] = (1.0 + out["ret_mkv"]).cumprod()
    out["capital_mkv_brl"] = INITIAL_CAPITAL_BRL * out["capital_mkv"]
    out["ganho_mkv_pct"] = (out["capital_mkv"] - 1.0) * 100.0

    # Tabela de pesos (alvo e executado) por data.
    weights_df = pd.DataFrame({"view_date": dates.values, "mode": mode})
    for j, a in enumerate(assets):
        weights_df[f"{a}_target"] = weights[:, j]
        weights_df[f"{a}_exec"] = w_exec[:, j]
    weights_df["rebalance_flag"] = rebalance_flag.astype(int).values
    weights_df["gross_target"] = np.abs(weights).sum(axis=1)
    weights_df["gross_exec"] = np.nansum(np.abs(w_exec), axis=1)

    # Alinha com out (remove primeiras linhas sem execucao).
    valid_dates = set(pd.to_datetime(out["view_date"]))
    weights_df = weights_df[weights_df["view_date"].isin(valid_dates)].reset_index(drop=True)
    return out, weights_df


def load_bl_curve(mode: str) -> pd.DataFrame:
    path_map = {
        "daily": OUT_GAIN_SERIES_DAILY,
        "weekly": OUT_GAIN_SERIES_WEEKLY,
        "monthly": OUT_GAIN_SERIES_MONTHLY,
    }
    path = path_map[mode]
    if not path.exists():
        raise FileNotFoundError(f"Arquivo BL nao encontrado para modo {mode}: {path}")
    df = pd.read_csv(path)
    df["view_date"] = pd.to_datetime(df["view_date"], errors="coerce")
    df = df.dropna(subset=["view_date"]).sort_values("view_date").reset_index(drop=True)
    needed = ["view_date", "capital_real_brl", "capital_selic_brl", "ganho_real_acum_pct", "ganho_selic_acum_pct"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Coluna ausente em {path.name}: {c}")
    out = df[needed].copy()
    out["mode"] = mode
    out = out.rename(
        columns={
            "capital_real_brl": "capital_bl_real_brl",
            "capital_selic_brl": "capital_selic_brl",
            "ganho_real_acum_pct": "ganho_bl_real_pct",
            "ganho_selic_acum_pct": "ganho_selic_pct",
        }
    )
    return out


def summarize(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, g in comp.groupby("mode", sort=False):
        g = g.sort_values("view_date")
        if g.empty:
            continue
        ret_mkv = pd.to_numeric(g["ret_mkv"], errors="coerce").dropna()
        ret_bl = pd.to_numeric(g["ret_bl_real"], errors="coerce").dropna()

        def _sharpe(ret: pd.Series) -> float:
            if ret.empty:
                return np.nan
            sd = float(ret.std(ddof=1))
            if sd <= 0:
                return np.nan
            return float(ret.mean() / sd * np.sqrt(252))

        def _mdd(cap: pd.Series) -> float:
            cap = pd.to_numeric(cap, errors="coerce").dropna()
            if cap.empty:
                return np.nan
            running_max = cap.cummax()
            dd = cap / running_max - 1.0
            return float(dd.min())

        rows.append(
            {
                "mode": mode,
                "final_bl_real_brl": float(g["capital_bl_real_brl"].iloc[-1]),
                "final_markowitz_brl": float(g["capital_mkv_brl"].iloc[-1]),
                "final_selic_brl": float(g["capital_selic_brl"].iloc[-1]),
                "sharpe_bl_real": _sharpe(ret_bl),
                "sharpe_markowitz": _sharpe(ret_mkv),
                "mdd_bl_real": _mdd(g["capital_bl_real_brl"]),
                "mdd_markowitz": _mdd(g["capital_mkv_brl"]),
                "dias": int(len(g)),
            }
        )
    return pd.DataFrame(rows).sort_values("mode")


def plot_comparison(comp: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(13, 7))
    for mode, g in comp.groupby("mode", sort=False):
        g = g.sort_values("view_date")
        plt.plot(g["view_date"], g["capital_bl_real_brl"], label=f"BL realizado - {mode}", linewidth=1.8)
        plt.plot(g["view_date"], g["capital_mkv_brl"], label=f"Markowitz puro - {mode}", linewidth=1.8, linestyle="--")

    # Selic unica
    ref = comp.sort_values("view_date").drop_duplicates(subset=["view_date"], keep="first")
    plt.plot(ref["view_date"], ref["capital_selic_brl"], label="Selic 15% a.a.", linewidth=2.0, linestyle=":")
    plt.axhline(INITIAL_CAPITAL_BRL, color="black", linewidth=0.8, alpha=0.5)
    plt.title("Comparativo: BL realizado vs Markowitz puro")
    plt.xlabel("Data")
    plt.ylabel("Capital (R$)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def main() -> None:
    returns_mat = load_returns_matrix()

    parts = []
    weights_parts = []
    for mode in MODES:
        mkv, w_mkv = compute_markowitz_curve(returns_mat, mode=mode)
        bl = load_bl_curve(mode=mode)
        merged = bl.merge(mkv, on=["view_date", "mode"], how="inner")
        # retorno BL realizado para comparacoes de risco
        merged["ret_bl_real"] = pd.to_numeric(merged["capital_bl_real_brl"], errors="coerce").pct_change()
        parts.append(merged)
        weights_parts.append(w_mkv)

    comp = pd.concat(parts, axis=0, ignore_index=True).sort_values(["mode", "view_date"]).reset_index(drop=True)
    comp_out = comp.copy()
    comp_out["view_date"] = pd.to_datetime(comp_out["view_date"]).dt.strftime("%Y-%m-%d")
    comp_out.to_csv(OUT_MKZ_COMP, index=False)

    summary = summarize(comp)
    summary.to_csv(OUT_MKZ_SUMMARY, index=False)
    plot_comparison(comp, OUT_MKZ_PLOT)

    w_all = pd.concat(weights_parts, axis=0, ignore_index=True).sort_values(["mode", "view_date"]).reset_index(drop=True)
    w_daily = w_all[w_all["mode"] == "daily"].copy()
    w_weekly = w_all[w_all["mode"] == "weekly"].copy()
    w_monthly = w_all[w_all["mode"] == "monthly"].copy()
    for df, path in (
        (w_daily, OUT_MKZ_W_DAILY),
        (w_weekly, OUT_MKZ_W_WEEKLY),
        (w_monthly, OUT_MKZ_W_MONTHLY),
    ):
        out_df = df.copy()
        out_df["view_date"] = pd.to_datetime(out_df["view_date"]).dt.strftime("%Y-%m-%d")
        out_df.to_csv(path, index=False)

    print(f"Comparativo salvo em: {OUT_MKZ_COMP}")
    print(f"Resumo salvo em: {OUT_MKZ_SUMMARY}")
    print(f"Gráfico salvo em: {OUT_MKZ_PLOT}")
    print(f"Pesos Markowitz daily: {OUT_MKZ_W_DAILY}")
    print(f"Pesos Markowitz weekly: {OUT_MKZ_W_WEEKLY}")
    print(f"Pesos Markowitz monthly: {OUT_MKZ_W_MONTHLY}")


if __name__ == "__main__":
    main()
