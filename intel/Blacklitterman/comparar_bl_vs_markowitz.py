"""
Compara Black-Litterman hibrido vs baselines (Markowitz, equal-weight, prior-only, Selic).

Saidas:
- bl_vs_markowitz_comparativo.csv
- bl_vs_markowitz_resumo.csv
- bl_vs_markowitz_plot.png
- markowitz_weights_*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .backtest import apply_turnover_cost, build_gain_series
    from .config import (
        ATIVOS,
        EXECUTION_LAG_DAYS,
        INITIAL_CAPITAL_BRL,
        MAX_WEIGHT_PER_ASSET,
        OUT_MKZ_COMP,
        OUT_MKZ_COST_COMPARE,
        OUT_MKZ_PLOT,
        OUT_MKZ_SUMMARY,
        OUT_MKZ_W_DAILY,
        OUT_MKZ_W_MONTHLY,
        OUT_MKZ_W_WEEKLY,
        OUT_POSTERIOR_MU,
        OUT_POSTERIOR_W_CTRL_DAILY,
        OUT_POSTERIOR_W_CTRL_MONTHLY,
        OUT_POSTERIOR_W_CTRL_WEEKLY,
        PERIOD_END,
        PERIOD_START,
        PRIOR_END_DATE,
        PRIOR_START_DATE,
        RISK_FREE_ANNUAL,
        SELIC_ANNUAL,
        TAU,
        TRANSACTION_COST_BPS,
    )
    from .prior import (
        compute_pi_and_prior_uncertainty,
        get_daily_returns_for_prior,
        get_market_cap_weights,
    )
    from .weights import long_only_capped_weights
    from .xgb_views import month_bounds
except ImportError:
    from backtest import apply_turnover_cost, build_gain_series
    from config import (
        ATIVOS,
        EXECUTION_LAG_DAYS,
        INITIAL_CAPITAL_BRL,
        MAX_WEIGHT_PER_ASSET,
        OUT_MKZ_COMP,
        OUT_MKZ_COST_COMPARE,
        OUT_MKZ_PLOT,
        OUT_MKZ_SUMMARY,
        OUT_MKZ_W_DAILY,
        OUT_MKZ_W_MONTHLY,
        OUT_MKZ_W_WEEKLY,
        OUT_POSTERIOR_MU,
        OUT_POSTERIOR_W_CTRL_DAILY,
        OUT_POSTERIOR_W_CTRL_MONTHLY,
        OUT_POSTERIOR_W_CTRL_WEEKLY,
        PERIOD_END,
        PERIOD_START,
        PRIOR_END_DATE,
        PRIOR_START_DATE,
        RISK_FREE_ANNUAL,
        SELIC_ANNUAL,
        TAU,
        TRANSACTION_COST_BPS,
    )
    from prior import (
        compute_pi_and_prior_uncertainty,
        get_daily_returns_for_prior,
        get_market_cap_weights,
    )
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


def _curve_from_weights(
    returns_mat: pd.DataFrame,
    weights: np.ndarray,
    mode: str,
    ret_col: str,
    cap_col: str,
    ganho_col: str,
    *,
    cost_bps: float,
) -> pd.DataFrame:
    dates = returns_mat.index.to_series().reset_index(drop=True)
    w_exec = np.roll(weights, shift=EXECUTION_LAG_DAYS, axis=0)
    w_exec[:EXECUTION_LAG_DAYS, :] = np.nan
    ret_simple = np.expm1(returns_mat.to_numpy(dtype=float))
    ret_gross = np.nansum(w_exec * ret_simple, axis=1)
    ret_gross[:EXECUTION_LAG_DAYS] = np.nan

    valid = np.isfinite(ret_gross)
    ret_g = ret_gross[valid]
    w_v = np.nan_to_num(w_exec[valid], nan=0.0)
    ret_net, _cost = apply_turnover_cost(w_v, ret_g, cost_bps=cost_bps)

    out = pd.DataFrame(
        {
            "view_date": dates.values[valid],
            "mode": mode,
            ret_col: ret_net,
        }
    )
    out[cap_col.replace("_brl", "")] = (1.0 + out[ret_col]).cumprod()
    out[cap_col] = INITIAL_CAPITAL_BRL * out[cap_col.replace("_brl", "")]
    out[ganho_col] = (out[cap_col] / INITIAL_CAPITAL_BRL - 1.0) * 100.0
    return out.reset_index(drop=True)


def compute_markowitz_curve(
    returns_mat: pd.DataFrame,
    mode: str,
    *,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    out = _curve_from_weights(
        returns_mat,
        weights,
        mode,
        "ret_mkv",
        "capital_mkv_brl",
        "ganho_mkv_pct",
        cost_bps=cost_bps,
    )

    w_exec = np.roll(weights, shift=EXECUTION_LAG_DAYS, axis=0)
    w_exec[:EXECUTION_LAG_DAYS, :] = np.nan
    weights_df = pd.DataFrame({"view_date": dates.values, "mode": mode})
    for j, a in enumerate(assets):
        weights_df[f"{a}_target"] = weights[:, j]
        weights_df[f"{a}_exec"] = w_exec[:, j]
    weights_df["rebalance_flag"] = rebalance_flag.astype(int).values
    weights_df["gross_target"] = np.abs(weights).sum(axis=1)
    weights_df["gross_exec"] = np.nansum(np.abs(w_exec), axis=1)
    valid_dates = set(pd.to_datetime(out["view_date"]))
    weights_df = weights_df[weights_df["view_date"].isin(valid_dates)].reset_index(drop=True)
    return out, weights_df


def compute_equal_weight_curve(
    returns_mat: pd.DataFrame,
    mode: str,
    *,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    n = returns_mat.shape[1]
    dates = returns_mat.index.to_series().reset_index(drop=True)
    reb_key = _mode_period_key(dates, mode)
    rebalance_flag = ~reb_key.duplicated()
    target = np.full(n, 1.0 / n, dtype=float)
    weights = np.zeros((len(dates), n), dtype=float)
    current = target.copy()
    for i in range(len(dates)):
        if bool(rebalance_flag.iloc[i]):
            current = target.copy()
        weights[i] = current
    return _curve_from_weights(
        returns_mat,
        weights,
        mode,
        "ret_ew",
        "capital_ew_brl",
        "ganho_ew_pct",
        cost_bps=cost_bps,
    )


def compute_prior_only_curve(
    returns_mat: pd.DataFrame,
    mode: str,
    *,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Pesos de equilibrio w ∝ Σ⁻¹π (sem views), long-only com teto."""
    assets = list(returns_mat.columns)
    market_weights = get_market_cap_weights(ATIVOS).reindex(assets).fillna(0.0)
    if float(market_weights.sum()) <= 0:
        market_weights = pd.Series(1.0 / len(assets), index=assets)
    else:
        market_weights = market_weights / float(market_weights.sum())

    hist = get_daily_returns_for_prior(
        tickers=ATIVOS,
        start_date=PRIOR_START_DATE,
        end_date=PRIOR_END_DATE,
    )
    hist = hist[[c for c in assets if c in hist.columns]].dropna(how="any")
    pi, sigma, _, delta = compute_pi_and_prior_uncertainty(
        returns=hist,
        market_weights=market_weights.reindex(hist.columns),
        risk_free_annual=RISK_FREE_ANNUAL,
        tau=TAU,
    )
    pi = pi.reindex(assets).fillna(0.0)
    sigma = sigma.reindex(index=assets, columns=assets).fillna(0.0)
    raw = np.linalg.pinv(float(delta) * sigma.to_numpy(dtype=float)) @ pi.to_numpy(dtype=float)
    if not np.isfinite(raw).all() or abs(float(raw.sum())) <= 1e-12:
        target = np.full(len(assets), 1.0 / len(assets), dtype=float)
    else:
        target = long_only_capped_weights(raw / float(raw.sum()), MAX_WEIGHT_PER_ASSET)

    dates = returns_mat.index.to_series().reset_index(drop=True)
    reb_key = _mode_period_key(dates, mode)
    rebalance_flag = ~reb_key.duplicated()
    weights = np.zeros((len(dates), len(assets)), dtype=float)
    current = target.copy()
    for i in range(len(dates)):
        if bool(rebalance_flag.iloc[i]):
            current = target.copy()
        weights[i] = current
    return _curve_from_weights(
        returns_mat,
        weights,
        mode,
        "ret_prior",
        "capital_prior_brl",
        "ganho_prior_pct",
        cost_bps=cost_bps,
    )


def load_bl_curve(
    mode: str,
    returns_mat: pd.DataFrame,
    *,
    cost_bps: float = TRANSACTION_COST_BPS,
) -> pd.DataFrame:
    """Reconstrói curva BL com o custo pedido (não reusa CSV de ganho já com bps fixo)."""
    w_map = {
        "daily": OUT_POSTERIOR_W_CTRL_DAILY,
        "weekly": OUT_POSTERIOR_W_CTRL_WEEKLY,
        "monthly": OUT_POSTERIOR_W_CTRL_MONTHLY,
    }
    w_path = w_map[mode]
    if not OUT_POSTERIOR_MU.exists() or not w_path.exists():
        raise FileNotFoundError(
            f"Artefatos BL ausentes (mu={OUT_POSTERIOR_MU.exists()}, w={w_path.exists()})."
        )
    mu = pd.read_csv(OUT_POSTERIOR_MU)
    w = pd.read_csv(w_path)
    gain = build_gain_series(
        mu,
        w,
        returns_mat,
        mode_label=mode,
        transaction_cost_bps=cost_bps,
    )
    out = gain[
        ["view_date", "capital_real_brl", "capital_selic_brl", "ganho_real_acum_pct", "ganho_selic_acum_pct"]
    ].copy()
    out["mode"] = mode
    return out.rename(
        columns={
            "capital_real_brl": "capital_bl_real_brl",
            "ganho_real_acum_pct": "ganho_bl_real_pct",
            "ganho_selic_acum_pct": "ganho_selic_pct",
        }
    )


def summarize(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mode, g in comp.groupby("mode", sort=False):
        g = g.sort_values("view_date")
        if g.empty:
            continue

        def _sharpe(ret: pd.Series) -> float:
            ret = pd.to_numeric(ret, errors="coerce").dropna()
            if len(ret) < 2:
                return float("nan")
            sd = float(ret.std(ddof=1))
            if sd <= 0:
                return float("nan")
            return float(ret.mean() / sd * np.sqrt(252))

        def _mdd(cap: pd.Series) -> float:
            cap = pd.to_numeric(cap, errors="coerce").dropna()
            if cap.empty:
                return float("nan")
            running_max = cap.cummax()
            return float((cap / running_max - 1.0).min())

        row = {
            "mode": mode,
            "final_bl_real_brl": float(g["capital_bl_real_brl"].iloc[-1]),
            "final_markowitz_brl": float(g["capital_mkv_brl"].iloc[-1]),
            "final_ew_brl": float(g["capital_ew_brl"].iloc[-1]),
            "final_prior_brl": float(g["capital_prior_brl"].iloc[-1]),
            "final_selic_brl": float(g["capital_selic_brl"].iloc[-1]),
            "sharpe_bl_real": _sharpe(g["ret_bl_real"]),
            "sharpe_markowitz": _sharpe(g["ret_mkv"]),
            "sharpe_ew": _sharpe(g["ret_ew"]),
            "sharpe_prior": _sharpe(g["ret_prior"]),
            "mdd_bl_real": _mdd(g["capital_bl_real_brl"]),
            "mdd_markowitz": _mdd(g["capital_mkv_brl"]),
            "mdd_ew": _mdd(g["capital_ew_brl"]),
            "mdd_prior": _mdd(g["capital_prior_brl"]),
            "dias": int(len(g)),
            "execution_lag_days": EXECUTION_LAG_DAYS,
            "transaction_cost_bps": float(g["transaction_cost_bps"].iloc[0])
            if "transaction_cost_bps" in g.columns
            else TRANSACTION_COST_BPS,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mode")


def plot_comparison(comp: pd.DataFrame, output_path: Path, *, cost_bps: float) -> None:
    plt.figure(figsize=(13, 7))
    # Um modo por vez fica ilegível com 4 estratégias × 3 modos; plota weekly como foco + daily BL/Mkz
    focus_modes = ("weekly", "daily", "monthly")
    for mode in focus_modes:
        g = comp[comp["mode"] == mode].sort_values("view_date")
        if g.empty:
            continue
        ls = "-" if mode == "weekly" else (":" if mode == "daily" else "-.")
        lw = 2.0 if mode == "weekly" else 1.3
        plt.plot(g["view_date"], g["capital_bl_real_brl"], label=f"BL - {mode}", linewidth=lw, linestyle=ls)
        plt.plot(g["view_date"], g["capital_mkv_brl"], label=f"Markowitz - {mode}", linewidth=lw, linestyle=ls, alpha=0.85)
        if mode == "weekly":
            plt.plot(g["view_date"], g["capital_ew_brl"], label="Equal-weight - weekly", linewidth=1.8, alpha=0.9)
            plt.plot(g["view_date"], g["capital_prior_brl"], label="Prior-only (π) - weekly", linewidth=1.8, alpha=0.9)

    ref = comp.sort_values("view_date").drop_duplicates(subset=["view_date"], keep="first")
    plt.plot(ref["view_date"], ref["capital_selic_brl"], label=f"Selic {SELIC_ANNUAL*100:.0f}% a.a.", linewidth=2.0, linestyle="--")
    plt.axhline(INITIAL_CAPITAL_BRL, color="black", linewidth=0.8, alpha=0.5)
    plt.title(
        f"BL vs baselines (lag={EXECUTION_LAG_DAYS}d, custo={cost_bps:.0f} bps)"
    )
    plt.xlabel("Data")
    plt.ylabel("Capital (R$)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def run_comparison(
    *,
    cost_bps: float = TRANSACTION_COST_BPS,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    returns_mat = load_returns_matrix()

    parts = []
    weights_parts = []
    for mode in MODES:
        mkv, w_mkv = compute_markowitz_curve(returns_mat, mode=mode, cost_bps=cost_bps)
        ew = compute_equal_weight_curve(returns_mat, mode=mode, cost_bps=cost_bps)
        prior = compute_prior_only_curve(returns_mat, mode=mode, cost_bps=cost_bps)
        bl = load_bl_curve(mode=mode, returns_mat=returns_mat, cost_bps=cost_bps)
        merged = bl.merge(mkv, on=["view_date", "mode"], how="inner")
        merged = merged.merge(ew, on=["view_date", "mode"], how="inner")
        merged = merged.merge(prior, on=["view_date", "mode"], how="inner")
        merged["ret_bl_real"] = pd.to_numeric(merged["capital_bl_real_brl"], errors="coerce").pct_change()
        merged["transaction_cost_bps"] = cost_bps
        parts.append(merged)
        weights_parts.append(w_mkv)

    comp = pd.concat(parts, axis=0, ignore_index=True).sort_values(["mode", "view_date"]).reset_index(drop=True)
    summary = summarize(comp)

    if save_artifacts:
        comp_out = comp.copy()
        comp_out["view_date"] = pd.to_datetime(comp_out["view_date"]).dt.strftime("%Y-%m-%d")
        comp_out.to_csv(OUT_MKZ_COMP, index=False)
        summary.to_csv(OUT_MKZ_SUMMARY, index=False)
        plot_comparison(comp, OUT_MKZ_PLOT, cost_bps=cost_bps)

        w_all = pd.concat(weights_parts, axis=0, ignore_index=True).sort_values(["mode", "view_date"]).reset_index(drop=True)
        for mode, path in (
            ("daily", OUT_MKZ_W_DAILY),
            ("weekly", OUT_MKZ_W_WEEKLY),
            ("monthly", OUT_MKZ_W_MONTHLY),
        ):
            out_df = w_all[w_all["mode"] == mode].copy()
            out_df["view_date"] = pd.to_datetime(out_df["view_date"]).dt.strftime("%Y-%m-%d")
            out_df.to_csv(path, index=False)

        print(f"Comparativo salvo em: {OUT_MKZ_COMP}")
        print(f"Resumo salvo em: {OUT_MKZ_SUMMARY}")
        print(f"Gráfico salvo em: {OUT_MKZ_PLOT}")
        print(summary.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="BL vs Markowitz / EW / prior-only")
    parser.add_argument("--cost-bps", type=float, default=TRANSACTION_COST_BPS)
    parser.add_argument(
        "--also-zero-cost",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Também gera resumo com 0 bps para comparar drag de custo",
    )
    args = parser.parse_args()
    summary = run_comparison(cost_bps=args.cost_bps, save_artifacts=True)
    if args.also_zero_cost and abs(args.cost_bps) > 1e-12:
        zero = run_comparison(cost_bps=0.0, save_artifacts=False)
        cmp = summary.copy()
        cmp["cost_bps"] = args.cost_bps
        z = zero.copy()
        z["cost_bps"] = 0.0
        out = pd.concat([z, cmp], ignore_index=True)
        out.to_csv(OUT_MKZ_COST_COMPARE, index=False)
        print(f"Comparativo 0 vs {args.cost_bps:.0f} bps: {OUT_MKZ_COST_COMPARE}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
