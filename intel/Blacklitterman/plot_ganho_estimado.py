"""Plota comparativo de ganho estimado por frequência de rebalanceamento."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from .config import (
        EXECUTION_LAG_DAYS,
        INITIAL_CAPITAL_BRL,
        OUT_GAIN_COMPARATIVE,
        OUT_GAIN_PLOT,
        OUT_GAIN_SERIES_DAILY,
        OUT_GAIN_SERIES_MONTHLY,
        OUT_GAIN_SERIES_WEEKLY,
        OUT_POSTERIOR_MU,
        OUT_POSTERIOR_W_CONTROLLED,
        OUT_POSTERIOR_W_CTRL_DAILY,
        OUT_POSTERIOR_W_CTRL_MONTHLY,
        OUT_POSTERIOR_W_CTRL_WEEKLY,
        SELIC_ANNUAL,
    )
    from .io_utils import asset_columns, save_csv_with_date
    from .prior import get_daily_returns_for_prior
except ImportError:
    from config import (
        EXECUTION_LAG_DAYS,
        INITIAL_CAPITAL_BRL,
        OUT_GAIN_COMPARATIVE,
        OUT_GAIN_PLOT,
        OUT_GAIN_SERIES_DAILY,
        OUT_GAIN_SERIES_MONTHLY,
        OUT_GAIN_SERIES_WEEKLY,
        OUT_POSTERIOR_MU,
        OUT_POSTERIOR_W_CONTROLLED,
        OUT_POSTERIOR_W_CTRL_DAILY,
        OUT_POSTERIOR_W_CTRL_MONTHLY,
        OUT_POSTERIOR_W_CTRL_WEEKLY,
        SELIC_ANNUAL,
    )
    from io_utils import asset_columns, save_csv_with_date
    from prior import get_daily_returns_for_prior


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    mu = pd.read_csv(OUT_POSTERIOR_MU)
    weights_by_mode: dict[str, pd.DataFrame] = {}

    if OUT_POSTERIOR_W_CTRL_DAILY.exists() and OUT_POSTERIOR_W_CTRL_WEEKLY.exists() and OUT_POSTERIOR_W_CTRL_MONTHLY.exists():
        weights_by_mode["daily"] = pd.read_csv(OUT_POSTERIOR_W_CTRL_DAILY)
        weights_by_mode["weekly"] = pd.read_csv(OUT_POSTERIOR_W_CTRL_WEEKLY)
        weights_by_mode["monthly"] = pd.read_csv(OUT_POSTERIOR_W_CTRL_MONTHLY)
    elif OUT_POSTERIOR_W_CONTROLLED.exists():
        weights_by_mode["daily"] = pd.read_csv(OUT_POSTERIOR_W_CONTROLLED)
    else:
        raise FileNotFoundError("Pesos controlados não encontrados. Execute pipeline.py primeiro.")

    mu["view_date"] = pd.to_datetime(mu["view_date"], errors="coerce")
    mu = mu.dropna(subset=["view_date"]).sort_values("view_date").reset_index(drop=True)

    for mode, w_df in list(weights_by_mode.items()):
        w_df["view_date"] = pd.to_datetime(w_df["view_date"], errors="coerce")
        weights_by_mode[mode] = w_df.dropna(subset=["view_date"]).sort_values("view_date").reset_index(drop=True)

    assets = sorted(set(asset_columns(mu)))
    for w_df in weights_by_mode.values():
        assets = sorted(set(assets).intersection(asset_columns(w_df)))
    if not assets:
        raise ValueError("Não encontrei colunas de ativos (.SA) em posterior_mu e pesos.")

    start = mu["view_date"].min().strftime("%Y-%m-%d")
    end = mu["view_date"].max().strftime("%Y-%m-%d")
    daily_log_returns = get_daily_returns_for_prior(assets, start_date=start, end_date=end)
    daily_log_returns.index = pd.to_datetime(daily_log_returns.index)

    return mu, weights_by_mode, daily_log_returns


def build_gain_series(
    mu: pd.DataFrame,
    w: pd.DataFrame,
    daily_log_returns: pd.DataFrame,
    mode_label: str,
) -> pd.DataFrame:
    assets = sorted(set(asset_columns(mu)).intersection(asset_columns(w)))
    if not assets:
        raise ValueError("Não encontrei colunas de ativos (.SA) em posterior_mu e posterior_weights.")

    merged = mu[["view_date", *assets]].merge(
        w[["view_date", *assets]],
        on="view_date",
        suffixes=("_mu", "_w"),
        how="inner",
    )

    for asset in assets:
        merged[f"{asset}_w_exec"] = merged[f"{asset}_w"].shift(EXECUTION_LAG_DAYS)
    merged = merged.dropna(subset=[f"{a}_w_exec" for a in assets]).reset_index(drop=True)

    # μ posterior e retornos diários estão em log-retorno; converte para simples no P&L.
    est_parts = [merged[f"{a}_w_exec"] * np.expm1(merged[f"{a}_mu"]) for a in assets]
    merged["ret_est_portfolio"] = np.sum(np.column_stack(est_parts), axis=1)

    real_assets = [a for a in assets if a in daily_log_returns.columns]
    if real_assets:
        for asset in real_assets:
            merged[f"{asset}_real_log"] = merged["view_date"].map(daily_log_returns[asset])
        real_parts = [
            merged[f"{a}_w_exec"] * np.expm1(merged[f"{a}_real_log"]) for a in real_assets
        ]
        merged["ret_real_portfolio"] = np.sum(np.column_stack(real_parts), axis=1)
    else:
        merged["ret_real_portfolio"] = np.nan

    merged["capital_est"] = (1.0 + merged["ret_est_portfolio"]).cumprod()
    merged["ganho_est_acum_pct"] = (merged["capital_est"] - 1.0) * 100.0

    if merged["ret_real_portfolio"].notna().any():
        merged["capital_real"] = (1.0 + merged["ret_real_portfolio"].fillna(0.0)).cumprod()
        merged["ganho_real_acum_pct"] = (merged["capital_real"] - 1.0) * 100.0
    else:
        merged["capital_real"] = np.nan
        merged["ganho_real_acum_pct"] = np.nan

    selic_daily = (1.0 + SELIC_ANNUAL) ** (1.0 / 252.0) - 1.0
    merged["ret_selic_benchmark"] = selic_daily
    merged["capital_selic"] = (1.0 + merged["ret_selic_benchmark"]).cumprod()
    merged["ganho_selic_acum_pct"] = (merged["capital_selic"] - 1.0) * 100.0
    merged["capital_est_brl"] = INITIAL_CAPITAL_BRL * merged["capital_est"]
    merged["capital_real_brl"] = INITIAL_CAPITAL_BRL * merged["capital_real"]
    merged["capital_selic_brl"] = INITIAL_CAPITAL_BRL * merged["capital_selic"]
    merged["mode"] = mode_label

    keep_cols = [
        "view_date",
        "ret_est_portfolio",
        "ret_real_portfolio",
        "ret_selic_benchmark",
        "capital_est",
        "capital_real",
        "capital_selic",
        "capital_est_brl",
        "capital_real_brl",
        "capital_selic_brl",
        "ganho_est_acum_pct",
        "ganho_real_acum_pct",
        "ganho_selic_acum_pct",
        "mode",
    ]
    return merged[keep_cols].sort_values("view_date").reset_index(drop=True)


def plot_gain_comparative(series_by_mode: dict[str, pd.DataFrame], output_path: Path) -> None:
    daily_ref = series_by_mode[sorted(series_by_mode.keys())[0]].copy()
    daily_ref["view_date"] = pd.to_datetime(daily_ref["view_date"], errors="coerce")

    plt.figure(figsize=(12, 6))
    for mode, df in series_by_mode.items():
        plot_df = df.copy()
        plot_df["view_date"] = pd.to_datetime(plot_df["view_date"], errors="coerce")
        plt.plot(plot_df["view_date"], plot_df["capital_est_brl"], label=f"Portfolio estimado - {mode}", linewidth=2)

    for mode, df in series_by_mode.items():
        if not df["capital_real_brl"].notna().any():
            continue
        d = df.copy()
        d["view_date"] = pd.to_datetime(d["view_date"], errors="coerce")
        plt.plot(
            d["view_date"],
            d["capital_real_brl"],
            label=f"Portfolio realizado - {mode}",
            linewidth=1.4,
            alpha=0.75,
            linestyle=":",
        )

    plt.plot(
        daily_ref["view_date"],
        daily_ref["capital_selic_brl"],
        label=f"Benchmark Selic ({SELIC_ANNUAL * 100:.0f}% a.a.)",
        linewidth=2,
        linestyle="--",
    )
    plt.axhline(INITIAL_CAPITAL_BRL, color="black", linewidth=0.8, alpha=0.5)
    plt.title("Black-Litterman Híbrido - Daily vs Weekly vs Monthly")
    plt.xlabel("Data")
    plt.ylabel("Capital (R$)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def main() -> None:
    mu, weights_by_mode, daily_log_returns = load_inputs()
    series_by_mode = {
        mode: build_gain_series(mu, w_df, daily_log_returns, mode_label=mode)
        for mode, w_df in weights_by_mode.items()
    }

    output_map = {
        "daily": OUT_GAIN_SERIES_DAILY,
        "weekly": OUT_GAIN_SERIES_WEEKLY,
        "monthly": OUT_GAIN_SERIES_MONTHLY,
    }
    for mode, path in output_map.items():
        if mode in series_by_mode:
            save_csv_with_date(series_by_mode[mode], path)
            print(f"Série {mode} salva em: {path}")

    comparative = None
    for mode, df in series_by_mode.items():
        part = df[
            ["view_date", "capital_est_brl", "ganho_est_acum_pct", "capital_real_brl", "ganho_real_acum_pct"]
        ].copy()
        part = part.rename(
            columns={
                "capital_est_brl": f"capital_{mode}_brl",
                "ganho_est_acum_pct": f"ganho_{mode}_pct",
                "capital_real_brl": f"capital_real_{mode}_brl",
                "ganho_real_acum_pct": f"ganho_real_{mode}_pct",
            }
        )
        comparative = part if comparative is None else comparative.merge(part, on="view_date", how="outer")

    if comparative is not None:
        base = series_by_mode[sorted(series_by_mode.keys())[0]][
            ["view_date", "capital_selic_brl", "ganho_selic_acum_pct"]
        ]
        comparative = comparative.merge(base, on="view_date", how="left").sort_values("view_date")
        save_csv_with_date(comparative, OUT_GAIN_COMPARATIVE)
        print(f"Comparativo salvo em: {OUT_GAIN_COMPARATIVE}")

    plot_gain_comparative(series_by_mode, OUT_GAIN_PLOT)
    print(f"Gráfico salvo em: {OUT_GAIN_PLOT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
