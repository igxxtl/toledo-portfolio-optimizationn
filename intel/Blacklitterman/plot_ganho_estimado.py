"""
Plota comparativo de ganho estimado por frequencia de rebalanceamento.

Entradas:
- bl_hibrido_posterior_mu.csv
- bl_hibrido_posterior_weights_controlled_daily.csv
- bl_hibrido_posterior_weights_controlled_weekly.csv
- bl_hibrido_posterior_weights_controlled_monthly.csv
- bl_hibrido_q_long.csv (serie realizada opcional)

Saidas:
- bl_hibrido_ganho_series_daily.csv
- bl_hibrido_ganho_series_weekly.csv
- bl_hibrido_ganho_series_monthly.csv
- bl_hibrido_ganho_comparativo.csv
- bl_hibrido_ganho_estimado.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BL_DIR = Path(__file__).resolve().parent

MU_PATH = BL_DIR / "bl_hibrido_posterior_mu.csv"
W_DAILY_PATH = BL_DIR / "bl_hibrido_posterior_weights_controlled_daily.csv"
W_WEEKLY_PATH = BL_DIR / "bl_hibrido_posterior_weights_controlled_weekly.csv"
W_MONTHLY_PATH = BL_DIR / "bl_hibrido_posterior_weights_controlled_monthly.csv"
W_FALLBACK_PATH = BL_DIR / "bl_hibrido_posterior_weights_controlled.csv"
Q_LONG_PATH = BL_DIR / "bl_hibrido_q_long.csv"

OUT_SERIES_DAILY = BL_DIR / "bl_hibrido_ganho_series_daily.csv"
OUT_SERIES_WEEKLY = BL_DIR / "bl_hibrido_ganho_series_weekly.csv"
OUT_SERIES_MONTHLY = BL_DIR / "bl_hibrido_ganho_series_monthly.csv"
OUT_COMPARATIVE = BL_DIR / "bl_hibrido_ganho_comparativo.csv"
OUT_PLOT = BL_DIR / "bl_hibrido_ganho_estimado.png"

INITIAL_CAPITAL_BRL = 100_000.0
SELIC_ANNUAL = 0.15
EXECUTION_LAG_DAYS = 1


def _asset_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith(".SA")]


def load_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    mu = pd.read_csv(MU_PATH)
    q_long = pd.read_csv(Q_LONG_PATH)
    weights_by_mode: dict[str, pd.DataFrame] = {}

    if W_DAILY_PATH.exists() and W_WEEKLY_PATH.exists() and W_MONTHLY_PATH.exists():
        weights_by_mode["daily"] = pd.read_csv(W_DAILY_PATH)
        weights_by_mode["weekly"] = pd.read_csv(W_WEEKLY_PATH)
        weights_by_mode["monthly"] = pd.read_csv(W_MONTHLY_PATH)
    elif W_FALLBACK_PATH.exists():
        # Compatibilidade: se ainda nao gerou os 3 modos, usa o controlado unico como daily.
        weights_by_mode["daily"] = pd.read_csv(W_FALLBACK_PATH)
    else:
        raise FileNotFoundError(
            "Nao encontrei arquivos de pesos controlados. Rode o Blacklitterman-hibrido.py primeiro."
        )

    mu["view_date"] = pd.to_datetime(mu["view_date"], errors="coerce")
    q_long["view_date"] = pd.to_datetime(q_long["view_date"], errors="coerce")

    mu = mu.dropna(subset=["view_date"]).sort_values("view_date").reset_index(drop=True)
    q_long = q_long.dropna(subset=["view_date"]).sort_values("view_date").reset_index(drop=True)
    for mode, w_df in list(weights_by_mode.items()):
        w_df["view_date"] = pd.to_datetime(w_df["view_date"], errors="coerce")
        weights_by_mode[mode] = w_df.dropna(subset=["view_date"]).sort_values("view_date").reset_index(drop=True)

    return mu, weights_by_mode, q_long


def build_gain_series(mu: pd.DataFrame, w: pd.DataFrame, q_long: pd.DataFrame, mode_label: str) -> pd.DataFrame:
    assets = sorted(set(_asset_columns(mu)).intersection(_asset_columns(w)))
    if not assets:
        raise ValueError("Nao encontrei colunas de ativos (.SA) em posterior_mu e posterior_weights.")

    merged = mu[["view_date", *assets]].merge(
        w[["view_date", *assets]],
        on="view_date",
        suffixes=("_mu", "_w"),
        how="inner",
    )

    # Backtest sem look-ahead:
    # pesos de t-1 executam no retorno de t.
    for a in assets:
        merged[f"{a}_w_exec"] = merged[f"{a}_w"].shift(EXECUTION_LAG_DAYS)
    merged = merged.dropna(subset=[f"{a}_w_exec" for a in assets]).reset_index(drop=True)

    # Retorno estimado do portfolio no dia t (com pesos executados):
    # r_est_t = sum_i (w_exec_i,t * mu_i,t)
    est_parts = []
    for a in assets:
        est_parts.append(merged[f"{a}_mu"] * merged[f"{a}_w_exec"])
    merged["ret_est_portfolio"] = np.sum(np.column_stack(est_parts), axis=1)

    # Serie realizada (opcional) usando os mesmos pesos no retorno real de cada ativo.
    q_real = q_long.copy()
    q_real["ticker_sa"] = q_real["ticker"].astype(str).str.upper() + ".SA"
    real_pivot = q_real.pivot(index="view_date", columns="ticker_sa", values="ret_7h_real")
    real_assets = [a for a in assets if a in real_pivot.columns]
    if real_assets:
        for a in real_assets:
            merged[f"{a}_real"] = merged["view_date"].map(real_pivot[a])
        real_parts = []
        for a in real_assets:
            real_parts.append(merged[f"{a}_w_exec"] * merged[f"{a}_real"])
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

    # Benchmark Selic: converte 15% a.a. para taxa diaria util e capitaliza por linha.
    selic_daily = (1.0 + SELIC_ANNUAL) ** (1.0 / 252.0) - 1.0
    merged["ret_selic_benchmark"] = selic_daily
    merged["capital_selic"] = (1.0 + merged["ret_selic_benchmark"]).cumprod()
    merged["ganho_selic_acum_pct"] = (merged["capital_selic"] - 1.0) * 100.0

    # Converte capitais normalizados para valor monetario (base R$ 100.000).
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
        plt.plot(
            plot_df["view_date"],
            plot_df["capital_est_brl"],
            label=f"Portfolio estimado - {mode}",
            linewidth=2,
        )

    # Plota as series realizadas por modo.
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
    plt.title("Black-Litterman Hibrido - Daily vs Weekly vs Monthly")
    plt.xlabel("Data")
    plt.ylabel("Capital (R$)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def _save_series_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    out["view_date"] = pd.to_datetime(out["view_date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)


def main() -> None:
    mu, weights_by_mode, q_long = load_inputs()
    series_by_mode: dict[str, pd.DataFrame] = {}
    for mode, w_df in weights_by_mode.items():
        series_by_mode[mode] = build_gain_series(mu, w_df, q_long, mode_label=mode)

    if "daily" in series_by_mode:
        _save_series_csv(series_by_mode["daily"], OUT_SERIES_DAILY)
        print(f"Serie daily salva em: {OUT_SERIES_DAILY}")
    if "weekly" in series_by_mode:
        _save_series_csv(series_by_mode["weekly"], OUT_SERIES_WEEKLY)
        print(f"Serie weekly salva em: {OUT_SERIES_WEEKLY}")
    if "monthly" in series_by_mode:
        _save_series_csv(series_by_mode["monthly"], OUT_SERIES_MONTHLY)
        print(f"Serie monthly salva em: {OUT_SERIES_MONTHLY}")

    comparative = None
    for mode, df in series_by_mode.items():
        part = df[["view_date", "capital_est_brl", "ganho_est_acum_pct", "capital_real_brl", "ganho_real_acum_pct"]].copy()
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
        base = series_by_mode[sorted(series_by_mode.keys())[0]][["view_date", "capital_selic_brl", "ganho_selic_acum_pct"]]
        comparative = comparative.merge(base, on="view_date", how="left").sort_values("view_date")
        comp_out = comparative.copy()
        comp_out["view_date"] = pd.to_datetime(comp_out["view_date"]).dt.strftime("%Y-%m-%d")
        comp_out.to_csv(OUT_COMPARATIVE, index=False)
        print(f"Comparativo salvo em: {OUT_COMPARATIVE}")

    plot_gain_comparative(series_by_mode, OUT_PLOT)
    print(f"Grafico salvo em: {OUT_PLOT}")
    for mode, df in series_by_mode.items():
        if df.empty:
            continue
        last = df.tail(1).iloc[0]
        print(
            f"Ultimo ponto ({mode}) | Estimado={float(last['capital_est_brl']):,.2f} "
            f"({float(last['ganho_est_acum_pct']):.2f}%) | "
            f"Realizado={float(last['capital_real_brl']):,.2f} ({float(last['ganho_real_acum_pct']):.2f}%)"
        )


if __name__ == "__main__":
    main()
