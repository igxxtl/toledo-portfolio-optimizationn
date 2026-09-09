"""Calcula features quantitativas relativas ao Ibovespa em ``dados_diarios/``."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import DATA_DAILY_DIR, FEATURE_WINDOWS, IBOV_FILE
except ImportError:
    from config import DATA_DAILY_DIR, FEATURE_WINDOWS, IBOV_FILE


def log_ret(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))


def rolling_beta(r_a: pd.Series, r_m: pd.Series, n: int) -> pd.Series:
    """Beta movel: Cov(R_A, R_M, N) / Var(R_M, N)."""
    cov = r_a.rolling(n, min_periods=n).cov(r_m)
    var = r_m.rolling(n, min_periods=n).var(ddof=1)
    return cov / var.replace(0, np.nan)


def calcular_features(df: pd.DataFrame, ibov: pd.DataFrame, n: int) -> pd.DataFrame:
    """Calcula todas as features para a janela N e adiciona colunas ao DataFrame."""
    p_a = df["close"]
    p_m = ibov["close"]
    v_a = df["volume"].astype(float)
    v_m = ibov["volume"].astype(float)

    r_a = log_ret(p_a)
    r_m = log_ret(p_m)

    sfx = f"_{n}d"

    if "log_ret" not in df.columns:
        df["log_ret"] = r_a.values
        df["ibov_log_ret"] = r_m.values

    df[f"spread_retorno{sfx}"] = np.log(p_a / p_a.shift(n)) - np.log(p_m / p_m.shift(n))

    sma_a = p_a.rolling(n, min_periods=n).mean()
    sma_m = p_m.rolling(n, min_periods=n).mean()
    df[f"dist_sma{sfx}"] = (
        (p_a / sma_a.replace(0, np.nan) - 1.0) - (p_m / sma_m.replace(0, np.nan) - 1.0)
    )

    df[f"spread_vol{sfx}"] = (
        r_a.rolling(n, min_periods=n).std(ddof=1) - r_m.rolling(n, min_periods=n).std(ddof=1)
    )
    df[f"beta{sfx}"] = rolling_beta(r_a, r_m, n)

    mu_va = v_a.rolling(n, min_periods=n).mean()
    mu_vm = v_m.rolling(n, min_periods=n).mean()
    df[f"spread_volratio{sfx}"] = v_a / mu_va.replace(0, np.nan) - v_m / mu_vm.replace(0, np.nan)

    alpha = r_a - r_m
    mu_alpha = alpha.rolling(n, min_periods=n).mean()
    std_alpha = alpha.rolling(n, min_periods=n).std(ddof=1)
    df[f"z_alpha{sfx}"] = (alpha - mu_alpha) / std_alpha.replace(0, np.nan)

    df[f"trend_mercado{sfx}"] = np.log(p_m / p_m.shift(n))
    df[f"risco_mercado{sfx}"] = r_m.rolling(n, min_periods=n).std(ddof=1)

    mu_pm = p_m.rolling(n, min_periods=n).mean()
    std_pm = p_m.rolling(n, min_periods=n).std(ddof=1)
    df[f"z_mercado{sfx}"] = (p_m - mu_pm) / std_pm.replace(0, np.nan)

    return df


def processar_ticker(fp: Path, ibov: pd.DataFrame) -> None:
    ticker = fp.stem
    print(f"  [{ticker}] calculando features ...", end=" ", flush=True)

    df = pd.read_csv(fp, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    ibov_alinhado = (
        df[["date"]]
        .merge(ibov[["date", "close", "volume"]], on="date", how="left")
        .set_index("date")
    )
    ibov_alinhado = ibov_alinhado.ffill().reset_index()

    df["ibov_close"] = ibov_alinhado["close"].values

    feature_prefixes = (
        "spread_retorno",
        "dist_sma",
        "spread_vol",
        "beta",
        "spread_volratio",
        "z_alpha",
        "trend_mercado",
        "risco_mercado",
        "z_mercado",
    )
    df = df.drop(
        columns=[c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)],
        errors="ignore",
    )

    for n in FEATURE_WINDOWS:
        df = calcular_features(df, ibov_alinhado, n)

    df.to_csv(fp, index=False)

    n_features = len(FEATURE_WINDOWS) * 9
    print(f"ok - {len(df)} dias | {n_features} features ({len(FEATURE_WINDOWS)} janela(s))")


def main() -> None:
    if not IBOV_FILE.exists():
        print(f"Arquivo do Ibovespa nao encontrado: {IBOV_FILE}")
        print("Rode download_daily.py primeiro.")
        sys.exit(1)

    ibov = pd.read_csv(IBOV_FILE, parse_dates=["date"])
    ibov = ibov.sort_values("date").reset_index(drop=True)
    print(f"[IBOV] {len(ibov)} dias carregados\n")

    csvs = sorted(p for p in DATA_DAILY_DIR.glob("*.csv") if p.stem != "IBOV")
    if not csvs:
        print(f"Nenhum CSV de ativo em {DATA_DAILY_DIR}. Rode download_daily.py primeiro.")
        sys.exit(1)

    print(f"Processando {len(csvs)} tickers")
    print(f"Janelas: {FEATURE_WINDOWS} dias ({len(FEATURE_WINDOWS) * 9} features por ativo)\n")
    erros: list[tuple[str, str]] = []

    for fp in csvs:
        try:
            processar_ticker(fp, ibov)
        except Exception as exc:
            erros.append((fp.stem, str(exc)))
            print(f"ERRO: {exc}")

    if erros:
        print("\nErros:")
        for ticker, msg in erros:
            print(f"  {ticker}: {msg}")
    else:
        print(f"\nTodos os {len(csvs)} tickers processados com sucesso.")


if __name__ == "__main__":
    main()
