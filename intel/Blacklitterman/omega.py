"""Calcula Omega a partir do Q standalone (``outputs/q_walkforward_long.csv``)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from .config import (
        OUT_OMEGA_STANDALONE_LONG,
        OUT_OMEGA_STANDALONE_MATRIX,
        OUT_Q_OMEGA_STANDALONE_LONG,
        OUT_Q_OMEGA_STANDALONE_WIDE,
        OUT_Q_STANDALONE_LONG,
        OUTPUTS_DIR,
        TAU,
    )
    from .io_utils import save_csv_with_date
    from .omega_calc import calculate_omega
    from .xgb_views import normalize_return_columns
except ImportError:
    from config import (
        OUT_OMEGA_STANDALONE_LONG,
        OUT_OMEGA_STANDALONE_MATRIX,
        OUT_Q_OMEGA_STANDALONE_LONG,
        OUT_Q_OMEGA_STANDALONE_WIDE,
        OUT_Q_STANDALONE_LONG,
        OUTPUTS_DIR,
        TAU,
    )
    from io_utils import save_csv_with_date
    from omega_calc import calculate_omega
    from xgb_views import normalize_return_columns


def load_q_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de views Q não encontrado: {path}. Execute Q.py primeiro.")

    df = normalize_return_columns(pd.read_csv(path))
    required = {"view_date", "ticker", "ret_pred", "ret_real", "news_count"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes em {path.name}: {sorted(missing)}")

    df["view_date"] = pd.to_datetime(df["view_date"], errors="coerce")
    df["ret_pred"] = pd.to_numeric(df["ret_pred"], errors="coerce")
    df["ret_real"] = pd.to_numeric(df["ret_real"], errors="coerce")
    df["news_count"] = pd.to_numeric(df["news_count"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["view_date", "ticker", "ret_pred", "ret_real"]).copy()

    if df.empty:
        raise ValueError("Arquivo de Q não possui linhas válidas para calcular Omega.")
    return df.sort_values(["ticker", "view_date"]).reset_index(drop=True)


def run_omega_standalone() -> None:
    print("Carregando Q long...")
    q_long = load_q_long(OUT_Q_STANDALONE_LONG)

    print("Calculando Omega...")
    omega_long = calculate_omega(q_long, tau=TAU)
    omega_matrix = omega_long.pivot(index="view_date", columns="ticker", values="omega").sort_index()
    q_matrix = q_long.pivot(index="view_date", columns="ticker", values="Q").sort_index()

    q_omega_long = (
        q_long.merge(
            omega_long[["view_date", "ticker", "omega", "omega_base", "view_confidence"]],
            on=["view_date", "ticker"],
            how="inner",
        )
        .sort_values(["view_date", "ticker"])
        .reset_index(drop=True)
    )

    q_wide = q_omega_long.pivot(index="view_date", columns="ticker", values="Q")
    q_wide.columns = [f"Q_{col}" for col in q_wide.columns]
    omega_wide = q_omega_long.pivot(index="view_date", columns="ticker", values="omega")
    omega_wide.columns = [f"OMEGA_{col}" for col in omega_wide.columns]
    q_omega_wide = pd.concat([q_wide, omega_wide], axis=1).sort_index()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    save_csv_with_date(omega_long, OUT_OMEGA_STANDALONE_LONG)
    save_csv_with_date(q_omega_long, OUT_Q_OMEGA_STANDALONE_LONG)
    omega_matrix.index = omega_matrix.index.strftime("%Y-%m-%d")
    q_matrix.index = q_matrix.index.strftime("%Y-%m-%d")
    q_omega_wide.index = q_omega_wide.index.strftime("%Y-%m-%d")
    omega_matrix.to_csv(OUT_OMEGA_STANDALONE_MATRIX)
    q_omega_wide.to_csv(OUT_Q_OMEGA_STANDALONE_WIDE)

    print(f"Omega long: {OUT_OMEGA_STANDALONE_LONG}")
    print(f"Q+Omega long: {OUT_Q_OMEGA_STANDALONE_LONG}")
    print(f"Linhas: {len(omega_long)}")


def main() -> None:
    run_omega_standalone()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
