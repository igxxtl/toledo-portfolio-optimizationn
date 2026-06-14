"""
Calcula a matriz de incertezas Omega (Black-Litterman) em arquivo separado.

Entrada esperada:
- q_ab_ev3_walkforward_long.csv (gerado por Q.py)

Saidas:
- omega_ab_ev3_walkforward_long.csv   (componentes por dia)
- omega_ab_ev3_walkforward_matrix.csv (matriz Omega por dia/ticker)

Regra base (He-Litterman):
- omega_base = tau * sigma_ii

Regra usada no script (dinamica com confianca):
- confianca_t combina erro preditivo e volume de noticias
- omega_t = omega_base * (1 - confianca_t) / confianca_t
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
Q_LONG_PATH = PROJECT_ROOT / "q_ab_ev3_walkforward_long.csv"
OUT_LONG = PROJECT_ROOT / "omega_ab_ev3_walkforward_long.csv"
OUT_MATRIX = PROJECT_ROOT / "omega_ab_ev3_walkforward_matrix.csv"
OUT_Q_OMEGA_LONG = PROJECT_ROOT / "q_omega_ab_ev3_walkforward_long.csv"
OUT_Q_OMEGA_WIDE = PROJECT_ROOT / "q_omega_ab_ev3_walkforward_wide.csv"

# Parametros Black-Litterman
TAU = 0.025

# Parametros da confianca dinamica
ERR_WINDOW = 14
ERR_MIN_PERIODS = 5
EPS = 1e-8
CONFIDENCE_FLOOR = 0.05
CONFIDENCE_CAP = 0.95
W_MODEL_CONF = 0.7
W_NEWS_CONF = 0.3


def load_q_long(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de views Q nao encontrado: {path}")

    df = pd.read_csv(path)
    required_cols = {"view_date", "ticker", "ret_7h_pred", "ret_7h_real", "news_count"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes em {path.name}: {sorted(missing)}")

    df["view_date"] = pd.to_datetime(df["view_date"], errors="coerce")
    df["ret_7h_pred"] = pd.to_numeric(df["ret_7h_pred"], errors="coerce")
    df["ret_7h_real"] = pd.to_numeric(df["ret_7h_real"], errors="coerce")
    df["news_count"] = pd.to_numeric(df["news_count"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["view_date", "ticker", "ret_7h_pred", "ret_7h_real"]).copy()

    if df.empty:
        raise ValueError("Arquivo de Q nao possui linhas validas para calcular Omega.")

    return df.sort_values(["ticker", "view_date"]).reset_index(drop=True)


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

    out["error_scale"] = (
        out["pred_abs_error"]
        .rolling(ERR_WINDOW, min_periods=ERR_MIN_PERIODS)
        .median()
        .shift(1)
    )
    global_err_scale = float(out["pred_abs_error"].median())
    if not np.isfinite(global_err_scale) or global_err_scale <= 0:
        global_err_scale = 0.005
    out["error_scale"] = out["error_scale"].fillna(global_err_scale).clip(lower=EPS)

    # Confianca do modelo cai quando erro atual aumenta.
    out["model_confidence"] = np.exp(-out["pred_abs_error"] / out["error_scale"])

    # Confianca de noticias cresce com quantidade de noticias (saturacao suave).
    out["news_confidence"] = 1.0 - np.exp(-out["news_count"] / 3.0)

    out["view_confidence"] = (
        W_MODEL_CONF * out["model_confidence"] + W_NEWS_CONF * out["news_confidence"]
    ).clip(lower=CONFIDENCE_FLOOR, upper=CONFIDENCE_CAP)

    return out


def calculate_omega(q_long: pd.DataFrame, tau: float) -> pd.DataFrame:
    # Para view absoluta por ativo, com P=I (ou P com unica linha [1] no caso 1 ativo):
    # Omega_base_i = tau * sigma_ii
    sigma_by_ticker: dict[str, float] = {}
    for ticker, group in q_long.groupby("ticker", sort=True):
        sigma_ii = _safe_sample_var(group["ret_7h_real"])
        if not np.isfinite(sigma_ii):
            # fallback defensivo
            sigma_ii = _safe_sample_var(group["ret_7h_pred"])
        if not np.isfinite(sigma_ii):
            sigma_ii = 2.5e-5
        sigma_by_ticker[ticker] = sigma_ii

    parts = []
    for _, group in q_long.groupby("ticker", sort=True):
        parts.append(_compute_confidence_per_ticker(group))
    with_conf = pd.concat(parts, axis=0, ignore_index=True)

    with_conf["sigma_ii"] = with_conf["ticker"].map(sigma_by_ticker)
    with_conf["omega_base"] = tau * with_conf["sigma_ii"]
    with_conf["omega"] = with_conf["omega_base"] * (1.0 - with_conf["view_confidence"]) / with_conf["view_confidence"]
    with_conf["omega"] = with_conf["omega"].clip(lower=EPS)
    with_conf["tau"] = tau

    cols = [
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
    out = with_conf[cols].sort_values(["view_date", "ticker"]).reset_index(drop=True)
    return out


def main() -> None:
    print("Carregando Q long...")
    q_long = load_q_long(Q_LONG_PATH)

    print("Calculando Omega...")
    omega_long = calculate_omega(q_long, tau=TAU)
    omega_matrix = omega_long.pivot(index="view_date", columns="ticker", values="omega").sort_index()
    q_matrix = q_long.pivot(index="view_date", columns="ticker", values="Q").sort_index()

    # Integracao direta para uso no BL: Q + Omega na mesma linha (por dia/ticker).
    q_omega_long = q_long.merge(
        omega_long[["view_date", "ticker", "omega", "omega_base", "view_confidence"]],
        on=["view_date", "ticker"],
        how="inner",
    ).sort_values(["view_date", "ticker"]).reset_index(drop=True)

    # Matriz wide com Q e Omega lado a lado.
    q_wide = q_omega_long.pivot(index="view_date", columns="ticker", values="Q")
    q_wide.columns = [f"Q_{col}" for col in q_wide.columns]
    omega_wide = q_omega_long.pivot(index="view_date", columns="ticker", values="omega")
    omega_wide.columns = [f"OMEGA_{col}" for col in omega_wide.columns]
    q_omega_wide = pd.concat([q_wide, omega_wide], axis=1).sort_index()

    last_row = omega_long.sort_values("view_date").tail(1).copy()

    omega_long["view_date"] = omega_long["view_date"].dt.strftime("%Y-%m-%d")
    q_omega_long["view_date"] = q_omega_long["view_date"].dt.strftime("%Y-%m-%d")
    omega_matrix.index = omega_matrix.index.strftime("%Y-%m-%d")
    q_matrix.index = q_matrix.index.strftime("%Y-%m-%d")
    q_omega_wide.index = q_omega_wide.index.strftime("%Y-%m-%d")
    omega_long.to_csv(OUT_LONG, index=False)
    omega_matrix.to_csv(OUT_MATRIX)
    q_omega_long.to_csv(OUT_Q_OMEGA_LONG, index=False)
    q_omega_wide.to_csv(OUT_Q_OMEGA_WIDE)

    print(f"Omega long salvo em: {OUT_LONG}")
    print(f"Matriz Omega salva em: {OUT_MATRIX}")
    print(f"Q+Omega long salvo em: {OUT_Q_OMEGA_LONG}")
    print(f"Q+Omega wide salvo em: {OUT_Q_OMEGA_WIDE}")
    print(f"Linhas: {len(omega_long)}")
    print(f"Dias com Q: {len(q_matrix)}")
    print("Ultima linha da matriz Omega:")
    print(omega_matrix.tail(1).round(8))
    if not last_row.empty:
        print("\nResumo da ultima view:")
        print(
            "ticker={ticker} | data={data} | view_confidence={conf:.4f} | "
            "omega_base={omega_base:.8f} | omega={omega:.8f}".format(
                ticker=str(last_row["ticker"].iloc[0]),
                data=pd.Timestamp(last_row["view_date"].iloc[0]).strftime("%Y-%m-%d"),
                conf=float(last_row["view_confidence"].iloc[0]),
                omega_base=float(last_row["omega_base"].iloc[0]),
                omega=float(last_row["omega"].iloc[0]),
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
