"""
Compara desempenho do modelo híbrido (Q) e do Lag-Llama puro (ret_7h_pred)
contra o retorno realizado (ret_7h_real).

Uso:
    python teste_modelo.py
    python teste_modelo.py --csv q_ab_ev3_walkforward_long.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_ROOT / "q_ab_ev3_walkforward_long.csv"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or not np.isfinite(denominator):
        return np.nan
    return numerator / denominator


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if y_true.size == 0:
        return {
            "n_amostras": 0.0,
            "acuracia_direcional": np.nan,
            "information_ratio": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "bias_medio": np.nan,
        }

    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    bias = float(np.mean(error))

    directional_hits = np.sign(y_pred) == np.sign(y_true)
    directional_acc = float(np.mean(directional_hits))

    tracking_error = float(np.std(error, ddof=1)) if y_true.size > 1 else np.nan
    information_ratio = np.nan if not np.isfinite(tracking_error) or tracking_error == 0 else bias / tracking_error

    return {
        "n_amostras": float(y_true.size),
        "acuracia_direcional": directional_acc,
        "information_ratio": information_ratio,
        "mae": mae,
        "rmse": rmse,
        "bias_medio": bias,
    }


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"Q", "ret_7h_pred", "ret_7h_real"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {sorted(missing)}")

    clean = df.dropna(subset=["Q", "ret_7h_pred", "ret_7h_real"]).copy()
    if clean.empty:
        raise ValueError("Não há linhas válidas após remover valores ausentes.")

    y_true = clean["ret_7h_real"].to_numpy(dtype=float)
    y_hibrido = clean["Q"].to_numpy(dtype=float)
    y_lag_llama = clean["ret_7h_pred"].to_numpy(dtype=float)

    metrics_hibrido = compute_metrics(y_true, y_hibrido)
    metrics_lag_llama = compute_metrics(y_true, y_lag_llama)

    return pd.DataFrame(
        {
            "hibrido_Q": metrics_hibrido,
            "lag_llama_pred": metrics_lag_llama,
        }
    )


def print_winners(metrics_df: pd.DataFrame) -> None:
    print("\nComparativo por métrica:")

    # Maior é melhor
    acc_h = metrics_df.loc["acuracia_direcional", "hibrido_Q"]
    acc_l = metrics_df.loc["acuracia_direcional", "lag_llama_pred"]
    ir_h = metrics_df.loc["information_ratio", "hibrido_Q"]
    ir_l = metrics_df.loc["information_ratio", "lag_llama_pred"]

    # Menor é melhor
    mae_h = metrics_df.loc["mae", "hibrido_Q"]
    mae_l = metrics_df.loc["mae", "lag_llama_pred"]
    rmse_h = metrics_df.loc["rmse", "hibrido_Q"]
    rmse_l = metrics_df.loc["rmse", "lag_llama_pred"]

    print(
        f"- Acurácia direcional: {'Híbrido' if acc_h > acc_l else 'Lag-Llama' if acc_l > acc_h else 'Empate'} "
        f"({_safe_div(acc_h, 1):.2%} vs {_safe_div(acc_l, 1):.2%})"
    )
    print(
        f"- Information Ratio (IR): {'Híbrido' if ir_h > ir_l else 'Lag-Llama' if ir_l > ir_h else 'Empate'} "
        f"({ir_h:.4f} vs {ir_l:.4f})"
    )
    print(
        f"- MAE: {'Híbrido' if mae_h < mae_l else 'Lag-Llama' if mae_l < mae_h else 'Empate'} "
        f"({mae_h:.6f} vs {mae_l:.6f})"
    )
    print(
        f"- RMSE: {'Híbrido' if rmse_h < rmse_l else 'Lag-Llama' if rmse_l < rmse_h else 'Empate'} "
        f"({rmse_h:.6f} vs {rmse_l:.6f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia lado a lado o modelo híbrido (Q) e o Lag-Llama (ret_7h_pred)."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Caminho do CSV long (default: {DEFAULT_CSV.name})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    metrics_df = compare_models(df)

    print(f"Arquivo analisado: {csv_path}")
    print("\nMétricas (quanto maior melhor: acurácia/IR; quanto menor melhor: MAE/RMSE):")
    print("IR = média(y_pred - y_true) / desvio_padrão(y_pred - y_true)")
    print(metrics_df.round(6).to_string())
    print_winners(metrics_df)


if __name__ == "__main__":
    main()
