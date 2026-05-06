"""
Avaliacao do sinal de sentimento de noticias vs retorno realizado.

Compara duas visoes:
1) Direcional pura: sentiment_signal (-1..1) contra direcao de ret_7h_real.
2) Em unidade de retorno: sentiment_component (signal * ret_scale) contra ret_7h_real.

Uso:
    python teste_sentimento.py
    python teste_sentimento.py --csv q_ab_ev3_walkforward_long.csv --somente-com-noticia
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_ROOT / "q_ab_ev3_walkforward_long.csv"


def _fmt_pct(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.2%}"


def _fmt_num(value: float, digits: int = 6) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.{digits}f}"


def directional_accuracy(pred: np.ndarray, real: np.ndarray) -> float:
    mask = pred != 0
    if not np.any(mask):
        return np.nan
    hits = np.sign(pred[mask]) == np.sign(real[mask])
    return float(np.mean(hits))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(pred: np.ndarray, real: np.ndarray) -> tuple[float, float]:
    err = pred - real
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return mae, rmse


def build_report(df: pd.DataFrame) -> dict[str, float]:
    y_real = df["ret_7h_real"].to_numpy(dtype=float)
    y_signal = df["sentiment_signal"].to_numpy(dtype=float)
    y_component = df["sentiment_component"].to_numpy(dtype=float)

    acc_signal = directional_accuracy(y_signal, y_real)
    corr_signal = pearson_corr(y_signal, y_real)

    acc_component = directional_accuracy(y_component, y_real)
    corr_component = pearson_corr(y_component, y_real)
    mae_component, rmse_component = regression_metrics(y_component, y_real)

    mean_real_when_pos = float(np.mean(y_real[y_signal > 0])) if np.any(y_signal > 0) else np.nan
    mean_real_when_neg = float(np.mean(y_real[y_signal < 0])) if np.any(y_signal < 0) else np.nan

    return {
        "n_linhas": float(len(df)),
        "dias_com_sentimento_diferente_de_zero": float(np.sum(y_signal != 0)),
        "acuracia_direcional_signal": acc_signal,
        "correlacao_signal_vs_real": corr_signal,
        "acuracia_direcional_component": acc_component,
        "correlacao_component_vs_real": corr_component,
        "mae_component_vs_real": mae_component,
        "rmse_component_vs_real": rmse_component,
        "media_real_quando_signal_positivo": mean_real_when_pos,
        "media_real_quando_signal_negativo": mean_real_when_neg,
    }


def print_report(title: str, report: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Linhas avaliadas: {int(report['n_linhas'])}")
    print(f"Dias com signal != 0: {int(report['dias_com_sentimento_diferente_de_zero'])}")
    print(f"Acurácia direcional (signal): {_fmt_pct(report['acuracia_direcional_signal'])}")
    print(f"Correlação (signal vs real): {_fmt_num(report['correlacao_signal_vs_real'], 4)}")
    print(f"Acurácia direcional (component): {_fmt_pct(report['acuracia_direcional_component'])}")
    print(f"Correlação (component vs real): {_fmt_num(report['correlacao_component_vs_real'], 4)}")
    print(f"MAE (component vs real): {_fmt_num(report['mae_component_vs_real'])}")
    print(f"RMSE (component vs real): {_fmt_num(report['rmse_component_vs_real'])}")
    print(
        "Média de retorno real quando signal>0: "
        f"{_fmt_num(report['media_real_quando_signal_positivo'])}"
    )
    print(
        "Média de retorno real quando signal<0: "
        f"{_fmt_num(report['media_real_quando_signal_negativo'])}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia sinal de sentimento de notícias contra ret_7h_real."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Caminho do CSV long (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--somente-com-noticia",
        action="store_true",
        help="Se informado, analisa somente linhas com news_count > 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {args.csv}")

    df = pd.read_csv(args.csv)
    required = {"ret_7h_real", "sentiment_signal", "sentiment_component", "news_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {sorted(missing)}")

    base = df.dropna(subset=list(required)).copy()
    if base.empty:
        raise ValueError("Não há dados válidos para análise após remover nulos.")

    print(f"Arquivo analisado: {args.csv}")
    print(f"Total de linhas no CSV: {len(df)}")
    print(f"Linhas válidas para análise: {len(base)}")

    if args.somente_com_noticia:
        news = base[base["news_count"] > 0].copy()
        if news.empty:
            raise ValueError("Não há linhas com news_count > 0 para análise.")
        report_news = build_report(news)
        print_report("Avaliação (somente dias com notícia)", report_news)
        return

    report_all = build_report(base)
    report_news = build_report(base[base["news_count"] > 0].copy())

    print_report("Avaliação (todos os dias)", report_all)
    print_report("Avaliação (somente dias com notícia)", report_news)


if __name__ == "__main__":
    main()
