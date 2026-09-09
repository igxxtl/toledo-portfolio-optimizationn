"""Executa o pipeline completo de treino XGBoost (download -> features -> treino)."""
from __future__ import annotations

import argparse
import sys


def _run(step: str) -> None:
    if step == "download":
        from .download_daily import main as download_main

        download_main()
    elif step == "features":
        from .prepare_features import main as features_main

        features_main()
    elif step == "train":
        from .train_xgb import main as train_main

        train_main()
    else:
        raise ValueError(f"Etapa desconhecida: {step}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pipeline de treino XGBoost para o vetor Q.")
    parser.add_argument(
        "steps",
        nargs="*",
        choices=["download", "features", "train", "all"],
        default=["all"],
        help="Etapas a executar (padrao: all).",
    )
    args = parser.parse_args(argv)

    steps = ["download", "features", "train"] if "all" in args.steps else list(args.steps)

    for step in steps:
        print(f"\n{'=' * 60}\nEtapa: {step}\n{'=' * 60}")
        _run(step)

    print("\nPipeline de treino XGB concluido.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
