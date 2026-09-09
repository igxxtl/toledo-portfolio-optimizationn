"""
Aplica os melhores hiperparâmetros (``outputs/optimize/best_params.csv``)
e executa pipeline + backtest + comparação Markowitz.

Uso:
    python aplicar_melhores_params.py
    python aplicar_melhores_params.py --params outputs/optimize/best_params.csv
    python aplicar_melhores_params.py --write-config   # atualiza config.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

try:
    from .config import (
        OUT_OPT_BEST_PARAMS,
        ALPHA_Q,
        CONFIDENCE_CAP,
        CONFIDENCE_FLOOR,
        ERR_MIN_PERIODS,
        ERR_WINDOW,
        MAX_WEIGHT_PER_ASSET,
        ROLLING_MIN_OBS,
        TAU,
        W_MODEL_CONF,
        W_NEWS_CONF,
    )
    from .otimizar_parametros import apply_best_params_to_config
    from .params import BLParams
    from .pipeline import run_pipeline
except ImportError:
    from config import (
        OUT_OPT_BEST_PARAMS,
        ALPHA_Q,
        CONFIDENCE_CAP,
        CONFIDENCE_FLOOR,
        ERR_MIN_PERIODS,
        ERR_WINDOW,
        MAX_WEIGHT_PER_ASSET,
        ROLLING_MIN_OBS,
        TAU,
        W_MODEL_CONF,
        W_NEWS_CONF,
    )
    from otimizar_parametros import apply_best_params_to_config
    from params import BLParams
    from pipeline import run_pipeline

BL_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BL_DIR / "config.py"

_PARAM_COLUMNS = (
    "alpha_q",
    "tau",
    "max_weight_per_asset",
    "w_model_conf",
    "w_news_conf",
    "confidence_floor",
    "confidence_cap",
    "err_window",
    "err_min_periods",
    "rolling_min_obs",
    "use_rolling_prior",
    "long_only",
    "rebalance_mode",
)


def load_params_from_csv(path: Path) -> BLParams:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de parâmetros não encontrado: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV vazio: {path}")
    row = df.iloc[0]
    missing = [c for c in _PARAM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {missing}")
    return BLParams(
        alpha_q=float(row["alpha_q"]),
        tau=float(row["tau"]),
        max_weight_per_asset=float(row["max_weight_per_asset"]),
        w_model_conf=float(row["w_model_conf"]),
        w_news_conf=float(row["w_news_conf"]),
        confidence_floor=float(row["confidence_floor"]),
        confidence_cap=float(row["confidence_cap"]),
        err_window=int(row["err_window"]),
        err_min_periods=int(row["err_min_periods"]),
        rolling_min_obs=int(row["rolling_min_obs"]),
        use_rolling_prior=bool(row["use_rolling_prior"]),
        long_only=bool(row["long_only"]),
        rebalance_mode=str(row.get("rebalance_mode", "weekly")),
    ).normalized()


def write_config_py(params: BLParams) -> None:
    """Atualiza constantes tunáveis em ``config.py``."""
    mapping = {
        "ALPHA_Q": params.alpha_q,
        "TAU": params.tau,
        "MAX_WEIGHT_PER_ASSET": params.max_weight_per_asset,
        "W_MODEL_CONF": params.w_model_conf,
        "W_NEWS_CONF": params.w_news_conf,
        "CONFIDENCE_FLOOR": params.confidence_floor,
        "CONFIDENCE_CAP": params.confidence_cap,
        "ERR_WINDOW": params.err_window,
        "ERR_MIN_PERIODS": params.err_min_periods,
        "ROLLING_MIN_OBS": params.rolling_min_obs,
    }
    text = CONFIG_PATH.read_text(encoding="utf-8")
    for key, value in mapping.items():
        if isinstance(value, float):
            replacement = f"{key} = {value:.6g}"
        else:
            replacement = f"{key} = {value}"
        pattern = rf"^{re.escape(key)}\s*=\s*.+$"
        if not re.search(pattern, text, flags=re.MULTILINE):
            raise ValueError(f"Constante {key} não encontrada em config.py")
        text = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
    CONFIG_PATH.write_text(text, encoding="utf-8")


def run_evaluation_scripts() -> None:
    for script in ("plot_ganho_estimado.py", "comparar_bl_vs_markowitz.py"):
        print(f"\n>>> python {script}")
        subprocess.run([sys.executable, str(BL_DIR / script)], check=True, cwd=BL_DIR)


def apply_and_run(
    params_path: Path,
    *,
    write_config: bool = False,
    skip_eval: bool = False,
) -> BLParams:
    params = load_params_from_csv(params_path)
    print("Parâmetros carregados:")
    for k, v in params.to_dict().items():
        print(f"  {k}: {v}")

    if write_config:
        write_config_py(params)
        print(f"\nconfig.py atualizado: {CONFIG_PATH}")

    print("\n>>> pipeline.py (com parâmetros otimizados)")
    run_pipeline(params, save_outputs=True, verbose=True)

    if not skip_eval:
        run_evaluation_scripts()

    print("\nConcluído. Saídas em outputs/pipeline, outputs/backtest, outputs/markowitz, outputs/figures")
    print("\n" + apply_best_params_to_config(params))
    return params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aplica best_params.csv e roda pipeline + plot + Markowitz.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=OUT_OPT_BEST_PARAMS,
        help="CSV com melhores parâmetros (default: outputs/optimize/best_params.csv)",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Grava os valores em config.py antes de executar",
    )
    parser.add_argument(
        "--pipeline-only",
        action="store_true",
        help="Roda só o pipeline (sem plot e comparar)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_and_run(
        args.params.resolve(),
        write_config=args.write_config,
        skip_eval=args.pipeline_only,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
