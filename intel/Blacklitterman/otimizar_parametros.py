"""
Otimização de hiperparâmetros do Black-Litterman (Optuna).

Parâmetros tunados:
- alpha_q, tau, max_weight_per_asset
- w_model_conf / w_news_conf, confidence_floor, confidence_cap
- err_window, rolling_min_obs

Objetivo padrão: Sharpe do portfólio realizado no período de validação (out-of-sample).

Uso:
    cd intel/Blacklitterman
    python otimizar_parametros.py
    python otimizar_parametros.py --trials 60 --rebalance weekly
    python otimizar_parametros.py --apply-best   # roda pipeline com melhores params
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

try:
    from .backtest import build_gain_series, portfolio_metrics
    from .config import (
        ATIVOS,
        OPT_DEFAULT_REBALANCE_MODE,
        OPT_DEFAULT_TRIALS,
        OPT_TRAIN_START,
        OPT_VAL_END,
        OPT_VAL_START,
        OUT_OPT_BEST_PARAMS,
        OUT_OPT_SUMMARY,
        OUT_OPT_TRIALS,
        PRIOR_END_DATE,
        PRIOR_START_DATE,
    )
    from .params import BLParams
    from .pipeline import run_pipeline
    from .prior import get_daily_returns_for_prior, get_market_cap_weights, ticker_from_sa
    from .sentiment import load_daily_sentiment_for_ticker
    from .xgb_views import (
        add_ret_scale,
        build_q_from_cached_views,
        load_xgb_predictions,
        month_bounds,
    )
except ImportError:
    from backtest import build_gain_series, portfolio_metrics
    from config import (
        ATIVOS,
        OPT_DEFAULT_REBALANCE_MODE,
        OPT_DEFAULT_TRIALS,
        OPT_TRAIN_START,
        OPT_VAL_END,
        OPT_VAL_START,
        OUT_OPT_BEST_PARAMS,
        OUT_OPT_SUMMARY,
        OUT_OPT_TRIALS,
        PRIOR_END_DATE,
        PRIOR_START_DATE,
    )
    from params import BLParams
    from pipeline import run_pipeline
    from prior import get_daily_returns_for_prior, get_market_cap_weights, ticker_from_sa
    from sentiment import load_daily_sentiment_for_ticker
    from xgb_views import (
        add_ret_scale,
        build_q_from_cached_views,
        load_xgb_predictions,
        month_bounds,
    )


@dataclass
class OptimizationContext:
    views_by_ticker: dict[str, pd.DataFrame]
    sentiment_by_ticker: dict[str, pd.DataFrame]
    market_weights: pd.Series
    returns_history: pd.DataFrame
    daily_log_returns: pd.DataFrame
    period_start: str
    period_end: str
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    rebalance_mode: str


def _period_bounds(period_start: str, period_end: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = month_bounds(period_start, period_end)
    return start, end


def build_context(
    period_start: str = OPT_TRAIN_START,
    period_end: str = OPT_VAL_END,
    val_start: str = OPT_VAL_START,
    val_end: str = OPT_VAL_END,
    rebalance_mode: str = OPT_DEFAULT_REBALANCE_MODE,
) -> OptimizationContext:
    view_tickers = [ticker_from_sa(t) for t in ATIVOS]
    views_by_ticker: dict[str, pd.DataFrame] = {}
    sentiment_by_ticker: dict[str, pd.DataFrame] = {}

    for ticker in view_tickers:
        views = load_xgb_predictions(ticker, period_start=period_start, period_end=period_end)
        if views.empty:
            raise RuntimeError(f"Sem predições XGB para {ticker} em {period_start}-{period_end}")
        views_by_ticker[ticker] = add_ret_scale(views)
        sentiment_by_ticker[ticker] = load_daily_sentiment_for_ticker(ticker, match_company=True)

    market_weights = get_market_cap_weights(ATIVOS)
    returns_history = get_daily_returns_for_prior(
        tickers=ATIVOS,
        start_date=PRIOR_START_DATE,
        end_date=PRIOR_END_DATE,
    )
    start_ts, end_ts = _period_bounds(period_start, period_end)
    daily_log_returns = returns_history.copy()
    daily_log_returns.index = pd.to_datetime(daily_log_returns.index)
    daily_log_returns = daily_log_returns[
        (daily_log_returns.index >= start_ts) & (daily_log_returns.index <= end_ts)
    ]

    val_start_ts, val_end_ts = _period_bounds(val_start, val_end)
    return OptimizationContext(
        views_by_ticker=views_by_ticker,
        sentiment_by_ticker=sentiment_by_ticker,
        market_weights=market_weights,
        returns_history=returns_history,
        daily_log_returns=daily_log_returns,
        period_start=period_start,
        period_end=period_end,
        val_start=val_start_ts,
        val_end=val_end_ts,
        rebalance_mode=rebalance_mode,
    )


def evaluate_params(params: BLParams, ctx: OptimizationContext) -> dict[str, float]:
    q_long = build_q_from_cached_views(
        ctx.views_by_ticker,
        ctx.sentiment_by_ticker,
        alpha=params.alpha_q,
    )
    outputs = run_pipeline(
        params,
        period_start=ctx.period_start,
        period_end=ctx.period_end,
        q_long=q_long,
        market_weights=ctx.market_weights,
        returns_history=ctx.returns_history,
        save_outputs=False,
        verbose=False,
    )
    weights = outputs["weights_by_mode"][params.rebalance_mode].copy()
    mu = outputs["posterior_mu"].copy()
    mu["view_date"] = pd.to_datetime(mu["view_date"], errors="coerce")
    weights["view_date"] = pd.to_datetime(weights["view_date"], errors="coerce")
    gain = build_gain_series(
        mu,
        weights,
        ctx.daily_log_returns,
        mode_label=params.rebalance_mode,
    )
    metrics = portfolio_metrics(gain, eval_start=ctx.val_start, eval_end=ctx.val_end)
    full_metrics = portfolio_metrics(gain)
    return {**metrics, **{f"full_{k}": v for k, v in full_metrics.items()}}


def objective_score(metrics: dict[str, float], *, mdd_penalty: float = 0.5) -> float:
    """Sharpe na validação com penalidade leve por drawdown profundo."""
    sharpe = metrics.get("sharpe", float("nan"))
    mdd = metrics.get("max_drawdown", float("nan"))
    if not np.isfinite(sharpe):
        return -1e6
    penalty = 0.0
    if np.isfinite(mdd):
        penalty = mdd_penalty * abs(min(float(mdd), 0.0))
    return float(sharpe + penalty)


def suggest_params(trial: optuna.Trial, rebalance_mode: str) -> BLParams:
    w_model = trial.suggest_float("w_model_conf", 0.3, 0.9)
    err_window = trial.suggest_int("err_window", 7, 28)
    err_min_periods = trial.suggest_int("err_min_periods", 3, min(10, err_window))
    return BLParams(
        alpha_q=trial.suggest_float("alpha_q", 0.1, 0.9),
        tau=trial.suggest_float("tau", 0.005, 0.08, log=True),
        max_weight_per_asset=trial.suggest_float("max_weight_per_asset", 0.20, 0.50),
        w_model_conf=w_model,
        w_news_conf=1.0 - w_model,
        confidence_floor=trial.suggest_float("confidence_floor", 0.01, 0.20),
        confidence_cap=trial.suggest_float("confidence_cap", 0.70, 0.99),
        err_window=err_window,
        err_min_periods=err_min_periods,
        rolling_min_obs=trial.suggest_int("rolling_min_obs", 40, 120, step=10),
        use_rolling_prior=True,
        rebalance_mode=rebalance_mode,
    ).normalized()


def run_optimization(
    *,
    n_trials: int = OPT_DEFAULT_TRIALS,
    rebalance_mode: str = OPT_DEFAULT_REBALANCE_MODE,
    train_start: str = OPT_TRAIN_START,
    val_start: str = OPT_VAL_START,
    val_end: str = OPT_VAL_END,
) -> tuple[BLParams, pd.DataFrame, dict[str, float]]:
    print("Carregando cache (XGB, sentimento, retornos, market cap)...")
    # Pipeline na otimização: histórico longo (train_start→val_end);
    # objective só olha a janela de validação (ex.: 2025).
    ctx = build_context(
        period_start=train_start,
        period_end=val_end,
        val_start=val_start,
        val_end=val_end,
        rebalance_mode=rebalance_mode,
    )

    print(f"Histórico na otimização: {train_start} -> {val_end}")
    print(f"Validação out-of-sample: {ctx.val_start.date()} -> {ctx.val_end.date()}")
    print(f"Modo de rebalanceamento: {rebalance_mode}")
    print(f"Trials Optuna: {n_trials}")

    study = optuna.create_study(direction="maximize", study_name="bl_hibrido_params")

    def _objective(trial: optuna.Trial) -> float:
        try:
            params = suggest_params(trial, rebalance_mode)
            metrics = evaluate_params(params, ctx)
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("params", params.to_dict())
            return objective_score(metrics)
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return -1e6

    study.optimize(
        _objective,
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Nenhum trial concluiu com sucesso. Verifique os dados e parâmetros.")

    best = study.best_trial
    best_params = BLParams(**{k: v for k, v in best.user_attrs["params"].items()}).normalized()
    best_metrics = best.user_attrs["metrics"]

    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": trial.number, "objective": trial.value}
        row.update(trial.user_attrs.get("params", {}))
        row.update({f"val_{k}": v for k, v in trial.user_attrs.get("metrics", {}).items()})
        rows.append(row)
    trials_df = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)

    OUT_OPT_TRIALS.parent.mkdir(parents=True, exist_ok=True)
    trials_df.to_csv(OUT_OPT_TRIALS, index=False)

    best_row = best_params.to_dict()
    best_row.update({f"val_{k}": v for k, v in best_metrics.items()})
    best_row["objective"] = best.value
    pd.DataFrame([best_row]).to_csv(OUT_OPT_BEST_PARAMS, index=False)

    summary = {
        "best_objective": float(best.value),
        "n_trials": n_trials,
        "rebalance_mode": rebalance_mode,
        "train_start": train_start,
        "val_start": str(ctx.val_start.date()),
        "val_end": str(ctx.val_end.date()),
        "best_params": best_params.to_dict(),
        "val_metrics": {k: v for k, v in best_metrics.items() if not str(k).startswith("full_")},
    }
    OUT_OPT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nMelhor objective (val): {best.value:.4f}")
    print(f"Melhores parâmetros: {json.dumps(best_params.to_dict(), ensure_ascii=False)}")
    print(f"Sharpe val: {best_metrics.get('sharpe', float('nan')):.4f}")
    print(f"Retorno val: {best_metrics.get('total_return_pct', float('nan')):.2f}%")
    print(f"Trials salvos em: {OUT_OPT_TRIALS}")
    print(f"Melhor config salva em: {OUT_OPT_BEST_PARAMS}")
    print(f"Resumo JSON em: {OUT_OPT_SUMMARY}")

    return best_params, trials_df, best_metrics


def apply_best_params_to_config(best: BLParams) -> str:
    """Retorna snippet para colar em config.py (não altera arquivo automaticamente)."""
    mapping = {
        "ALPHA_Q": best.alpha_q,
        "TAU": best.tau,
        "MAX_WEIGHT_PER_ASSET": best.max_weight_per_asset,
        "W_MODEL_CONF": best.w_model_conf,
        "W_NEWS_CONF": best.w_news_conf,
        "CONFIDENCE_FLOOR": best.confidence_floor,
        "CONFIDENCE_CAP": best.confidence_cap,
        "ERR_WINDOW": best.err_window,
        "ERR_MIN_PERIODS": best.err_min_periods,
        "ROLLING_MIN_OBS": best.rolling_min_obs,
    }
    lines = ["# Melhores parâmetros (otimizar_parametros.py):"]
    for key, value in mapping.items():
        if isinstance(value, float):
            lines.append(f"{key} = {value:.6g}")
        else:
            lines.append(f"{key} = {value}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Otimiza hiperparâmetros do BL híbrido (Optuna).")
    parser.add_argument("--trials", type=int, default=OPT_DEFAULT_TRIALS, help="Número de trials Optuna")
    parser.add_argument(
        "--rebalance",
        choices=("daily", "weekly", "monthly"),
        default=OPT_DEFAULT_REBALANCE_MODE,
        help="Modo de rebalanceamento avaliado",
    )
    parser.add_argument("--val-start", default=OPT_VAL_START, help="Início validação YYYYMM")
    parser.add_argument("--val-end", default=OPT_VAL_END, help="Fim validação YYYYMM")
    parser.add_argument(
        "--train-start",
        default=OPT_TRAIN_START,
        help="Início do histórico na otimização YYYYMM (warm-up + notícias longas)",
    )
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Após otimizar, roda pipeline completo com melhores parâmetros e salva CSVs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_params, _, _ = run_optimization(
        n_trials=args.trials,
        rebalance_mode=args.rebalance,
        train_start=args.train_start,
        val_start=args.val_start,
        val_end=args.val_end,
    )

    print("\n" + apply_best_params_to_config(best_params))

    if args.apply_best:
        print("\nAplicando melhores parâmetros (pipeline + plot + Markowitz)...")
        try:
            from .aplicar_melhores_params import apply_and_run
        except ImportError:
            from aplicar_melhores_params import apply_and_run
        apply_and_run(OUT_OPT_BEST_PARAMS, write_config=False, skip_eval=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Erro na otimização: {exc}")
        sys.exit(1)
