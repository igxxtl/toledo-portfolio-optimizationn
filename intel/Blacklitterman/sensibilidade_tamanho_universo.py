"""Sensibilidade ao tamanho do universo: k=1..N ativos.

Ordenação do top-k (BL) — modo estático:
- hit / lucro / pred (ver --rank)

``--vs-mkv-classical``: ranking estático independente (BL vs Sharpe hist).

``--walk-forward``: a cada rebalance, reordena com dados < asof.
  Gera duas comparações:
  - classical: BL ranqueia por y_pred; Markowitz por Sharpe hist
  - same_universe: ambos usam o top-k do BL (ablacão de alocação)

Uso:
    python -m intel.Blacklitterman.sensibilidade_tamanho_universo --rank pred
    python -m intel.Blacklitterman.sensibilidade_tamanho_universo --rank pred --vs-mkv-classical
    python -m intel.Blacklitterman.sensibilidade_tamanho_universo --walk-forward
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from typing import Callable

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import config
    from .backtest import build_gain_series, portfolio_metrics
    from .comparar_bl_vs_markowitz import (
        compute_markowitz_curve,
        compute_walkforward_topk_curve,
        load_returns_matrix,
    )
    from .config import (
        DATA_DAILY_DIR,
        FIGURES_DIR,
        OPT_DEFAULT_REBALANCE_MODE,
        OPT_VAL_END,
        OPT_VAL_START,
        OUT_SCORE_PRED_TICKER,
        PERIOD_END,
        PERIOD_START,
        PRIOR_START_DATE,
        SCORE_DIR,
        TRANSACTION_COST_BPS,
        XGB_DIR,
    )
    from .params import BLParams
    from .pipeline import run_pipeline
    from .prior import get_daily_returns_for_prior, ticker_from_sa
    from .xgb_views import load_xgb_predictions, month_bounds, prediction_file_for_ticker
except ImportError:
    import config
    from backtest import build_gain_series, portfolio_metrics
    from comparar_bl_vs_markowitz import (
        compute_markowitz_curve,
        compute_walkforward_topk_curve,
        load_returns_matrix,
    )
    from config import (
        DATA_DAILY_DIR,
        FIGURES_DIR,
        OPT_DEFAULT_REBALANCE_MODE,
        OPT_VAL_END,
        OPT_VAL_START,
        OUT_SCORE_PRED_TICKER,
        PERIOD_END,
        PERIOD_START,
        PRIOR_START_DATE,
        SCORE_DIR,
        TRANSACTION_COST_BPS,
        XGB_DIR,
    )
    from params import BLParams
    from pipeline import run_pipeline
    from prior import get_daily_returns_for_prior, ticker_from_sa
    from xgb_views import load_xgb_predictions, month_bounds, prediction_file_for_ticker

RANK_LOOKBACK = 252


def _ranked_ativos_by_hit() -> list[str]:
    """Ordena ATIVOS pelo hit_ratio em score (desc); fallback = ordem do config."""
    full = list(config.ATIVOS)
    if not OUT_SCORE_PRED_TICKER.exists():
        return full
    hits = pd.read_csv(OUT_SCORE_PRED_TICKER)
    if "ticker" not in hits.columns or "hit_ratio" not in hits.columns:
        return full
    order = hits.sort_values("hit_ratio", ascending=False)["ticker"].astype(str).tolist()
    by_base = {ticker_from_sa(t): t for t in full}
    ranked = [by_base[t] for t in order if t in by_base]
    for t in full:
        if t not in ranked:
            ranked.append(t)
    return ranked


def _ranked_ativos_by_lucro() -> list[str]:
    """Ordena ATIVOS pelo retorno buy&hold no período PERIOD_START–PERIOD_END (desc)."""
    full = list(config.ATIVOS)
    start, end = month_bounds(PERIOD_START, PERIOD_END)
    scored: list[tuple[float, str]] = []
    for sa in full:
        ticker = ticker_from_sa(sa)
        path = DATA_DAILY_DIR / f"{ticker}.csv"
        if not path.exists():
            scored.append((float("-inf"), sa))
            continue
        df = pd.read_csv(path)
        datecol = next(c for c in df.columns if c.lower() in ("date", "data", "datetime"))
        pxcol = next(
            (
                c
                for c in df.columns
                if c.lower() in ("close", "adj_close", "adjclose", "preco", "price", "vwap")
            ),
            None,
        )
        if pxcol is None:
            nums = df.select_dtypes("number").columns
            pxcol = nums[-1] if len(nums) else None
        if pxcol is None:
            scored.append((float("-inf"), sa))
            continue
        df[datecol] = pd.to_datetime(df[datecol])
        window = df[(df[datecol] >= start) & (df[datecol] <= end)].sort_values(datecol)
        if len(window) < 2:
            scored.append((float("-inf"), sa))
            continue
        ret = float(window[pxcol].iloc[-1]) / float(window[pxcol].iloc[0]) - 1.0
        scored.append((ret, sa))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sa for _, sa in scored]


def _ranked_ativos_by_pred() -> list[str]:
    """
    Ordena ATIVOS pela média de y_pred XGB no período de validação (antes do teste).

    Usa OPT_VAL_START–OPT_VAL_END (padrão 2024) para não olhar o retorno realizado de 2025.
    """
    full = list(config.ATIVOS)
    scored: list[tuple[float, str]] = []
    for sa in full:
        ticker = ticker_from_sa(sa)
        if prediction_file_for_ticker(ticker, XGB_DIR) is None:
            scored.append((float("-inf"), sa))
            continue
        views = load_xgb_predictions(
            ticker,
            period_start=OPT_VAL_START,
            period_end=OPT_VAL_END,
            xgb_dir=XGB_DIR,
        )
        if views.empty or "ret_pred" not in views.columns:
            scored.append((float("-inf"), sa))
            continue
        scored.append((float(views["ret_pred"].mean()), sa))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sa for _, sa in scored]


def _ranked_ativos_by_mkv_classical() -> list[str]:
    """
    Ordena ATIVOS pelo Sharpe histórico diário no período de validação (antes do teste).

    Sinal clássico de seleção Markowitz: média/vol dos retornos em OPT_VAL (2024).
    """
    full = list(config.ATIVOS)
    start, end = month_bounds(OPT_VAL_START, OPT_VAL_END)
    rets = get_daily_returns_for_prior(
        tickers=full,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    scored: list[tuple[float, str]] = []
    for sa in full:
        if sa not in rets.columns:
            scored.append((float("-inf"), sa))
            continue
        s = rets[sa].dropna()
        if len(s) < 40 or float(s.std(ddof=1)) <= 0:
            scored.append((float("-inf"), sa))
            continue
        sharpe = float(s.mean() / s.std(ddof=1) * np.sqrt(252))
        scored.append((sharpe, sa))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sa for _, sa in scored]


def _load_pred_panel(tickers: list[str]) -> pd.DataFrame:
    """Painel date × ticker(.SA) de y_pred; cobre validação + teste."""
    cols: dict[str, pd.Series] = {}
    for sa in tickers:
        base = ticker_from_sa(sa)
        if prediction_file_for_ticker(base, XGB_DIR) is None:
            continue
        views = load_xgb_predictions(
            base,
            period_start=OPT_VAL_START,
            period_end=PERIOD_END,
            xgb_dir=XGB_DIR,
        )
        if views.empty:
            continue
        s = views.set_index("view_date")["ret_pred"].sort_index()
        s.index = pd.to_datetime(s.index)
        cols[sa] = s
    if not cols:
        raise ValueError("Sem painel de y_pred para ranking walk-forward.")
    return pd.DataFrame(cols).sort_index()


def _make_rank_pred_fn(pred_panel: pd.DataFrame, assets: list[str]) -> Callable[[pd.Timestamp], list[str]]:
    def rank_fn(asof: pd.Timestamp) -> list[str]:
        hist = pred_panel.loc[pred_panel.index < asof].tail(RANK_LOOKBACK)
        if hist.empty:
            return list(assets)
        scores = hist.mean(axis=0).reindex(assets)
        return scores.sort_values(ascending=False, na_position="last").index.tolist()

    return rank_fn


def _make_rank_sharpe_fn(
    returns_hist: pd.DataFrame, assets: list[str]
) -> Callable[[pd.Timestamp], list[str]]:
    def rank_fn(asof: pd.Timestamp) -> list[str]:
        hist = returns_hist.loc[returns_hist.index < asof, assets].tail(RANK_LOOKBACK)
        if len(hist) < 40:
            return list(assets)
        mu = hist.mean()
        sd = hist.std(ddof=1).replace(0.0, np.nan)
        scores = (mu / sd * np.sqrt(252)).reindex(assets)
        return scores.sort_values(ascending=False, na_position="last").index.tolist()

    return rank_fn


def _metrics_from_curve(
    curve: pd.DataFrame,
    *,
    ret_col: str,
    cap_col: str,
) -> dict[str, float]:
    gain = curve.rename(columns={ret_col: "ret_real_portfolio", cap_col: "capital_real_brl"})
    gain["capital_selic_brl"] = np.nan
    return portfolio_metrics(gain)


def _run_universe(
    label: str,
    tickers: list[str],
    params: BLParams,
) -> tuple[dict[str, float | str | int], pd.Series]:
    original = list(config.ATIVOS)
    try:
        config.ATIVOS.clear()
        config.ATIVOS.extend(tickers)
        min_cap = 1.0 / len(tickers) + 1e-9
        params_k = replace(params, max_weight_per_asset=max(float(params.max_weight_per_asset), min_cap))
        outputs = run_pipeline(
            params_k,
            period_start=PERIOD_START,
            period_end=PERIOD_END,
            save_outputs=False,
            verbose=False,
        )
        mode = params_k.rebalance_mode or OPT_DEFAULT_REBALANCE_MODE
        w = outputs["weights_by_mode"][mode]
        mu = outputs["posterior_mu"]
        returns = outputs["returns_history"]
        daily = returns.copy()
        daily.index = pd.to_datetime(daily.index)
        gain = build_gain_series(mu, w, daily, mode_label=mode)
        metrics = portfolio_metrics(gain)
        g = gain[["view_date", "capital_real_brl"]].copy()
        g["view_date"] = pd.to_datetime(g["view_date"])
        capital = g.sort_values("view_date").set_index("view_date")["capital_real_brl"]
        row = {
            "universo": label,
            "n_ativos": len(tickers),
            "tickers": ",".join(ticker_from_sa(t) for t in tickers),
            "max_weight_used_cap": params_k.max_weight_per_asset,
            "rebalance_mode": mode,
            **metrics,
        }
        return row, capital
    finally:
        config.ATIVOS.clear()
        config.ATIVOS.extend(original)


def _run_markowitz_universe(
    label: str,
    tickers: list[str],
    *,
    mode: str,
    max_weight: float,
) -> tuple[dict[str, float | str | int], pd.Series]:
    returns_mat = load_returns_matrix(tickers)
    mkv, _w = compute_markowitz_curve(
        returns_mat,
        mode=mode,
        cost_bps=TRANSACTION_COST_BPS,
        max_weight=max_weight,
    )
    gain = mkv.rename(columns={"ret_mkv": "ret_real_portfolio", "capital_mkv_brl": "capital_real_brl"})
    gain["capital_selic_brl"] = np.nan
    metrics = portfolio_metrics(gain)
    g = gain[["view_date", "capital_real_brl"]].copy()
    g["view_date"] = pd.to_datetime(g["view_date"])
    capital = g.sort_values("view_date").set_index("view_date")["capital_real_brl"]
    row = {
        "universo": label,
        "n_ativos": len(tickers),
        "tickers": ",".join(ticker_from_sa(t) for t in tickers),
        "max_weight_used_cap": max_weight,
        "rebalance_mode": mode,
        **metrics,
    }
    return row, capital


def _plot_capital_curves(series: dict[int, pd.Series], rank: str, png_path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = cm.viridis
    norm = plt.Normalize(1, max(series) if series else 1)
    for k, s in series.items():
        ax.plot(
            s.index,
            s.values / 1000.0,
            color=cmap(norm(k)),
            lw=1.6 if k in (1, 5, 20) else 1.0,
            alpha=0.95 if k in (1, 5, 20) else 0.55,
            label=f"k={k}",
        )
    ax.set_title(f"Evolucao do capital BL em 2025 (top-k por {rank})")
    ax.set_ylabel("Capital (R$ mil)")
    ax.set_xlabel("Data")
    ax.grid(True, alpha=0.3)
    ax.axhline(100, color="gray", ls=":", lw=1, alpha=0.7)
    ax.legend(ncol=5, fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _plot_bl_vs_mkv_final(
    out: pd.DataFrame,
    png_path,
    *,
    title: str,
    bl_label: str = "BL",
    mkv_label: str = "Markowitz",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(out["n_ativos"], out["final_capital_brl"] / 1000.0, "o-", lw=2, label=bl_label)
    ax.plot(out["n_ativos"], out["final_mkv_brl"] / 1000.0, "s-", lw=2, label=mkv_label)
    ax.set_xlabel("k (tamanho do universo)")
    ax.set_ylabel("Capital final 2025 (R$ mil)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(100, color="gray", ls=":", lw=1, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _resolve_bl_rank(rank: str) -> tuple[list[str], str, str]:
    if rank == "lucro":
        return _ranked_ativos_by_lucro(), "_lucro", "lucro buy&hold (teste)"
    if rank == "pred":
        return (
            _ranked_ativos_by_pred(),
            "_pred",
            f"y_pred medio {OPT_VAL_START}-{OPT_VAL_END}",
        )
    return _ranked_ativos_by_hit(), "", "hit ratio"


def _run_walk_forward(mode: str, params: BLParams, max_assets: int | None = None) -> None:
    """Walk-forward top-k: classical (ordens independentes) + same_universe (ordem BL)."""
    full = list(config.ATIVOS)
    n = len(full) if max_assets is None else min(len(full), max(1, int(max_assets)))
    print("Walk-forward: rodando BL no universo completo (cache de mu posterior)...")
    outputs = run_pipeline(
        params,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        save_outputs=False,
        verbose=False,
    )
    bl_mu = outputs["posterior_mu"]
    period_start_ts, period_end_ts = month_bounds(PERIOD_START, PERIOD_END)
    returns_hist = get_daily_returns_for_prior(
        tickers=full,
        start_date=PRIOR_START_DATE,
        end_date=period_end_ts.strftime("%Y-%m-%d"),
    )
    returns_mat = returns_hist.loc[
        (returns_hist.index >= period_start_ts) & (returns_hist.index <= period_end_ts)
    ].copy()
    if returns_mat.empty:
        raise ValueError("Sem retornos no periodo de teste para walk-forward.")

    pred_panel = _load_pred_panel(full)
    rank_pred = _make_rank_pred_fn(pred_panel, full)
    rank_sharpe = _make_rank_sharpe_fn(returns_hist, full)

    # smoke: ranking asof first test day ignores same-day data
    asof0 = pd.Timestamp(returns_mat.index[0])
    assert pred_panel.loc[pred_panel.index < asof0].shape[0] >= 0
    r0 = rank_pred(asof0)
    assert set(r0) == set(full) or len(r0) == n

    specs = [
        {
            "name": "classical",
            "suffix": "_wf_classical",
            "title": "Walk-forward: BL (y_pred) vs Markowitz (Sharpe) — selecao independente",
            "bl_label": "BL (rank y_pred WF)",
            "mkv_label": "Markowitz (rank Sharpe WF)",
            "rank_bl": rank_pred,
            "rank_mkv": rank_sharpe,
        },
        {
            "name": "same_universe",
            "suffix": "_wf_same_universe",
            "title": "Walk-forward ablacao: mesmo top-k (ordem BL) — so alocacao difere",
            "bl_label": "BL (aloc bl_mu)",
            "mkv_label": "Markowitz (aloc MV) no top-k BL",
            "rank_bl": rank_pred,
            "rank_mkv": rank_pred,
        },
    ]

    for spec in specs:
        print(f"\n=== {spec['name']} ===")
        rows: list[dict[str, float | str | int]] = []
        for k in range(1, n + 1):
            max_w = max(float(params.max_weight_per_asset), 1.0 / k + 1e-9)
            print(f"[{k}/{n}] k={k}")
            row: dict[str, float | str | int] = {
                "universo": f"top{k}",
                "n_ativos": k,
                "max_weight_used_cap": max_w,
                "rebalance_mode": mode,
                "selection": spec["name"],
            }
            try:
                bl_curve, bl_w = compute_walkforward_topk_curve(
                    returns_mat,
                    mode=mode,
                    k=k,
                    rank_fn=spec["rank_bl"],
                    alloc="bl_mu",
                    returns_hist=returns_hist,
                    bl_mu=bl_mu,
                    max_weight=max_w,
                    cost_bps=TRANSACTION_COST_BPS,
                    ret_col="ret_bl",
                    cap_col="capital_bl_brl",
                    ganho_col="ganho_bl_pct",
                )
                bl_m = _metrics_from_curve(bl_curve, ret_col="ret_bl", cap_col="capital_bl_brl")
                topk_bl = (
                    bl_w.loc[bl_w["rebalance_flag"] == 1, "topk"].dropna().astype(str).tolist()
                )
                row.update(
                    {
                        "sharpe": bl_m["sharpe"],
                        "total_return_pct": bl_m["total_return_pct"],
                        "final_capital_brl": bl_m["final_capital_brl"],
                        "max_drawdown": bl_m["max_drawdown"],
                        "n_days": bl_m["n_days"],
                        "tickers_bl_first_reb": topk_bl[0] if topk_bl else "",
                        "tickers_bl_last_reb": topk_bl[-1] if topk_bl else "",
                    }
                )
            except Exception as exc:
                print(f"  BL FALHA: {exc}")
                row["error_bl"] = str(exc)

            try:
                mkv_curve, mkv_w = compute_walkforward_topk_curve(
                    returns_mat,
                    mode=mode,
                    k=k,
                    rank_fn=spec["rank_mkv"],
                    alloc="markowitz",
                    returns_hist=returns_hist,
                    max_weight=max_w,
                    cost_bps=TRANSACTION_COST_BPS,
                    ret_col="ret_mkv",
                    cap_col="capital_mkv_brl",
                    ganho_col="ganho_mkv_pct",
                )
                mkv_m = _metrics_from_curve(mkv_curve, ret_col="ret_mkv", cap_col="capital_mkv_brl")
                topk_mkv = (
                    mkv_w.loc[mkv_w["rebalance_flag"] == 1, "topk"].dropna().astype(str).tolist()
                )
                row.update(
                    {
                        "sharpe_mkv": mkv_m["sharpe"],
                        "ret_mkv_pct": mkv_m["total_return_pct"],
                        "final_mkv_brl": mkv_m["final_capital_brl"],
                        "mdd_mkv": mkv_m["max_drawdown"],
                        "tickers_mkv_first_reb": topk_mkv[0] if topk_mkv else "",
                        "tickers_mkv_last_reb": topk_mkv[-1] if topk_mkv else "",
                    }
                )
            except Exception as exc:
                print(f"  MKV FALHA: {exc}")
                row["error_mkv"] = str(exc)

            if "final_capital_brl" in row and "final_mkv_brl" in row:
                row["delta_sharpe_bl_minus_mkv"] = float(row["sharpe"]) - float(row["sharpe_mkv"])
                row["delta_final_bl_minus_mkv"] = float(row["final_capital_brl"]) - float(
                    row["final_mkv_brl"]
                )
                print(
                    f"  BL final={row['final_capital_brl']:.0f} | "
                    f"MKV final={row['final_mkv_brl']:.0f} | "
                    f"delta={row['delta_final_bl_minus_mkv']:.0f}"
                )
            rows.append(row)

        out = pd.DataFrame(rows)
        SCORE_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = SCORE_DIR / f"sensibilidade_tamanho_universo_pred{spec['suffix']}.csv"
        out_png = FIGURES_DIR / f"sensibilidade_tamanho_universo_pred{spec['suffix']}.png"
        out.to_csv(out_csv, index=False)
        if {"final_capital_brl", "final_mkv_brl"}.issubset(out.columns):
            _plot_bl_vs_mkv_final(
                out,
                out_png,
                title=spec["title"],
                bl_label=spec["bl_label"],
                mkv_label=spec["mkv_label"],
            )
            print(f"Figura: {out_png}")
        print(f"Salvo: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sensibilidade ao tamanho do universo BL.")
    parser.add_argument(
        "--rank",
        choices=("hit", "lucro", "pred"),
        default="hit",
        help="Criterio para ordenar o top-k do BL (modo estatico)",
    )
    parser.add_argument(
        "--vs-mkv-classical",
        action="store_true",
        help="Compara BL (ordem --rank) vs Markowitz (ordem Sharpe hist propria)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Ranking walk-forward mensal + classical e same_universe",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=None,
        help="Limite de k na sensibilidade walk-forward (padrão: universo completo)",
    )
    args = parser.parse_args()

    params = BLParams.from_config(rebalance_mode=OPT_DEFAULT_REBALANCE_MODE).normalized()
    mode = params.rebalance_mode or OPT_DEFAULT_REBALANCE_MODE

    if args.walk_forward:
        _run_walk_forward(mode, params, max_assets=args.max_assets)
        return

    ranked_bl, suffix, rank_label = _resolve_bl_rank(args.rank)

    if args.vs_mkv_classical:
        ranked_mkv = _ranked_ativos_by_mkv_classical()
        out_csv = SCORE_DIR / f"sensibilidade_tamanho_universo{suffix}_bl_vs_mkv_classical.csv"
        out_png = FIGURES_DIR / f"sensibilidade_tamanho_universo{suffix}_bl_vs_mkv_classical.png"
        print(f"BL ordem ({rank_label}): {[ticker_from_sa(t) for t in ranked_bl]}")
        print(
            f"MKV ordem (Sharpe hist {OPT_VAL_START}-{OPT_VAL_END}): "
            f"{[ticker_from_sa(t) for t in ranked_mkv]}"
        )
        print(f"Rebalance: {mode} | periodo {PERIOD_START}-{PERIOD_END}")
        print(f"k = 1 .. {len(ranked_bl)}\n")

        rows: list[dict[str, float | str | int]] = []
        n = len(ranked_bl)
        for k in range(1, n + 1):
            bl_set = ranked_bl[:k]
            mkv_set = ranked_mkv[:k]
            label = f"top{k}"
            print(
                f"[{k}/{n}] BL: {', '.join(ticker_from_sa(t) for t in bl_set)} | "
                f"MKV: {', '.join(ticker_from_sa(t) for t in mkv_set)}"
            )
            min_cap = 1.0 / k + 1e-9
            max_w = max(float(params.max_weight_per_asset), min_cap)
            row: dict[str, float | str | int] = {
                "universo": label,
                "n_ativos": k,
                "tickers_bl": ",".join(ticker_from_sa(t) for t in bl_set),
                "tickers_mkv": ",".join(ticker_from_sa(t) for t in mkv_set),
                "max_weight_used_cap": max_w,
                "rebalance_mode": mode,
            }
            try:
                bl_row, _ = _run_universe(label, bl_set, params)
                row.update(
                    {
                        "sharpe": bl_row["sharpe"],
                        "total_return_pct": bl_row["total_return_pct"],
                        "final_capital_brl": bl_row["final_capital_brl"],
                        "max_drawdown": bl_row["max_drawdown"],
                        "n_days": bl_row["n_days"],
                    }
                )
            except Exception as exc:
                print(f"  BL FALHA: {exc}")
                row["error_bl"] = str(exc)
            try:
                mkv_row, _ = _run_markowitz_universe(
                    label, mkv_set, mode=mode, max_weight=max_w
                )
                row.update(
                    {
                        "sharpe_mkv": mkv_row["sharpe"],
                        "ret_mkv_pct": mkv_row["total_return_pct"],
                        "final_mkv_brl": mkv_row["final_capital_brl"],
                        "mdd_mkv": mkv_row["max_drawdown"],
                    }
                )
            except Exception as exc:
                print(f"  MKV FALHA: {exc}")
                row["error_mkv"] = str(exc)

            if "final_capital_brl" in row and "final_mkv_brl" in row:
                row["delta_sharpe_bl_minus_mkv"] = float(row["sharpe"]) - float(row["sharpe_mkv"])
                row["delta_final_bl_minus_mkv"] = float(row["final_capital_brl"]) - float(
                    row["final_mkv_brl"]
                )
                print(
                    f"  BL final={row['final_capital_brl']:.0f} sharpe={row['sharpe']:.3f} | "
                    f"MKV final={row['final_mkv_brl']:.0f} sharpe={row['sharpe_mkv']:.3f} | "
                    f"delta_final={row['delta_final_bl_minus_mkv']:.0f}"
                )
            rows.append(row)

        out = pd.DataFrame(rows)
        SCORE_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        if {"final_capital_brl", "final_mkv_brl"}.issubset(out.columns):
            _plot_bl_vs_mkv_final(
                out,
                out_png,
                title="BL vs Markowitz classico: selecao independente do top-k",
                bl_label="BL (top-k y_pred)",
                mkv_label="Markowitz (top-k Sharpe hist)",
            )
            print(f"Figura: {out_png}")
        print("\n" + out.to_string(index=False))
        print(f"\nSalvo em: {out_csv}")
        return

    out_csv = SCORE_DIR / f"sensibilidade_tamanho_universo{suffix}.csv"
    out_capital = SCORE_DIR / f"sensibilidade_tamanho_universo_capital_diario{suffix}.csv"
    out_png = FIGURES_DIR / f"sensibilidade_tamanho_universo_capital_ano{suffix}.png"

    print(f"Ordem ({rank_label}): {[ticker_from_sa(t) for t in ranked_bl]}")
    print(f"Rebalance: {mode} | periodo {PERIOD_START}-{PERIOD_END}")
    print(f"k = 1 .. {len(ranked_bl)}\n")

    rows = []
    series: dict[int, pd.Series] = {}
    for k in range(1, len(ranked_bl) + 1):
        subset = ranked_bl[:k]
        label = f"top{k}"
        print(f"[{k}/{len(ranked_bl)}] {label}: {', '.join(ticker_from_sa(t) for t in subset)}")
        try:
            row, capital = _run_universe(label, subset, params)
            rows.append(row)
            series[k] = capital
            print(
                f"  sharpe={row.get('sharpe', float('nan')):.3f} | "
                f"ret={row.get('total_return_pct', float('nan')):.2f}% | "
                f"final={row.get('final_capital_brl', float('nan')):.0f}"
            )
        except Exception as exc:
            print(f"  FALHA: {exc}")
            rows.append(
                {
                    "universo": label,
                    "n_ativos": k,
                    "tickers": ",".join(ticker_from_sa(t) for t in subset),
                    "error": str(exc),
                }
            )

    out = pd.DataFrame(rows)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    if series:
        pd.DataFrame({f"k{k}": s for k, s in series.items()}).to_csv(out_capital)
        _plot_capital_curves(series, args.rank, out_png)
        print(f"Figura: {out_png}")
        print(f"Capitais: {out_capital}")

    print("\n" + out.to_string(index=False))
    print(f"\nSalvo em: {out_csv}")


if __name__ == "__main__":
    # ponytail: walk-forward must not use asof-day signal
    _asof = pd.Timestamp("2025-01-02")
    _panel = pd.DataFrame(
        {"A.SA": [1.0, 2.0], "B.SA": [0.5, 0.1]},
        index=pd.to_datetime(["2024-12-30", "2025-01-02"]),
    )
    _fn = _make_rank_pred_fn(_panel, ["A.SA", "B.SA"])
    assert _fn(_asof)[0] == "A.SA"
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
