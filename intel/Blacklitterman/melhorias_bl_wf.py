"""
Melhorias BL walk-forward (pacote completo):

1. BL reestimado só no top-k a cada rebalance
2. k escolhido ex-ante na validação (2024)
3. Views mais limpas (hard prior / news_count tunáveis)
4. Regularização: max_w=min(cap, c/k), shrink μ→π, rolling prior
5. Híbrido: seleção y_pred + alocação Markowitz
6. Comparação OOS 2025: bl_subset vs hybrid vs markowitz

Uso:
    python -m intel.Blacklitterman.melhorias_bl_wf
    python -m intel.Blacklitterman.melhorias_bl_wf --trials 20
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd

try:
    from .backtest import portfolio_metrics
    from .comparar_bl_vs_markowitz import (
        ROLLING_WINDOW,
        MIN_OBS,
        RIDGE,
        _curve_from_weights,
        _mode_period_key,
        _mv_long_only,
    )
    from .config import (
        ATIVOS,
        FIGURES_DIR,
        OPT_DEFAULT_REBALANCE_MODE,
        OPT_VAL_END,
        OPT_VAL_START,
        OPTIMIZE_DIR,
        PERIOD_END,
        PERIOD_START,
        PRIOR_START_DATE,
        RISK_FREE_ANNUAL,
        SCORE_DIR,
        TRANSACTION_COST_BPS,
        XGB_DIR,
    )
    from .omega_calc import calculate_omega
    from .params import BLParams
    from .prior import (
        compute_pi_and_prior_uncertainty,
        get_daily_returns_for_prior,
        get_market_cap_weights,
        ticker_from_sa,
        ticker_to_sa,
    )
    from .sentiment import load_daily_sentiment_for_ticker
    from .weights import long_only_capped_weights
    from .xgb_views import (
        add_ret_scale,
        align_q_to_daily,
        build_q_from_cached_views,
        load_xgb_predictions,
        month_bounds,
        prediction_file_for_ticker,
    )
except ImportError:
    from backtest import portfolio_metrics
    from comparar_bl_vs_markowitz import (
        ROLLING_WINDOW,
        MIN_OBS,
        RIDGE,
        _curve_from_weights,
        _mode_period_key,
        _mv_long_only,
    )
    from config import (
        ATIVOS,
        FIGURES_DIR,
        OPT_DEFAULT_REBALANCE_MODE,
        OPT_VAL_END,
        OPT_VAL_START,
        OPTIMIZE_DIR,
        PERIOD_END,
        PERIOD_START,
        PRIOR_START_DATE,
        RISK_FREE_ANNUAL,
        SCORE_DIR,
        TRANSACTION_COST_BPS,
        XGB_DIR,
    )
    from omega_calc import calculate_omega
    from params import BLParams
    from prior import (
        compute_pi_and_prior_uncertainty,
        get_daily_returns_for_prior,
        get_market_cap_weights,
        ticker_from_sa,
        ticker_to_sa,
    )
    from sentiment import load_daily_sentiment_for_ticker
    from weights import long_only_capped_weights
    from xgb_views import (
        add_ret_scale,
        align_q_to_daily,
        build_q_from_cached_views,
        load_xgb_predictions,
        month_bounds,
        prediction_file_for_ticker,
    )

RANK_LOOKBACK = 252
K_CANDIDATES = (3, 4, 5, 6, 7, 8, 10)
EPS = 1e-8


@dataclass(frozen=True)
class ImproveParams:
    bl: BLParams
    mu_shrink: float = 0.25
    weight_cap_mult: float = 2.0  # max_w = min(bl.max_weight, mult/k)
    turnover_mix: float = 0.15  # w = (1-m)*w_new + m*w_old


@dataclass
class Cache:
    assets: list[str]
    returns_hist: pd.DataFrame
    market_weights: pd.Series
    pred_panel: pd.DataFrame
    views_by_ticker: dict[str, pd.DataFrame]
    sentiment_by_ticker: dict[str, pd.DataFrame]


def _effective_max_w(base_cap: float, k: int, mult: float) -> float:
    # inviável se max_w * k < 1
    raw = min(float(base_cap), float(mult) / max(k, 1))
    return max(raw, 1.0 / k + 1e-9)


def _load_pred_panel(assets: list[str], period_start: str, period_end: str) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    for sa in assets:
        base = ticker_from_sa(sa)
        if prediction_file_for_ticker(base, XGB_DIR) is None:
            continue
        views = load_xgb_predictions(
            base, period_start=period_start, period_end=period_end, xgb_dir=XGB_DIR
        )
        if views.empty:
            continue
        s = views.set_index("view_date")["ret_pred"].sort_index()
        s.index = pd.to_datetime(s.index)
        cols[sa] = s
    if not cols:
        raise ValueError("Sem y_pred para ranking.")
    return pd.DataFrame(cols).sort_index()


def build_cache(
    *,
    hist_start: str = PRIOR_START_DATE,
    pred_start: str = OPT_VAL_START,
    pred_end: str = PERIOD_END,
) -> Cache:
    assets = list(ATIVOS)
    _, end_ts = month_bounds(PERIOD_START, PERIOD_END)
    returns_hist = get_daily_returns_for_prior(
        tickers=assets,
        start_date=hist_start,
        end_date=end_ts.strftime("%Y-%m-%d"),
    )
    market_weights = get_market_cap_weights(assets)
    pred_panel = _load_pred_panel(assets, pred_start, pred_end)

    views_by_ticker: dict[str, pd.DataFrame] = {}
    sentiment_by_ticker: dict[str, pd.DataFrame] = {}
    for sa in assets:
        base = ticker_from_sa(sa)
        views = load_xgb_predictions(
            base, period_start=pred_start, period_end=pred_end, xgb_dir=XGB_DIR
        )
        if views.empty:
            continue
        views_by_ticker[base] = add_ret_scale(views)
        sentiment_by_ticker[base] = load_daily_sentiment_for_ticker(base, match_company=True)
    return Cache(
        assets=assets,
        returns_hist=returns_hist.sort_index(),
        market_weights=market_weights,
        pred_panel=pred_panel,
        views_by_ticker=views_by_ticker,
        sentiment_by_ticker=sentiment_by_ticker,
    )


def build_q_omega(cache: Cache, params: BLParams) -> pd.DataFrame:
    q_long = build_q_from_cached_views(
        cache.views_by_ticker,
        cache.sentiment_by_ticker,
        alpha=params.alpha_q,
        news_count_min=params.news_count_min,
    )
    q_long = align_q_to_daily(q_long)
    omega = calculate_omega(
        q_long,
        tau=params.tau,
        err_window=params.err_window,
        err_min_periods=params.err_min_periods,
        w_model_conf=params.w_model_conf,
        w_news_conf=params.w_news_conf,
        confidence_floor=params.confidence_floor,
        confidence_cap=params.confidence_cap,
        news_count_min=params.news_count_min,
        news_omega_boost=params.news_omega_boost,
        hard_prior_when_low_news=params.hard_prior_when_low_news,
    )
    return (
        q_long.merge(
            omega[
                [
                    "view_date",
                    "ticker",
                    "omega",
                    "omega_base",
                    "omega_boost",
                    "view_confidence",
                ]
            ],
            on=["view_date", "ticker"],
            how="inner",
        )
        .assign(view_date=lambda d: pd.to_datetime(d["view_date"]))
        .sort_values(["view_date", "ticker"])
        .reset_index(drop=True)
    )


def make_rank_pred(cache: Cache) -> Callable[[pd.Timestamp], list[str]]:
    assets = cache.assets
    panel = cache.pred_panel

    def rank_fn(asof: pd.Timestamp) -> list[str]:
        hist = panel.loc[panel.index < asof].tail(RANK_LOOKBACK)
        if hist.empty:
            return list(assets)
        scores = hist.mean(axis=0).reindex(assets)
        return scores.sort_values(ascending=False, na_position="last").index.tolist()

    return rank_fn


def make_rank_sharpe(cache: Cache) -> Callable[[pd.Timestamp], list[str]]:
    assets = cache.assets
    rets = cache.returns_hist

    def rank_fn(asof: pd.Timestamp) -> list[str]:
        hist = rets.loc[rets.index < asof, assets].tail(RANK_LOOKBACK)
        if len(hist) < 40:
            return list(assets)
        mu = hist.mean()
        sd = hist.std(ddof=1).replace(0.0, np.nan)
        scores = (mu / sd * np.sqrt(252)).reindex(assets)
        return scores.sort_values(ascending=False, na_position="last").index.tolist()

    return rank_fn


def _views_asof(q_omega: pd.DataFrame, asof: pd.Timestamp, subset: list[str]) -> pd.DataFrame:
    """Última view de cada ticker com data <= asof, filtrada ao subset."""
    bases = {ticker_from_sa(s): s for s in subset}
    q = q_omega[q_omega["view_date"] <= asof].copy()
    if q.empty:
        return q
    q["sa"] = q["ticker"].map(lambda t: ticker_to_sa(str(t)))
    q = q[q["sa"].isin(subset)]
    if q.empty:
        return q
    # última observação por ativo
    q = q.sort_values("view_date").groupby("sa", as_index=False).tail(1)
    return q


def bl_subset_weights(
    subset: list[str],
    asof: pd.Timestamp,
    q_omega: pd.DataFrame,
    cache: Cache,
    ip: ImproveParams,
    max_weight: float,
) -> np.ndarray:
    """BL completo só no subset, prior rolling (< asof), shrink μ→π."""
    params = ip.bl
    hist = cache.returns_hist.loc[cache.returns_hist.index < asof, subset].dropna(how="any")
    n = len(subset)
    if len(hist) < max(params.rolling_min_obs, MIN_OBS):
        return np.full(n, 1.0 / n)

    mw = cache.market_weights.reindex(subset).fillna(0.0)
    if float(mw.sum()) <= 0:
        mw = pd.Series(1.0 / n, index=subset)
    else:
        mw = mw / float(mw.sum())

    pi, sigma, _, delta = compute_pi_and_prior_uncertainty(
        returns=hist,
        market_weights=mw,
        risk_free_annual=RISK_FREE_ANNUAL,
        tau=params.tau,
    )
    pi = pi.reindex(subset).fillna(0.0)
    sigma = sigma.reindex(index=subset, columns=subset).fillna(0.0)
    sigma_np = sigma.to_numpy(dtype=float) + RIDGE * np.eye(n)
    pi_vec = pi.to_numpy(dtype=float).reshape(n, 1)
    inv_tau_sigma = np.linalg.pinv(params.tau * sigma_np)

    day = _views_asof(q_omega, asof, subset)
    if day.empty:
        mu = pi_vec.copy()
    else:
        valid: list[tuple[str, float, float]] = []
        for _, row in day.iterrows():
            sa = str(row["sa"])
            omega = max(float(row["omega"]), EPS)
            valid.append((sa, float(row["Q"]), omega))
        if not valid:
            mu = pi_vec.copy()
        else:
            k_v = len(valid)
            p = np.zeros((k_v, n), dtype=float)
            q_vec = np.zeros((k_v, 1), dtype=float)
            omega_diag = np.zeros(k_v, dtype=float)
            idx = {s: i for i, s in enumerate(subset)}
            for i, (sa, q_val, omega_val) in enumerate(valid):
                p[i, idx[sa]] = 1.0
                q_vec[i, 0] = q_val
                omega_diag[i] = omega_val
            inv_omega = np.diag(1.0 / omega_diag)
            a_mat = inv_tau_sigma + p.T @ inv_omega @ p
            b_vec = inv_tau_sigma @ pi_vec + p.T @ inv_omega @ q_vec
            mu = np.linalg.pinv(a_mat) @ b_vec

    lam = float(np.clip(ip.mu_shrink, 0.0, 1.0))
    mu = (1.0 - lam) * mu + lam * pi_vec
    raw = np.linalg.pinv(float(delta) * sigma_np) @ mu.ravel()
    if not np.isfinite(raw).all() or abs(float(raw.sum())) <= EPS:
        return np.full(n, 1.0 / n)
    return long_only_capped_weights(raw / float(raw.sum()), max_weight)


def mkv_subset_weights(
    subset: list[str],
    asof: pd.Timestamp,
    cache: Cache,
    max_weight: float,
) -> np.ndarray:
    hist = (
        cache.returns_hist.loc[cache.returns_hist.index < asof, subset]
        .tail(ROLLING_WINDOW)
        .dropna(how="any")
    )
    n = len(subset)
    if len(hist) < MIN_OBS:
        return np.full(n, 1.0 / n)
    mu = hist.mean().to_numpy(dtype=float)
    sigma = hist.cov().to_numpy(dtype=float)
    return _mv_long_only(mu, sigma, max_weight)


def run_strategy_curve(
    *,
    cache: Cache,
    q_omega: pd.DataFrame | None,
    rank_fn: Callable[[pd.Timestamp], list[str]],
    alloc: str,
    k: int,
    ip: ImproveParams,
    period_start: str,
    period_end: str,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """alloc: bl_subset | markowitz | hybrid (rank via rank_fn, alloc MV)."""
    start_ts, end_ts = month_bounds(period_start, period_end)
    returns_mat = cache.returns_hist.loc[
        (cache.returns_hist.index >= start_ts) & (cache.returns_hist.index <= end_ts),
        cache.assets,
    ].copy()
    if returns_mat.empty:
        raise ValueError(f"Sem retornos em {period_start}-{period_end}")

    dates = returns_mat.index.to_series().reset_index(drop=True)
    assets = list(returns_mat.columns)
    n = len(assets)
    asset_i = {a: i for i, a in enumerate(assets)}
    max_w = _effective_max_w(ip.bl.max_weight_per_asset, k, ip.weight_cap_mult)

    reb_key = _mode_period_key(dates, mode)
    rebalance_flag = ~reb_key.duplicated()
    weights = np.zeros((len(dates), n), dtype=float)
    current = np.zeros(n, dtype=float)
    current[: min(k, n)] = 1.0 / min(k, n)
    mix = float(np.clip(ip.turnover_mix, 0.0, 0.95))

    for i in range(len(dates)):
        if bool(rebalance_flag.iloc[i]):
            asof = pd.Timestamp(dates.iloc[i])
            ranked = [t for t in rank_fn(asof) if t in asset_i][:k]
            while len(ranked) < k:
                for a in assets:
                    if a not in ranked:
                        ranked.append(a)
                    if len(ranked) >= k:
                        break
            if alloc == "bl_subset":
                assert q_omega is not None
                w_k = bl_subset_weights(ranked, asof, q_omega, cache, ip, max_w)
            else:
                w_k = mkv_subset_weights(ranked, asof, cache, max_w)
            new_w = np.zeros(n, dtype=float)
            for j, a in enumerate(ranked):
                new_w[asset_i[a]] = float(w_k[j])
            if mix > 0 and float(current.sum()) > EPS:
                current = (1.0 - mix) * new_w + mix * current
                # renormaliza long-only no suporte do new_w (ativos do mês)
                mask = new_w > 0
                if mask.any():
                    current = np.where(mask, np.maximum(current, 0.0), 0.0)
                    s = float(current.sum())
                    current = current / s if s > EPS else new_w
                    # reaplica teto só nos ativos ativos
                    active = np.where(mask)[0]
                    current[active] = long_only_capped_weights(current[active], max_w)
            else:
                current = new_w
        weights[i] = current

    curve = _curve_from_weights(
        returns_mat,
        weights,
        mode,
        "ret_port",
        "capital_port_brl",
        "ganho_port_pct",
        cost_bps=TRANSACTION_COST_BPS,
    )
    gain = curve.rename(
        columns={"ret_port": "ret_real_portfolio", "capital_port_brl": "capital_real_brl"}
    )
    gain["capital_selic_brl"] = np.nan
    metrics = portfolio_metrics(gain)
    # turnover médio aproximado nos rebalances
    reb_idx = np.flatnonzero(rebalance_flag.to_numpy())
    turns = []
    for j in range(1, len(reb_idx)):
        a, b = reb_idx[j - 1], reb_idx[j]
        turns.append(0.5 * float(np.abs(weights[b] - weights[a]).sum()))
    metrics["avg_turnover"] = float(np.mean(turns)) if turns else 0.0
    metrics["k"] = float(k)
    metrics["max_weight_used"] = float(max_w)
    return curve, metrics


def objective_score(metrics: dict[str, float], *, mdd_penalty: float = 0.5) -> float:
    sharpe = metrics.get("sharpe", float("nan"))
    mdd = metrics.get("max_drawdown", float("nan"))
    if not np.isfinite(sharpe):
        return -1e6
    penalty = mdd_penalty * abs(min(float(mdd), 0.0)) if np.isfinite(mdd) else 0.0
    return float(sharpe + penalty)


def suggest_improve(trial: optuna.Trial, mode: str) -> ImproveParams:
    w_model = trial.suggest_float("w_model_conf", 0.3, 0.9)
    err_window = trial.suggest_int("err_window", 10, 28)
    bl = BLParams(
        alpha_q=trial.suggest_float("alpha_q", 0.15, 0.85),
        tau=trial.suggest_float("tau", 0.005, 0.06, log=True),
        max_weight_per_asset=trial.suggest_float("max_weight_per_asset", 0.22, 0.45),
        w_model_conf=w_model,
        w_news_conf=1.0 - w_model,
        confidence_floor=trial.suggest_float("confidence_floor", 0.01, 0.18),
        confidence_cap=trial.suggest_float("confidence_cap", 0.70, 0.98),
        err_window=err_window,
        err_min_periods=trial.suggest_int("err_min_periods", 3, min(10, err_window)),
        rolling_min_obs=trial.suggest_int("rolling_min_obs", 60, 120, step=10),
        use_rolling_prior=True,
        rebalance_mode=mode,
        news_count_min=trial.suggest_int("news_count_min", 2, 5),
        news_omega_boost=trial.suggest_float("news_omega_boost", 2.0, 6.0),
        hard_prior_when_low_news=trial.suggest_categorical(
            "hard_prior_when_low_news", [True, False]
        ),
    ).normalized()
    return ImproveParams(
        bl=bl,
        mu_shrink=trial.suggest_float("mu_shrink", 0.05, 0.55),
        weight_cap_mult=trial.suggest_float("weight_cap_mult", 1.5, 3.0),
        turnover_mix=trial.suggest_float("turnover_mix", 0.0, 0.35),
    )


def select_k_exante(
    cache: Cache,
    ip: ImproveParams,
    q_omega: pd.DataFrame,
    *,
    mode: str,
    val_start: str,
    val_end: str,
) -> tuple[int, pd.DataFrame]:
    rank_pred = make_rank_pred(cache)
    rows = []
    best_k, best_score = K_CANDIDATES[0], -1e9
    for k in K_CANDIDATES:
        _, m = run_strategy_curve(
            cache=cache,
            q_omega=q_omega,
            rank_fn=rank_pred,
            alloc="bl_subset",
            k=k,
            ip=ip,
            period_start=val_start,
            period_end=val_end,
            mode=mode,
        )
        score = objective_score(m)
        rows.append({"k": k, "score": score, **m})
        if score > best_score:
            best_score, best_k = score, k
        print(
            f"  val k={k}: sharpe={m['sharpe']:.3f} final={m['final_capital_brl']:.0f} "
            f"score={score:.3f}"
        )
    return best_k, pd.DataFrame(rows)


def apply_best_to_config(ip: ImproveParams, k_star: int, out_json: Path) -> None:
    payload = {
        "k_star": k_star,
        "mu_shrink": ip.mu_shrink,
        "weight_cap_mult": ip.weight_cap_mult,
        "turnover_mix": ip.turnover_mix,
        "bl": ip.bl.to_dict(),
        "notes": "Params do walk-forward melhorado (Optuna na val 2024).",
    }
    OPTIMIZE_DIR.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_config_extras(ip: ImproveParams) -> None:
    import re

    try:
        from .aplicar_melhores_params import write_config_py
    except ImportError:
        from aplicar_melhores_params import write_config_py
    write_config_py(ip.bl)
    cfg = Path(__file__).resolve().parent / "config.py"
    text = cfg.read_text(encoding="utf-8")
    extras = {
        "NEWS_COUNT_MIN": ip.bl.news_count_min,
        "NEWS_OMEGA_BOOST": ip.bl.news_omega_boost,
        "HARD_PRIOR_WHEN_LOW_NEWS": ip.bl.hard_prior_when_low_news,
    }
    for key, value in extras.items():
        if isinstance(value, bool):
            replacement = f"{key} = {value}"
        elif isinstance(value, float):
            replacement = f"{key} = {value:.6g}"
        else:
            replacement = f"{key} = {value}"
        pattern = rf"^{re.escape(key)}\s*=\s*.+$"
        text = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
    cfg.write_text(text, encoding="utf-8")


def compare_oos(
    cache: Cache,
    ip: ImproveParams,
    k: int,
    *,
    mode: str,
    period_start: str,
    period_end: str,
) -> pd.DataFrame:
    q_omega = build_q_omega(cache, ip.bl)
    rank_pred = make_rank_pred(cache)
    rank_sharpe = make_rank_sharpe(cache)
    specs = [
        ("bl_subset", rank_pred, "bl_subset"),
        ("hybrid_blrank_mkv", rank_pred, "markowitz"),
        ("markowitz_classical", rank_sharpe, "markowitz"),
    ]
    rows = []
    for name, rank_fn, alloc in specs:
        _, m = run_strategy_curve(
            cache=cache,
            q_omega=q_omega if alloc == "bl_subset" else None,
            rank_fn=rank_fn,
            alloc=alloc,
            k=k,
            ip=ip,
            period_start=period_start,
            period_end=period_end,
            mode=mode,
        )
        rows.append({"strategy": name, **m})
        print(
            f"  {name}: sharpe={m['sharpe']:.3f} final={m['final_capital_brl']:.0f} "
            f"mdd={m['max_drawdown']:.3f} turn={m['avg_turnover']:.3f}"
        )
    return pd.DataFrame(rows)


def sweep_k_oos(
    cache: Cache,
    ip: ImproveParams,
    *,
    mode: str,
    period_start: str,
    period_end: str,
) -> pd.DataFrame:
    q_omega = build_q_omega(cache, ip.bl)
    rank_pred = make_rank_pred(cache)
    rank_sharpe = make_rank_sharpe(cache)
    rows = []
    for k in range(1, len(cache.assets) + 1):
        for name, rank_fn, alloc in (
            ("bl_subset", rank_pred, "bl_subset"),
            ("hybrid", rank_pred, "markowitz"),
            ("mkv", rank_sharpe, "markowitz"),
        ):
            _, m = run_strategy_curve(
                cache=cache,
                q_omega=q_omega if alloc == "bl_subset" else None,
                rank_fn=rank_fn,
                alloc=alloc,
                k=k,
                ip=ip,
                period_start=period_start,
                period_end=period_end,
                mode=mode,
            )
            rows.append({"strategy": name, "n_ativos": k, **m})
        print(f"  sweep k={k} done")
    return pd.DataFrame(rows)


def plot_sweep(df: pd.DataFrame, k_star: int, png: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, style in (
        ("bl_subset", "o-"),
        ("hybrid", "s--"),
        ("mkv", "^-."),
    ):
        g = df[df["strategy"] == name].sort_values("n_ativos")
        ax.plot(g["n_ativos"], g["final_capital_brl"] / 1000.0, style, lw=2, label=name)
    ax.axvline(k_star, color="gray", ls=":", lw=1.5, label=f"k*={k_star} (ex-ante)")
    ax.axhline(100, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("k")
    ax.set_ylabel("Capital final (R$ mil)")
    ax.set_title("OOS walk-forward melhorado: BL subset vs hybrid vs Markowitz")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150)
    plt.close(fig)


def load_best_improve(path: Path | None = None) -> tuple[ImproveParams, int]:
    path = path or (OPTIMIZE_DIR / "melhorias_bl_wf_best.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    ip = ImproveParams(
        bl=BLParams(**raw["bl"]).normalized(),
        mu_shrink=float(raw["mu_shrink"]),
        weight_cap_mult=float(raw["weight_cap_mult"]),
        turnover_mix=float(raw["turnover_mix"]),
    )
    return ip, int(raw["k_star"])


def report_kstar_capital(
    *,
    mode: str | None = None,
    best_path: Path | None = None,
) -> None:
    """Curvas de capital OOS 2025 congeladas em k* (BL vs hybrid vs Markowitz)."""
    ip, k_star = load_best_improve(best_path)
    mode = mode or ip.bl.rebalance_mode or OPT_DEFAULT_REBALANCE_MODE
    print(f"Params congelados | k*={k_star} | mode={mode}")
    cache = build_cache()
    q_omega = build_q_omega(cache, ip.bl)
    rank_pred = make_rank_pred(cache)
    rank_sharpe = make_rank_sharpe(cache)

    specs = [
        ("bl_subset", rank_pred, "bl_subset"),
        ("hybrid", rank_pred, "markowitz"),
        ("markowitz", rank_sharpe, "markowitz"),
    ]
    curves: dict[str, pd.Series] = {}
    rows: list[dict[str, float | str | int]] = []
    for name, rank_fn, alloc in specs:
        curve, m = run_strategy_curve(
            cache=cache,
            q_omega=q_omega if alloc == "bl_subset" else None,
            rank_fn=rank_fn,
            alloc=alloc,
            k=k_star,
            ip=ip,
            period_start=PERIOD_START,
            period_end=PERIOD_END,
            mode=mode,
        )
        s = curve.copy()
        s["view_date"] = pd.to_datetime(s["view_date"])
        curves[name] = s.set_index("view_date")["capital_port_brl"].sort_index()
        rows.append({"strategy": name, "k_star": k_star, **m})
        print(
            f"  {name}: sharpe={m['sharpe']:.3f} final={m['final_capital_brl']:.0f} "
            f"mdd={m['max_drawdown']:.3f} turn={m['avg_turnover']:.3f}"
        )

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_metrics = SCORE_DIR / "melhorias_bl_wf_kstar_metricas.csv"
    out_cap = SCORE_DIR / "melhorias_bl_wf_kstar_capital_diario.csv"
    out_png = FIGURES_DIR / "melhorias_bl_wf_kstar_capital.png"

    pd.DataFrame(rows).to_csv(out_metrics, index=False)
    pd.DataFrame({k: v for k, v in curves.items()}).to_csv(out_cap)

    fig, ax = plt.subplots(figsize=(12, 6))
    styles = {"bl_subset": "-", "hybrid": "--", "markowitz": "-."}
    for name, ser in curves.items():
        ax.plot(ser.index, ser.values / 1000.0, ls=styles.get(name, "-"), lw=2.0, label=name)
    ax.axhline(100, color="gray", ls=":", lw=1, alpha=0.7)
    ax.set_title(f"Capital OOS 2025 — k*={k_star} (params congelados na val 2024)")
    ax.set_ylabel("Capital (R$ mil)")
    ax.set_xlabel("Data")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"\nMetricas: {out_metrics}")
    print(f"Capitais: {out_cap}")
    print(f"Figura: {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pacote de melhorias BL walk-forward")
    parser.add_argument("--trials", type=int, default=20, help="Trials Optuna na val 2024")
    parser.add_argument("--mode", default=OPT_DEFAULT_REBALANCE_MODE)
    parser.add_argument(
        "--skip-optuna",
        action="store_true",
        help="Usa params atuais + hard_prior; so escolhe k e avalia OOS",
    )
    parser.add_argument(
        "--report-kstar",
        action="store_true",
        help="Gera curvas de capital OOS so com k* congelado (lê best.json)",
    )
    args = parser.parse_args()

    if args.report_kstar:
        report_kstar_capital(mode=args.mode)
        return

    mode = args.mode

    print("1) Cache (retornos, views, sentimento, market cap)...")
    cache = build_cache()

    if args.skip_optuna:
        base = BLParams.from_config(rebalance_mode=mode).normalized()
        base = replace(
            base,
            hard_prior_when_low_news=True,
            news_count_min=max(base.news_count_min, 3),
        )
        ip = ImproveParams(bl=base, mu_shrink=0.25, weight_cap_mult=2.0, turnover_mix=0.15)
        print("2) Selecionando k* na validacao (sem Optuna)...")
        q_omega = build_q_omega(cache, ip.bl)
        k_star, val_k = select_k_exante(
            cache, ip, q_omega, mode=mode, val_start=OPT_VAL_START, val_end=OPT_VAL_END
        )
    else:
        print(f"2) Optuna walk-forward BL subset na val ({args.trials} trials)...")
        study = optuna.create_study(direction="maximize", study_name="bl_wf_improve")

        def _obj(trial: optuna.Trial) -> float:
            try:
                ip_t = suggest_improve(trial, mode)
                k = int(trial.suggest_categorical("k", list(K_CANDIDATES)))
                q_omega = build_q_omega(cache, ip_t.bl)
                _, m = run_strategy_curve(
                    cache=cache,
                    q_omega=q_omega,
                    rank_fn=make_rank_pred(cache),
                    alloc="bl_subset",
                    k=k,
                    ip=ip_t,
                    period_start=OPT_VAL_START,
                    period_end=OPT_VAL_END,
                    mode=mode,
                )
                trial.set_user_attr("k", k)
                trial.set_user_attr("sharpe", m.get("sharpe"))
                trial.set_user_attr("final", m.get("final_capital_brl"))
                trial.set_user_attr(
                    "improve",
                    {
                        "mu_shrink": ip_t.mu_shrink,
                        "weight_cap_mult": ip_t.weight_cap_mult,
                        "turnover_mix": ip_t.turnover_mix,
                        "bl": ip_t.bl.to_dict(),
                    },
                )
                return objective_score(m)
            except Exception as exc:
                trial.set_user_attr("error", str(exc))
                return -1e6

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(_obj, n_trials=args.trials, show_progress_bar=False)
        best = study.best_trial
        improv = best.user_attrs["improve"]
        ip = ImproveParams(
            bl=BLParams(**improv["bl"]).normalized(),
            mu_shrink=float(improv["mu_shrink"]),
            weight_cap_mult=float(improv["weight_cap_mult"]),
            turnover_mix=float(improv["turnover_mix"]),
        )
        k_star = int(best.user_attrs["k"])
        print(
            f"  best trial={best.number} score={best.value:.3f} k*={k_star} "
            f"sharpe_val={best.user_attrs.get('sharpe')}"
        )
        val_k = pd.DataFrame([{"k": k_star, "score": best.value, "source": "optuna_best"}])

    print(f"\n3) k* ex-ante = {k_star}")
    print("4) Comparacao OOS 2025 (k*)...")
    oos = compare_oos(
        cache, ip, k_star, mode=mode, period_start=PERIOD_START, period_end=PERIOD_END
    )

    print("5) Sweep k OOS (curvas)...")
    sweep = sweep_k_oos(
        cache, ip, mode=mode, period_start=PERIOD_START, period_end=PERIOD_END
    )

    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_oos = SCORE_DIR / "melhorias_bl_wf_oos_kstar.csv"
    out_sweep = SCORE_DIR / "melhorias_bl_wf_oos_sweep.csv"
    out_val = SCORE_DIR / "melhorias_bl_wf_val_k.csv"
    out_png = FIGURES_DIR / "melhorias_bl_wf_oos_sweep.png"
    out_json = OPTIMIZE_DIR / "melhorias_bl_wf_best.json"

    oos.to_csv(out_oos, index=False)
    sweep.to_csv(out_sweep, index=False)
    val_k.to_csv(out_val, index=False)
    plot_sweep(sweep, k_star, out_png)
    apply_best_to_config(ip, k_star, out_json)
    _write_config_extras(ip)
    print("config.py atualizado com params BL + news gate.")

    print("\n=== OOS k* ===")
    print(oos.to_string(index=False))
    print(f"\nSalvo: {out_oos}")
    print(f"Salvo: {out_sweep}")
    print(f"Figura: {out_png}")
    print(f"Params: {out_json}")


if __name__ == "__main__":
    assert _effective_max_w(0.4, 5, 2.0) * 5 >= 1.0 - 1e-6
    try:
        main()
    except Exception as exc:
        print(f"Erro: {exc}")
        sys.exit(1)
