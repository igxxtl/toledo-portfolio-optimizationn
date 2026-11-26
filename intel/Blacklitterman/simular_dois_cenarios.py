"""Experimento exploratorio pareado: conjuntos comuns e selecao propria, n=1..20.

python -m intel.Blacklitterman.simular_dois_cenarios --repeticoes 30
Nao altera os artefatos do pipeline historico. Usa somente numpy/pandas.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .sentiment import load_daily_sentiment_for_ticker


def project(v: np.ndarray, cap: float, floor: float = .001) -> np.ndarray:
    """Projecao euclidiana no simplex com limites inferior e superior."""
    if len(v) == 1:
        return np.ones(1)
    if abs(cap * len(v) - 1) < 1e-10:
        return np.full(len(v), 1 / len(v))
    # A soma e linear entre os pontos de quebra: encontra a raiz sem iteracoes.
    breaks = np.unique(np.r_[v - cap, v - floor])
    mass = np.clip(v[None, :] - breaks[:, None], floor, cap).sum(axis=1)
    k = int(np.searchsorted(-mass, -1.))
    k = min(max(k, 1), len(breaks) - 1)
    fraction = (mass[k - 1] - 1) / max(mass[k - 1] - mass[k], 1e-15)
    threshold = breaks[k - 1] + fraction * (breaks[k] - breaks[k - 1])
    return np.clip(v - threshold, floor, cap)


def allocate(mu: np.ndarray, cov: np.ndarray, cap: float, risk: float) -> np.ndarray:
    """Maximiza mu'w - risk/2 w'Cov w, long-only, soma=1, teto e piso."""
    n = len(mu)
    if n == 1 or abs(cap * n - 1) < 1e-10:
        return np.full(n, 1 / n)
    # Annualizacao melhora o condicionamento numerico, sem mudar o otimo.
    a, b = risk * cov * 252, mu * 252
    lipschitz = max(float(np.linalg.eigvalsh(a)[-1]), 1e-9)
    w = np.full(n, 1 / n)
    y, t = w.copy(), 1.
    for _ in range(2500):
        new = project(y + (b - a @ y) / lipschitz, cap)
        if np.max(np.abs(new - w)) < 1e-9:
            return new
        new_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        y = new + (t - 1) / new_t * (new - w)
        w, t = new, new_t
    # Nao aceitar silenciosamente um resultado sem condicao de otimalidade.
    residual = np.max(np.abs(project(w + (b - a @ w) / lipschitz, cap) - w))
    if residual > 2e-6:
        raise RuntimeError(f"Otimizador nao convergiu: residual={residual}")
    return w


def load_inputs():
    prices = {}
    paths = [Path(__file__), Path(cfg.__file__), Path(__file__).with_name('sentiment.py'), cfg.NEWS_PATH]
    for asset in cfg.ATIVOS:
        path = cfg.DATA_DAILY_DIR / f"{asset.removesuffix('.SA')}.csv"
        df = pd.read_csv(path, usecols=['date', 'close'])
        df['date'] = pd.to_datetime(df['date'])
        if df['date'].duplicated().any():
            raise ValueError(f'Datas duplicadas: {path}')
        prices[asset] = df.set_index('date')['close'].sort_index()
        paths.append(path)
    prices = pd.DataFrame(prices).dropna()
    returns = prices.pct_change(fill_method=None).dropna()
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError('Retornos invalidos')
    return prices, returns, paths


def prepare_predictions(prices: pd.DataFrame, cutoff: pd.Timestamp):
    """Escolhe horizonte so na validacao; desfechos medidos nos precos locais."""
    selected, choices, paths = {}, [], []
    for asset in prices:
        base = asset.removesuffix('.SA')
        candidates = []
        for h in (5, 10, 21):
            path = cfg.XGB_DIR / f'{base}_h{h}d_predicoes.csv'
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=['date', 'y_pred'])
            df['date'] = pd.to_datetime(df['date'])
            if df['date'].duplicated().any():
                raise ValueError(f'Datas duplicadas: {path}')
            pred = df.set_index('date')['y_pred'].reindex(prices.index)
            actual = np.log(prices[asset].shift(-h) / prices[asset])
            maturity = pd.Series(prices.index, index=prices.index).shift(-h)
            valid = (pred.notna() & actual.notna() & maturity.lt(cutoff)
                     & (prices.index >= cutoff - pd.DateOffset(years=1)))
            if valid.sum() < 30:
                continue
            score = float((np.sign(pred[valid]) == np.sign(actual[valid])).mean())
            candidates.append((score, -h, pred, actual, maturity, path))
            paths.append(path)
        if not candidates:
            raise ValueError(f'Sem validacao anterior ao teste para {asset}')
        score, neg_h, pred, actual, maturity, path = max(candidates, key=lambda x: x[:2])
        h = -neg_h
        selected[asset] = dict(h=h, pred=pred, actual=actual, maturity=maturity)
        choices.append(dict(asset=asset, horizon=h, validation_direction_accuracy=score,
                            prediction_file=path.name, cutoff=str(cutoff.date())))
    return selected, pd.DataFrame(choices), paths


def build_signals(returns, prices, predictions, start, end, window, risk, variant='hybrid'):
    """Sinais no fechamento; execucao no fechamento seguinte (sem retorno antecipado)."""
    if variant not in ('hybrid', 'prior', 'xgb', 'sentiment', 'xgb_gate'):
        raise ValueError(f'Variante desconhecida: {variant}')
    dates = returns.loc[start:end].index
    first = dates.to_series().groupby(dates.to_period('M')).first()
    all_dates = returns.index
    signals, audit = {}, []
    news = {a: load_daily_sentiment_for_ticker(a.removesuffix('.SA')).set_index('view_date')
            for a in returns.columns}
    rf = (1 + cfg.SELIC_ANNUAL) ** (1 / 252) - 1
    for signal in first:
        pos = all_dates.get_loc(signal)
        execution = all_dates[pos + 1]
        if execution > dates[-1]:
            continue
        hist = returns.loc[:signal].tail(window)
        if len(hist) < window:
            raise ValueError('Historico insuficiente')
        cov = hist.cov().to_numpy() + np.eye(len(hist.columns)) * 1e-7
        historical_mu = hist.mean().to_numpy() - rf
        # Prior neutro: nao utiliza market cap observado em 2026 num teste de 2025.
        pi = risk * cov @ np.full(len(hist.columns), 1 / len(hist.columns))
        q, omega = pi.copy(), np.full(len(pi), 1e6)
        for j, asset in enumerate(hist.columns):
            p = predictions[asset]
            available = p['pred'].loc[:signal].dropna()
            if available.empty:
                raise ValueError(f'Sem previsao disponivel: {asset} {signal}')
            pred_date = available.index[-1]
            # Recusa previsoes antigas em vez de preenchimento ilimitado.
            if len(prices.loc[pred_date:signal]) > 6:
                raise ValueError(f'Previsao desatualizada: {asset} {signal}')
            h, pred = p['h'], float(available.iloc[-1])
            matured = p['maturity'].le(signal) & p['pred'].notna() & p['actual'].notna()
            past_actual = p['actual'][matured].tail(cfg.ERR_WINDOW)
            errors = (p['pred'][matured] - p['actual'][matured]).abs().tail(cfg.ERR_WINDOW)
            if len(errors) < cfg.ERR_MIN_PERIODS:
                raise ValueError(f'Erros maduros insuficientes: {asset}')
            # Confianca agregada historica, nunca erro da previsao ainda nao realizada.
            scale = max(float(errors.median()), 1e-8)
            model_conf = float(np.exp(-errors / scale).mean())
            # Noticias do dia anterior: todas ja publicadas antes do fechamento do sinal.
            news_day = signal.normalize() - pd.Timedelta(days=1)
            ns = news[asset]
            count = int(ns.loc[news_day, 'news_count']) if news_day in ns.index else 0
            sent = float(ns.loc[news_day, 'sentiment_signal']) if news_day in ns.index else 0.
            gate = cfg.NEWS_COUNT_MIN
            coverage = min(1., count / gate) if gate else float(count > 0)
            ret_scale = max(float(past_actual.abs().mean()), 1e-8)
            q[j] = ((cfg.ALPHA_Q + (1 - cfg.ALPHA_Q) * (1 - coverage)) * pred
                    + (1 - cfg.ALPHA_Q) * coverage * sent * ret_scale) / h - rf
            if variant in ('xgb', 'xgb_gate'):
                q[j] = pred / h - rf
            elif variant == 'sentiment':
                q[j] = sent * ret_scale / h - rf
            confidence = np.clip(cfg.W_MODEL_CONF * model_conf
                                 + cfg.W_NEWS_CONF * (1 - np.exp(-count / 3)),
                                 cfg.CONFIDENCE_FLOOR, cfg.CONFIDENCE_CAP)
            if variant == 'xgb':
                confidence = np.clip(model_conf, cfg.CONFIDENCE_FLOOR, cfg.CONFIDENCE_CAP)
            elif variant == 'sentiment':
                confidence = np.clip(1 - np.exp(-count / 3), cfg.CONFIDENCE_FLOOR, cfg.CONFIDENCE_CAP)
            if cfg.HARD_PRIOR_WHEN_LOW_NEWS and count < gate and variant != 'xgb':
                confidence = cfg.EPS
            daily_var = max(float(past_actual.var()) / h**2, 1e-8)
            omega[j] = cfg.TAU * daily_var * (1 - confidence) / confidence
            if count < gate and variant != 'xgb':
                omega[j] *= cfg.NEWS_OMEGA_BOOST
            omega[j] = max(omega[j], cfg.EPS)
            audit.append(dict(signal_date=signal, execution_date=execution, asset=asset,
                              prediction_date=pred_date, mature_errors=len(errors),
                              max_outcome_date=p['maturity'][matured].max(), news_date=news_day,
                              news_count=count, Q=q[j], omega=omega[j], model_confidence=model_conf,
                              variant=variant, horizon=h, prediction=pred, sentiment_signal=sent,
                              sentiment_component=sent * ret_scale, confidence=confidence,
                              prior=pi[j], gate_active=count >= gate))
        prior_cov = cfg.TAU * cov
        bl_mu = pi + prior_cov @ np.linalg.solve(prior_cov + np.diag(omega), q - pi)
        if variant == 'prior':
            bl_mu = pi.copy()
        for row, posterior in zip(audit[-len(pi):], bl_mu):
            row['posterior'] = posterior
            row['posterior_minus_prior'] = posterior - row['prior']
        signals[execution] = dict(signal_date=signal, cov=cov, BL=bl_mu, Markowitz=historical_mu)
    return dates, signals, pd.DataFrame(audit)


def subsets(n, total, repetitions, seed):
    count = min(math.comb(total, n), repetitions)
    if math.comb(total, n) <= repetitions:
        return list(itertools.combinations(range(total), n))
    rng = np.random.default_rng(seed + n)
    found = set()
    while len(found) < count:
        found.add(tuple(sorted(rng.choice(total, n, replace=False).tolist())))
    return sorted(found)


def simulate(dates, returns, signals, n, model, subset, risk, cost_bps):
    assets = list(returns.columns)
    size = len(assets)
    cap = max(cfg.MAX_WEIGHT_PER_ASSET, 1 / n)
    holdings, cash, wealth = np.zeros(size), 1., 1.
    rf = (1 + cfg.SELIC_ANNUAL) ** (1 / 252) - 1
    rows, trades = [], []
    for date in dates:
        before = wealth
        # Retorno ocorre antes da execucao ao fechamento. A nova carteira so rende amanha.
        holdings *= 1 + returns.loc[date].to_numpy()
        cash *= 1 + rf
        pretrade = float(holdings.sum() + cash)
        cost, turnover = 0., 0.
        if date in signals:
            sig = signals[date]
            mu, cov = sig[model], sig['cov']
            if subset is None:
                ranking = np.argsort(-mu / np.sqrt(np.diag(cov)), kind='stable')
                chosen = np.sort(ranking[:n])
            else:
                chosen = np.array(subset)
            target = np.zeros(size)
            target[chosen] = allocate(mu[chosen], cov[np.ix_(chosen, chosen)], cap, risk)
            remaining = pretrade
            rate = cost_bps / 10000
            # Meia L1 incluindo caixa; entrada inicial tem turnover=1.
            for _ in range(20):
                turnover = .5 * (np.abs(target * remaining - holdings).sum() + abs(cash)) / pretrade
                cost = pretrade * rate * turnover
                remaining = pretrade - cost
            holdings, cash = target * remaining, 0.
            trades.append(dict(execution_date=date, signal_date=sig['signal_date'],
                               selected=';'.join(assets[i] for i in chosen), cap=cap,
                               weights=json.dumps({assets[i]: float(target[i]) for i in chosen}),
                               turnover=turnover, cost_fraction=cost / pretrade))
        wealth = float(holdings.sum() + cash)
        rows.append(dict(date=date, ret=wealth / before - 1, capital=cfg.INITIAL_CAPITAL_BRL * wealth,
                         turnover=turnover, cost_brl=cfg.INITIAL_CAPITAL_BRL * cost))
    curve = pd.DataFrame(rows)
    r = curve['ret'].to_numpy()
    cap_path = np.r_[cfg.INITIAL_CAPITAL_BRL, curve.capital.to_numpy()]
    excess = r - rf
    metrics = dict(return_pct=(wealth - 1) * 100, final_capital_brl=cap_path[-1],
                   volatility_pct=float(np.std(r, ddof=1) * np.sqrt(252) * 100),
                   sharpe=float(excess.mean() / excess.std(ddof=1) * np.sqrt(252)),
                   max_drawdown_pct=float(np.min(cap_path / np.maximum.accumulate(cap_path) - 1) * 100),
                   turnover_total=float(curve.turnover.sum()), cost_brl=float(curve.cost_brl.sum()),
                   selic_return_pct=((1 + rf)**len(r) - 1) * 100,
                   cap=cap, n_days=len(r))
    return metrics, curve, pd.DataFrame(trades)


def chart(summary, output):
    """SVG independente de bibliotecas graficas; bandas sao dispersao, nao IC."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="520" viewBox="0 0 1100 520">',
             '<rect width="1100" height="520" fill="white"/>',
             '<g font-family="Arial" font-size="13" fill="#223">',
             '<text x="35" y="28" font-size="19">BL híbrido × Markowitz × Selic — 2025, mensal (exploratório)</text>']
    low = min(0., summary.return_p10.min()) - 5
    high = max(summary.return_p90.max(), summary.selic_return_pct.max()) + 5
    for panel, scenario in enumerate(('mesmos_ativos', 'escolha_livre')):
        left, top, width, height = 65 + panel * 540, 90, 450, 320
        xx = lambda n: left + (n - 1) / 19 * width
        yy = lambda v: top + height - (v - low) / (high - low) * height
        parts.append(f'<text x="{left}" y="65" font-size="17">{scenario.replace("_", " ")}</text>')
        for v in np.linspace(low, high, 6):
            y = yy(v)
            parts.append(f'<path d="M{left} {y}h{width}" stroke="#ddd"/><text x="{left-42}" y="{y+4}">{v:.0f}%</text>')
        for model, color in [('BL', '#2166ac'), ('Markowitz', '#d66024')]:
            g = summary[(summary.scenario == scenario) & (summary.model == model)].sort_values('n')
            coords = [(xx(r.n), yy(r.return_p10)) for r in g.itertuples()]
            coords += [(xx(r.n), yy(r.return_p90)) for r in reversed(list(g.itertuples()))]
            parts.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in coords)}" fill="{color}" opacity=".15"/>')
            line = ' '.join(f'{xx(r.n)},{yy(r.return_median)}' for r in g.itertuples())
            parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        sy = yy(float(summary.selic_return_pct.iloc[0]))
        parts.append(f'<path d="M{left} {sy}h{width}" stroke="#555" stroke-dasharray="6 4"/>')
        for n in (1, 5, 10, 15, 20):
            parts.append(f'<text x="{xx(n)-5}" y="435">{n}</text>')
        parts.append(f'<text x="{left+150}" y="460">Número de ativos (n)</text>')
    parts.extend(['<text x="65" y="495">Azul: BL | Laranja: Markowitz | Tracejado: Selic | Faixa: P10–P90 dos subconjuntos; linha: mediana</text>', '</g></svg>'])
    output.write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeticoes', type=int, default=30)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--window', type=int, default=126)
    parser.add_argument('--risk-aversion', type=float, default=3.)
    parser.add_argument('--cost-bps', type=float, default=10.)
    parser.add_argument('--output', type=Path, default=cfg.OUTPUTS_DIR / 'dois_cenarios')
    args = parser.parse_args()
    if args.repeticoes < 1 or args.window < 40 or args.risk_aversion <= 0 or args.cost_bps < 0:
        parser.error('Parametros invalidos')
    args.output.mkdir(parents=True, exist_ok=True)
    prices, returns, paths = load_inputs()
    start, end = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')
    predictions, choices, pred_paths = prepare_predictions(prices, start)
    choices.to_csv(args.output / 'horizontes_validacao.csv', index=False)
    dates, signals, audit = build_signals(returns, prices, predictions, start, end, args.window, args.risk_aversion)
    audit.to_csv(args.output / 'auditoria_sinais.csv', index=False)
    rows, curves, trades, subset_rows = [], [], [], []
    for n in range(1, len(cfg.ATIVOS) + 1):
        common = subsets(n, len(cfg.ATIVOS), args.repeticoes, args.seed)
        cases = [('mesmos_ativos', i, s) for i, s in enumerate(common)] + [('escolha_livre', 0, None)]
        for scenario, repetition, subset in cases:
            if subset is not None:
                subset_rows.append(dict(n=n, repetition=repetition, assets=';'.join(cfg.ATIVOS[i] for i in subset)))
            for model in ('BL', 'Markowitz'):
                metrics, curve, trade = simulate(dates, returns, signals, n, model, subset,
                                                  args.risk_aversion, args.cost_bps)
                tags = dict(scenario=scenario, n=n, repetition=repetition, model=model)
                rows.append({**tags, **metrics})
                curves.append(curve.assign(**tags))
                trades.append(trade.assign(**tags))
        print(f'n={n}: {len(common)} conjuntos comuns + escolha livre concluidos', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / 'metricas.csv', index=False)
    pd.concat(curves).to_csv(args.output / 'series.csv', index=False)
    pd.concat(trades).to_csv(args.output / 'rebalanceamentos.csv', index=False)
    pd.DataFrame(subset_rows).to_csv(args.output / 'subconjuntos.csv', index=False)
    summary = result.groupby(['scenario', 'n', 'model'], as_index=False).agg(
        repetitions=('return_pct', 'size'), return_median=('return_pct', 'median'),
        return_p10=('return_pct', lambda s: s.quantile(.1)),
        return_p90=('return_pct', lambda s: s.quantile(.9)), sharpe_median=('sharpe', 'median'),
        drawdown_median=('max_drawdown_pct', 'median'), volatility_median=('volatility_pct', 'median'),
        turnover_median=('turnover_total', 'median'), cost_median_brl=('cost_brl', 'median'),
        selic_return_pct=('selic_return_pct', 'first'))
    summary.to_csv(args.output / 'resumo.csv', index=False)
    paired = result.pivot(index=['scenario', 'n', 'repetition'], columns='model', values='return_pct').reset_index()
    paired['BL_minus_Markowitz_pp'] = paired.BL - paired.Markowitz
    paired.to_csv(args.output / 'comparacao_pareada.csv', index=False)
    chart(summary, args.output / 'comparacao.svg')
    paths += pred_paths
    manifest = dict(status='complete', exploratory=True, args={**vars(args), 'output': str(args.output)},
                    period=[str(dates[0].date()), str(dates[-1].date())], n_days=len(dates),
                    assets=cfg.ATIVOS, bl_parameters={k: getattr(cfg, k) for k in (
                        'ALPHA_Q', 'TAU', 'NEWS_COUNT_MIN', 'HARD_PRIOR_WHEN_LOW_NEWS',
                        'NEWS_OMEGA_BOOST', 'MAX_WEIGHT_PER_ASSET', 'W_MODEL_CONF', 'W_NEWS_CONF',
                        'CONFIDENCE_FLOOR', 'CONFIDENCE_CAP', 'ERR_WINDOW', 'ERR_MIN_PERIODS')},
                    selic_annual=cfg.SELIC_ANNUAL, initial_capital=cfg.INITIAL_CAPITAL_BRL,
                    prior='equal-weight equilibrium, common fixed risk aversion',
                    selection='descending expected daily excess return / historical volatility',
                    min_target_weight=.001, cap_rule='max(config cap, 1/n)',
                    execution='signal close; trade next trading close; returns accrue before trades',
                    news_rule='previous calendar day only',
                    cost_rule='bps * half L1 traded value including cash, self financing; no terminal liquidation',
                    limitations=['Existing XGB classifier/regressor selection may use full OOF period.',
                                 'Existing XGB folds were not retrained or fully audited.',
                                 'Fixed current 20-stock universe; survivorship bias possible.',
                                 'Adjusted close only, no bid-ask spreads or execution price data.',
                                 'BL hyperparameters inherited from previous research; no new nested tuning.',
                                 'Pointwise subset percentiles are dispersion, not confidence intervals.',
                                 'n=20 posteriors estimated on full universe, then restricted to subset.'],
                    input_sha256={str(p.relative_to(cfg.REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(set(paths))})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(summary[summary.n.isin([1, 5, 10, 15, 20])].to_string(index=False))


if __name__ == '__main__':
    main()
