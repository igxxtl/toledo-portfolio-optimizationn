"""Sweep isolado de gate=1..5, mensal, 20 ativos; nao altera config.py.

Executar: python -m intel.Blacklitterman.simular_gates_rapido
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from . import config as cfg
from .simular_dois_cenarios import load_inputs, prepare_predictions, build_signals, simulate


def main():
    output = cfg.OUTPUTS_DIR / 'sensibilidade_gate_1a5'
    output.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')
    prices, returns, paths = load_inputs()
    predictions, choices, pred_paths = prepare_predictions(prices, start)
    original_gate = cfg.NEWS_COUNT_MIN
    rows, series, audits, trades = [], [], [], []
    baseline = None
    try:
        for gate in range(1, 6):
            # cfg e um modulo carregado neste processo; arquivo em disco permanece intacto.
            cfg.NEWS_COUNT_MIN = gate
            dates, signals, audit = build_signals(returns, prices, predictions, start, end, 126, 3.)
            metrics, curve, trade = simulate(dates, returns, signals, len(cfg.ATIVOS), 'BL', None, 3., 10.)
            eligible = audit.news_count.ge(gate)
            rows.append(dict(gate=gate, **metrics, eligible_views=int(eligible.sum()),
                             total_asset_decisions=len(audit),
                             dates_with_views=int(audit.loc[eligible, 'signal_date'].nunique())))
            series.append(curve.assign(gate=gate))
            audits.append(audit.assign(gate=gate, eligible=eligible))
            trades.append(trade.assign(gate=gate))
            if baseline is None:
                baseline, baseline_curve, baseline_trades = simulate(
                    dates, returns, signals, len(cfg.ATIVOS), 'Markowitz', None, 3., 10.)
            print(f'Gate {gate}: retorno={metrics["return_pct"]:.4f}%, '
                  f'Sharpe={metrics["sharpe"]:.4f}, views={int(eligible.sum())}', flush=True)
    finally:
        cfg.NEWS_COUNT_MIN = original_gate
    assert cfg.NEWS_COUNT_MIN == original_gate
    result = pd.DataFrame(rows)
    assert np.isfinite(result.select_dtypes('number').to_numpy()).all()
    assert result.n_days.nunique() == 1
    assert result.eligible_views.is_monotonic_decreasing
    checks = ['finite metrics; common 250 dates; monotonic eligible counts; official gate preserved']
    for gate, folder in [(1, 'dois_cenarios_gate1_historico'), (3, 'dois_cenarios'), (5, 'dois_cenarios_gate5_historico')]:
        previous = cfg.OUTPUTS_DIR / folder / 'metricas.csv'
        if previous.exists():
            saved = pd.read_csv(previous)
            saved = saved[(saved.scenario == 'escolha_livre') & (saved.n == len(cfg.ATIVOS)) & (saved.model == 'BL')].iloc[0]
            fresh = result[result.gate == gate].iloc[0]
            for metric in ['return_pct', 'sharpe', 'max_drawdown_pct', 'cost_brl', 'final_capital_brl']:
                np.testing.assert_allclose(fresh[metric], saved[metric], rtol=1e-10, atol=1e-8)
            checks.append(f'gate {gate} reproduces stored endpoint')
    result.to_csv(output / 'resumo.csv', index=False)
    pd.concat(series).to_csv(output / 'series_bl.csv', index=False)
    pd.concat(audits).to_csv(output / 'auditoria.csv', index=False)
    pd.concat(trades).to_csv(output / 'rebalanceamentos.csv', index=False)
    choices.to_csv(output / 'horizontes.csv', index=False)
    baseline_curve.to_csv(output / 'series_markowitz.csv', index=False)
    paths += pred_paths + [Path(__file__), Path(cfg.__file__), cfg.NEWS_PATH,
                            Path(__file__).with_name('simular_dois_cenarios.py'),
                            Path(__file__).with_name('sentiment.py')]
    manifest = dict(status='complete', exploratory=True, gates=list(range(1,6)), n_assets=len(cfg.ATIVOS),
                    period=[str(dates[0].date()), str(dates[-1].date())], rebalance='monthly',
                    window=126, risk_aversion=3., cost_bps=10., initial_capital=cfg.INITIAL_CAPITAL_BRL,
                    selic_annual=cfg.SELIC_ANNUAL, official_gate_preserved=original_gate,
                    news_window='previous calendar day only', hard_prior=cfg.HARD_PRIOR_WHEN_LOW_NEWS,
                    alpha=cfg.ALPHA_Q, tau=cfg.TAU, markowitz=baseline, checks=checks,
                    warning='Retrospective 2025 sensitivity, not independent validation or selection of official gate.',
                    input_sha256={str(p.relative_to(cfg.REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(set(paths))})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(result[['gate','return_pct','sharpe','max_drawdown_pct','turnover_total','cost_brl','eligible_views','dates_with_views']].to_string(index=False))
    print('Markowitz:', json.dumps(baseline))
    print('Saida:', output)


if __name__ == '__main__':
    main()
