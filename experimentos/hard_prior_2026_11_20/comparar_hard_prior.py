"""Ablacao retrospectiva de hard prior; nao modifica o repositorio original."""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
REPO = Path(r'C:\Users\igor.toledo_visagio\Documents\GitHub\toledo-portfolio-optimizationn')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from intel.Blacklitterman import config as cfg
from intel.Blacklitterman import simular_dois_cenarios as engine


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.time()
    source_pool = REPO / 'analise_resultados/simulacao_pool'
    reference_path = source_pool / 'metricas.csv'
    original = dict(HARD_PRIOR_WHEN_LOW_NEWS=cfg.HARD_PRIOR_WHEN_LOW_NEWS,
                    NEWS_OMEGA_BOOST=cfg.NEWS_OMEGA_BOOST,
                    NEWS_COUNT_MIN=cfg.NEWS_COUNT_MIN)
    assert original['HARD_PRIOR_WHEN_LOW_NEWS'] is True
    assert original['NEWS_COUNT_MIN'] == 3
    arms = [('atual', True, original['NEWS_OMEGA_BOOST']),
            ('sem_hard_prior', False, original['NEWS_OMEGA_BOOST']),
            ('somente_gate', False, 1.)]
    protocol = dict(period='2025', prediction='XGBoost conjunto (pool) ja existente',
                    arms=arms, original_config=original, recalibration=False,
                    gate=3, universe=cfg.ATIVOS, cardinalities=list(range(1,21)),
                    paired_subsets_per_size=30, seed=2025, window=126, risk=3.,
                    cost_bps=10., rebalance='monthly',
                    news_window='dia civil anterior ao sinal',
                    limitation='Comparacao retrospectiva em 2025; sem novo treinamento ou selecao de parametros.')
    (OUT/'protocolo.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    prices, returns, paths = engine.load_inputs()
    start, end = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')
    current, choices, pred_paths = engine.prepare_predictions(prices, start)
    pooled = pd.read_csv(source_pool/'previsoes_pool.csv', parse_dates=['date'])
    predictions = {}
    for asset, p in current.items():
        selected = pooled[(pooled.ticker == asset.removesuffix('.SA')) & (pooled.horizon == p['h'])]
        assert not selected.date.duplicated().any()
        predictions[asset] = dict(h=p['h'], actual=p['actual'], maturity=p['maturity'],
                                 pred=selected.set_index('date').y_pred.reindex(prices.index))
    choices.to_csv(OUT/'horizontes_preservados.csv', index=False)
    protected = set(paths + pred_paths + [source_pool/'previsoes_pool.csv', reference_path,
                    Path(cfg.__file__), Path(engine.__file__), Path(engine.__file__).with_name('sentiment.py')])
    hashes = {str(p): digest(p) for p in protected}
    signals, audits = {}, {}
    try:
        for name, hard, boost in arms:
            cfg.HARD_PRIOR_WHEN_LOW_NEWS = hard
            cfg.NEWS_OMEGA_BOOST = boost
            dates, signals[name], audits[name] = engine.build_signals(
                returns, prices, predictions, start, end, 126, 3.)
            assert len(audits[name]) == 240
            audits[name].to_csv(OUT/f'auditoria_{name}.csv', index=False)
            print(f'Sinais preparados: {name}', flush=True)
    finally:
        for key, val in original.items():
            setattr(cfg, key, val)
    baseline = audits['atual']
    saved_audit = pd.read_csv(source_pool/'auditoria_pool.csv')
    for col in ['Q','omega','prior','confidence','posterior']:
        np.testing.assert_allclose(baseline[col], saved_audit[col], rtol=1e-9, atol=1e-10)
    low = baseline.news_count.lt(3)
    for name in ['sem_hard_prior', 'somente_gate']:
        np.testing.assert_allclose(baseline.Q, audits[name].Q, rtol=0, atol=0)
        np.testing.assert_allclose(baseline.loc[~low,'omega'], audits[name].loc[~low,'omega'], rtol=0, atol=0)
        assert (audits[name].loc[low,'omega'] < baseline.loc[low,'omega']).all()
    rows, curves, trades, subset_records = [], [], [], []
    # Todos os modelos usam o mesmo calendario e motor financeiro.
    for n in range(1,21):
        common = engine.subsets(n, len(returns.columns), 30, 2025)
        cases = [('mesmos_ativos', j, s) for j,s in enumerate(common)] + [('escolha_livre',0,None)]
        for scenario, repetition, subset in cases:
            if subset is not None:
                subset_records.append(dict(n=n, repetition=repetition,
                    assets=';'.join(returns.columns[j] for j in subset)))
            for name,_,_ in arms:
                m,c,t = engine.simulate(dates, returns, signals[name], n, 'BL', subset, 3., 10.)
                tag = dict(scenario=scenario,n=n,repetition=repetition,model=name)
                rows.append(dict(**tag,**m))
                if scenario == 'escolha_livre':
                    curves.append(c.assign(**tag))
                    trades.append(t.assign(**tag))
            # Referencias apenas na selecao propria, sem repeticao desnecessaria.
            if scenario == 'escolha_livre':
                for name in ['Markowitz', 'Prior']:
                    sig = signals['atual']
                    key = 'Markowitz'
                    if name == 'Prior':
                        sig = {date: dict(signal_date=s['signal_date'],cov=s['cov'],
                              Prior=baseline[baseline.execution_date.eq(date)].prior.to_numpy())
                               for date,s in sig.items()}
                        key = 'Prior'
                    m,c,t = engine.simulate(dates, returns, sig, n, key, None, 3., 10.)
                    tag = dict(scenario=scenario,n=n,repetition=0,model=name)
                    rows.append(dict(**tag,**m)); curves.append(c.assign(**tag)); trades.append(t.assign(**tag))
        print(f'n={n}: {len(common)} conjuntos pareados + selecao propria concluidos ({time.time()-started:.1f}s)', flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'metricas.csv', index=False)
    pd.concat(curves).to_csv(OUT/'series_selecao_propria.csv', index=False)
    pd.concat(trades).to_csv(OUT/'rebalanceamentos.csv', index=False)
    pd.DataFrame(subset_records).to_csv(OUT/'subconjuntos.csv', index=False)
    historical = pd.read_csv(reference_path)
    check = result[result.model.eq('atual')].merge(historical[historical.model.eq('BL_pool')],
        on=['scenario','n','repetition'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(check) == len(result[result.model.eq('atual')])
    for col in ['return_pct','sharpe','max_drawdown_pct','cost_brl','final_capital_brl']:
        np.testing.assert_allclose(check[col+'_new'],check[col+'_old'],rtol=1e-9,atol=1e-7)
    paired = result[result.model.isin([a[0] for a in arms])].pivot(
        index=['scenario','n','repetition'],columns='model',values='return_pct').reset_index()
    for name in ['sem_hard_prior','somente_gate']:
        paired[name+'_delta_pp'] = paired[name] - paired.atual
    paired.to_csv(OUT/'comparacao_pareada.csv', index=False)
    summary = result.groupby(['scenario','n','model'],as_index=False).agg(
        runs=('return_pct','size'),return_median=('return_pct','median'),
        return_p10=('return_pct',lambda x:x.quantile(.1)),return_p90=('return_pct',lambda x:x.quantile(.9)),
        sharpe_median=('sharpe','median'),drawdown_median=('max_drawdown_pct','median'),
        turnover_median=('turnover_total','median'),cost_median=('cost_brl','median'))
    summary.to_csv(OUT/'resumo.csv', index=False)
    verification = dict(status='passed', runs=len(result),baseline_reproduced=len(check),
        same_Q=True,unchanged_omega_with_sufficient_news=True,
        source_hashes_unchanged=all(digest(Path(p))==v for p,v in hashes.items()),
        configuration_restored=all(getattr(cfg,k)==v for k,v in original.items()),
        low_coverage_decisions=int(low.sum()),total_asset_decisions=len(baseline),
        input_sha256=hashes,elapsed_seconds=time.time()-started)
    assert verification['source_hashes_unchanged'] and verification['configuration_restored']
    (OUT/'verificacao.json').write_text(json.dumps(verification,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nRESULTADO COM AS 20 ACOES:',flush=True)
    print(result[(result.scenario=='escolha_livre') & result.n.eq(20)][
        ['model','return_pct','sharpe','max_drawdown_pct','cost_brl','turnover_total']].to_string(index=False),flush=True)
    print('Verificacao:', {k:v for k,v in verification.items() if k!='input_sha256'},flush=True)


if __name__ == '__main__':
    main()
