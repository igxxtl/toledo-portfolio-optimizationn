"""Valida os artefatos gerados pelo experimento, sem executar novos backtests."""
import json
import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import config as cfg


def main():
    out = cfg.OUTPUTS_DIR / 'dois_cenarios'
    result = pd.read_csv(out / 'metricas.csv')
    paired = pd.read_csv(out / 'comparacao_pareada.csv')
    series = pd.read_csv(out / 'series.csv')
    trades = pd.read_csv(out / 'rebalanceamentos.csv')
    audit = pd.read_csv(out / 'auditoria_sinais.csv')
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    keys = ['scenario', 'n', 'repetition', 'model']
    assert manifest['status'] == 'complete'
    assert manifest['bl_parameters']['NEWS_COUNT_MIN'] == 3, 'Rodada atual deve usar gate de tres noticias'
    for relative, expected_hash in manifest['input_sha256'].items():
        assert hashlib.sha256((cfg.REPO_ROOT / relative).read_bytes()).hexdigest() == expected_hash, relative
    assert set(result.n) == set(range(1, 21))
    assert len(result.groupby(['scenario', 'n', 'model'])) == 80
    assert not result.duplicated(keys).any()
    assert np.isfinite(result.select_dtypes('number').to_numpy()).all()
    assert np.allclose(paired.loc[(paired.scenario == 'mesmos_ativos') & paired.n.isin([1, 2]),
                                 'BL_minus_Markowitz_pp'], 0, atol=1e-9)
    for model in ('BL', 'Markowitz'):
        a = result[(result.n == 20) & (result.model == model)]
        assert np.allclose(a.return_pct, a.return_pct.iloc[0], atol=1e-9)
    for row in trades.itertuples():
        w = json.loads(row.weights)
        assert len(w) == row.n
        assert abs(sum(w.values()) - 1) < 1e-8
        assert min(w.values()) >= .001 - 1e-8
        assert max(w.values()) <= row.cap + 1e-8
        assert pd.Timestamp(row.signal_date) < pd.Timestamp(row.execution_date)
    same = trades[trades.scenario == 'mesmos_ativos']
    assert same.groupby(['n', 'repetition', 'execution_date']).selected.nunique().eq(1).all()
    assert (pd.to_datetime(audit.max_outcome_date) <= pd.to_datetime(audit.signal_date)).all()
    assert (pd.to_datetime(audit.prediction_date) <= pd.to_datetime(audit.signal_date)).all()
    assert (pd.to_datetime(audit.news_date) < pd.to_datetime(audit.signal_date)).all()
    assert series.groupby(keys).size().eq(manifest['n_days']).all()
    last = series.groupby(keys).capital.last().sort_index()
    expected = result.set_index(keys).final_capital_brl.sort_index()
    np.testing.assert_allclose(last, expected, rtol=1e-12)
    capital = series.groupby(keys).ret.apply(lambda s: cfg.INITIAL_CAPITAL_BRL * (1 + s).prod()).sort_index()
    np.testing.assert_allclose(capital, expected, rtol=1e-12)
    rf = (1 + cfg.SELIC_ANNUAL)**(1 / 252) - 1
    np.testing.assert_allclose(result.selic_return_pct, ((1 + rf)**manifest['n_days'] - 1) * 100)
    ET.parse(out / 'comparacao.svg')
    verification = dict(status='passed', news_gate=3, simulations=len(result), daily_rows=len(series),
                        n_days=manifest['n_days'], checks=[
                            'input hashes match the completed run',
                            '80 scenario/n/model groups; finite metrics; unique simulation IDs',
                            'common n=1 and n=2 identical across models',
                            'n=20 identical across scenarios for each model',
                            'exactly n positive target holdings, budget, floor and cap respected',
                            'same assets in paired scenario; signal strictly before execution',
                            'matured outcomes and news dates precede allowed cutoff',
                            'same dates count, returns compound to reported capital',
                            'Selic benchmark aligned; valid SVG'])
    (out / 'verificacao.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
