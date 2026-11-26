"""Mede o efeito das previsões experimentais na simulação original."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2] / 'GitHub' / 'toledo-portfolio-optimizationn'
sys.path.insert(0, str(REPO))

from intel.Blacklitterman import simular_dois_cenarios as engine


def main():
    prices, returns, _ = engine.load_inputs()
    start, end = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')
    existing, _, _ = engine.prepare_predictions(prices, start)
    pooled = pd.read_csv(OUT / 'previsoes.csv', parse_dates=['date'])
    reference = REPO / 'analise_resultados' / 'simulacao_pool'
    archived = pd.read_csv(reference / 'metricas.csv').query("scenario == 'escolha_livre'")
    rows = []
    for arm in ['atual_publicado', 'atual_retreinado', 'economico', 'atual_mais_economico']:
        predictions = {}
        for asset, values in existing.items():
            mask = pooled.ticker.eq(asset.removesuffix('.SA')) & pooled.horizon.eq(values['h'])
            signal = pooled.loc[mask].set_index('date')[arm].reindex(prices.index)
            predictions[asset] = {**values, 'pred': signal}
        dates, signals, audit = engine.build_signals(returns, prices, predictions, start, end, 126, 3.)
        if arm == 'atual_publicado':
            previous = pd.read_csv(reference / 'auditoria_pool.csv')
            for column in ['Q', 'omega', 'prior', 'posterior']:
                np.testing.assert_allclose(audit[column], previous[column], rtol=1e-9, atol=1e-10)
        for size in [5, 10, 20]:
            metrics, _, _ = engine.simulate(dates, returns, signals, size, 'BL', None, 3., 10.)
            rows.append(dict(arm=arm, n=size, **metrics))
            if arm == 'atual_publicado':
                ref = archived[archived.n.eq(size) & archived.model.eq('BL_pool')].iloc[0]
                np.testing.assert_allclose(list(metrics.values()), ref[list(metrics)].to_numpy(float),
                                           rtol=1e-9, atol=1e-7)
    for size in [5, 10, 20]:
        ref = archived[archived.n.eq(size) & archived.model.eq('Prior')].iloc[0]
        rows.append(dict(arm='prior_sem_views', n=size, **ref.drop(['model', 'scenario', 'n', 'repetition'])))
    results = pd.DataFrame(rows)
    results.to_csv(OUT / 'carteiras.csv', index=False)
    print(results[['arm', 'n', 'return_pct', 'sharpe', 'max_drawdown_pct']].to_string(index=False))


if __name__ == '__main__':
    main()
