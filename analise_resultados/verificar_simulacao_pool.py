"""Auditoria dos resultados e das restricoes da comparacao financeira."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analise_resultados/simulacao_pool'


def main():
    metrics = pd.read_csv(OUT / 'metricas.csv')
    assert len(metrics) == 3997
    assert not metrics.duplicated(['scenario', 'n', 'repetition', 'model']).any()
    assert np.isfinite(metrics.select_dtypes(include='number').to_numpy()).all()
    assert metrics.n_days.eq(250).all()
    # Todos os ativos sao elegiveis em ambos os cenarios quando n=20.
    twenty = metrics[metrics.n.eq(20)].pivot(index='model', columns='scenario', values='return_pct')
    np.testing.assert_allclose(twenty.escolha_livre, twenty.mesmos_ativos, atol=1e-9, rtol=0)
    # Um ativo ou teto 1/2 determinam os pesos dos subconjuntos comuns.
    small = metrics[metrics.scenario.eq('mesmos_ativos') & metrics.n.le(2)]
    spreads = small.groupby(['n', 'repetition']).return_pct.agg(lambda s: s.max()-s.min())
    assert spreads.max() < 1e-9
    trades = pd.read_csv(OUT / 'rebalanceamentos_selecao_propria.csv')
    assert (pd.to_datetime(trades.execution_date) > pd.to_datetime(trades.signal_date)).all()
    for row in trades.itertuples():
        weights = np.array(list(json.loads(row.weights).values()))
        assert len(weights) == row.n
        assert abs(weights.sum()-1) < 1e-7
        assert weights.min() >= .001-1e-8
        assert weights.max() <= row.cap+1e-8
    train = pd.read_csv(OUT / 'auditoria_treino.csv')
    assert (pd.to_datetime(train.max_train_maturity) < pd.to_datetime(train.first_prediction)).all()
    assert (pd.to_datetime(train.max_fit_maturity) < pd.to_datetime(train.early_stop_start)).all()
    audit = pd.read_csv(OUT / 'auditoria_comparada.csv')
    assert len(audit) == 240
    for name in ['auditoria_atual.csv', 'auditoria_pool.csv']:
        temporal = pd.read_csv(OUT / name)
        signal = pd.to_datetime(temporal.signal_date)
        assert (pd.to_datetime(temporal.prediction_date) == signal).all()
        assert (pd.to_datetime(temporal.max_outcome_date) <= signal).all()
        assert (pd.to_datetime(temporal.news_date) < signal).all()
    protocol = json.loads((OUT / 'protocolo_financeiro.json').read_text(encoding='utf-8'))
    for name, digest in protocol['reference_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    verification = json.loads((OUT / 'verificacao.json').read_text(encoding='utf-8'))
    assert verification['status'] == 'passed'
    assert verification['original_runs_reproduced'] == 1142
    result = dict(status='passed', runs=len(metrics), checked_rebalances=len(trades),
                  finite_metrics=True, unique_cases=True, same_universe_n20=True,
                  fixed_weights_small_subsets=True, weights_within_constraints=True,
                  execution_after_signal=True, purged_training=True,
                  original_reference_unchanged=True)
    (OUT / 'verificacao_adicional.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
