"""Compara alvos de log-retorno e retorno simples com métricas na mesma escala."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / 'features_baixas_2026_11_25' / 'pesos_baixas'
sys.path.insert(0, str(PRIOR))
from treinar import FOLDS, KEYS, TICKERS, fit, indicators, legacy, prepare
from comparar_modelos import compute_metrics

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')


def main():
    prior_predictions = PRIOR / 'previsoes_ponderadas_2025.csv'
    protocol = PRIOR / 'protocolo.json'
    multiplier = float(json.loads(protocol.read_text(encoding='utf-8'))['peso_escolhido'])
    files = {ticker: legacy.DATA_DIR / f'{ticker}.csv' for ticker in TICKERS}
    protected = [prior_predictions, protocol, *files.values(), *legacy.OUT_DIR.glob('*_predicoes.csv'),
                 *OUT.parents[1].glob('**/*.tex')]
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in protected}
    frames, sets = prepare(files)
    frame = pd.concat([frames[t, 21].join(indicators(path)[['beta_baixas_63']], on='date', validate='one_to_one')
                       for t,path in files.items()], ignore_index=True).sort_values(['date','ticker'])
    ids = pd.get_dummies(frame.ticker, prefix='ticker', dtype=np.float32)
    frame = pd.concat([frame, ids], axis=1).reset_index(drop=True)
    x = frame[sets['simplificadas'] + ids.columns.tolist() + ['beta_baixas_63']].to_numpy(np.float32)
    log_y = frame.actual.to_numpy(float)
    results, logs = [], []
    for name,y in [('log', log_y), ('simples', np.expm1(log_y))]:
        assert np.array_equal(np.sign(log_y), np.sign(y))
        dataset = frame.assign(actual=y)
        for fold in FOLDS:
            predicted, audit = fit(dataset, x, y, multiplier, fold)
            if name == 'log':
                predicted[['actual','previsao']] = np.expm1(predicted[['actual','previsao']])
            results.append(predicted.assign(alvo=name))
            logs.append(dict(alvo=name, **audit))
        print(f'Alvo {name}: seis blocos concluídos.', flush=True)
    result = pd.concat(results, ignore_index=True)
    assert result.groupby(KEYS + ['actual']).size().eq(2).all()
    assert not result.duplicated(KEYS + ['alvo']).any()
    previous = pd.read_csv(prior_predictions, parse_dates=['date','maturity'])
    match = result.query("alvo == 'log' and fase == 'test' and dias_corridos > 1").merge(
        previous, on=KEYS, validate='one_to_one')
    assert len(match) == len(previous)
    np.testing.assert_allclose(match.previsao_x, np.expm1(match.previsao_y), atol=1e-12, rtol=0)
    metric_rows, pooled_rows = [], []
    for name,subset in [('todos',result),('apos_pausa',result[result.dias_corridos.gt(1)])]:
        metric_rows.append(compute_metrics(subset, ['fase','alvo','ticker']).assign(recorte=name))
        for (phase,target),group in subset.groupby(['fase','alvo']):
            pooled_rows.append(dict(recorte=name, fase=phase, alvo=target,
                r2_agregado=1-(group.actual-group.previsao).pow(2).mean()/group.actual.var(ddof=0)))
    by_asset = pd.concat(metric_rows, ignore_index=True)
    summary = by_asset.groupby(['recorte','fase','alvo']).agg(
        acoes=('ticker','nunique'), previsoes=('n','sum'), rmse=('rmse','mean'), mae=('mae','mean'),
        r2=('r2','mean'), altas=('altas_pct','mean'), baixas=('baixas_pct','mean'),
        balanceada=('balanceada_pct','mean'), acoes_r2_positivo=('r2',lambda r:int(r.gt(0).sum()))
    ).reset_index().merge(pd.DataFrame(pooled_rows), on=['recorte','fase','alvo'], validate='one_to_one')
    summary[['rmse','mae']] *= 100
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path,digest in hashes.items())
    summary.to_csv(OUT/'resumo.csv',index=False)
    by_asset.to_csv(OUT/'metricas_por_acao.csv',index=False)
    result.to_csv(OUT/'previsoes_em_retorno_simples.csv',index=False)
    pd.DataFrame(logs).to_csv(OUT/'auditoria.csv',index=False)
    (OUT/'protocolo.json').write_text(json.dumps(dict(horizonte=21, escala_avaliacao='retorno simples',
        peso_baixas=multiplier, limiar=0., arquivos_conferidos=len(hashes),
        previsoes_da_referencia_reproduzidas=len(match)),indent=2),encoding='utf-8')
    print(summary.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
