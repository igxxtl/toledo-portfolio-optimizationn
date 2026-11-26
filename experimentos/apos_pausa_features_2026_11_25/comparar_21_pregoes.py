"""Repete a comparação com horizonte fixo de 21 pregões para os quatro modelos."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from comparar_cenarios import OUT, PREVIOUS, KEYS, TICKERS, compute_metrics, legacy, predict, prepare


def main():
    horizon = 21
    output = OUT / 'horizonte_21'
    output.mkdir(exist_ok=True)
    files = {ticker: legacy.DATA_DIR / f'{ticker}.csv' for ticker in TICKERS}
    protected = [*files.values(), *legacy.OUT_DIR.glob('*_predicoes.csv'),
                 *OUT.parents[1].glob('**/*.tex'), OUT / 'resumo_2025.csv',
                 PREVIOUS / 'previsoes_pareadas.csv', OUT / 'previsoes.csv']
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in protected}
    frames, sets = prepare(files)
    frames = {ticker: frames[ticker, horizon] for ticker in TICKERS}

    def run_individual(job):
        scenario, ticker = job
        result, audit = predict(frames[ticker], sets[scenario], 1)
        info = dict(cenario=scenario, modelo='individual')
        return result.assign(**info), [dict(ticker=ticker, **info, **row) for row in audit]

    jobs = [(name, ticker) for name in sets for ticker in TICKERS]
    forecasts, logs = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, (result, audit) in enumerate(executor.map(run_individual, jobs), 1):
            forecasts.append(result)
            logs.extend(audit)
            if i % 10 == 0:
                print(f'{i}/{len(jobs)} modelos individuais concluídos.', flush=True)

    pool = pd.concat(frames.values(), ignore_index=True).sort_values(['date', 'ticker'])
    identities = pd.get_dummies(pool.ticker, prefix='ticker', dtype=np.float32)
    pool = pd.concat([pool, identities], axis=1).reset_index(drop=True)
    for scenario, columns in sets.items():
        info = dict(cenario=scenario, modelo='pool')
        print(f'Treinando pool com features {scenario}.', flush=True)
        result, audit = predict(pool, columns + identities.columns.tolist(), 3)
        forecasts.append(result.assign(**info))
        logs.extend(dict(ticker='todos', **info, **row) for row in audit)

    predictions = pd.concat(forecasts, ignore_index=True)
    paired = predictions.groupby(KEYS).size()
    assert paired.eq(4).all() and len(predictions) == len(paired) * 4
    assert not predictions.duplicated(KEYS + ['modelo','cenario']).any()
    assert predictions.horizonte.eq(horizon).all()
    test = predictions[predictions.fase.eq('test')]
    parts = []
    for label, sample in [('todos', test), ('apos_pausa', test[test.dias_corridos.gt(1)])]:
        metrics = compute_metrics(sample, ['ticker', 'modelo', 'cenario'])
        parts.append(metrics.assign(recorte=label))
    by_asset = pd.concat(parts, ignore_index=True)
    summary = by_asset.groupby(['recorte', 'cenario', 'modelo']).agg(
        acoes=('ticker','nunique'), previsoes=('n','sum'), rmse=('rmse','mean'), mae=('mae','mean'),
        acerto=('hr','mean'), altas=('altas_pct','mean'), baixas=('baixas_pct','mean'),
        balanceada=('balanceada_pct','mean'), skill_zero=('skill_zero','mean')).reset_index()
    assert summary.acoes.eq(20).all()
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in hashes.items())
    predictions.to_csv(output / 'previsoes.csv', index=False)
    by_asset.to_csv(output / 'metricas_por_acao.csv', index=False)
    summary.to_csv(output / 'resumo_2025.csv', index=False)
    pd.DataFrame(logs).to_csv(output / 'auditoria_treino.csv', index=False)
    (output / 'verificacao.json').write_text(json.dumps(dict(horizonte=horizon,
        observacoes_pareadas=len(paired), arquivos_conferidos=len(hashes),
        features={name: len(cols) for name, cols in sets.items()}), indent=2), encoding='utf-8')
    print(summary.round(5).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
