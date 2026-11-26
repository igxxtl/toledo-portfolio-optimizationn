"""Compara duas arquiteturas e dois conjuntos de atributos após pausas de pregão."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / 'individual_vs_pool_2026_11_25'
sys.path.insert(0, str(PREVIOUS))
os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')

from comparar_modelos import HORIZONS, TICKERS, compute_metrics, dataset, legacy, predict

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
KEYS = ['ticker', 'horizonte', 'date', 'maturity', 'actual', 'dias_corridos', 'fase', 'fold']


def prepare(files):
    ranks = legacy.build_cross_sectional_ranks(list(files.values()))
    frames = {}
    original = None
    for ticker, path in files.items():
        blocks, simplified = dataset(ticker, path)
        raw = legacy.carregar_df(path)
        for col in ranks[ticker]:
            raw[col] = ranks[ticker][col].reindex(pd.to_datetime(raw.date)).to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
            matrix, columns = legacy.build_features(raw)
        if original is None:
            original = list(columns)
        assert set(columns) == set(original), f'Conjunto legado diferente para {ticker}'
        matrix = matrix[['date', *original]].replace([np.inf, -np.inf], np.nan)
        matrix = matrix.dropna(subset=original)
        for h, block in zip(HORIZONS, blocks):
            frames[ticker, h] = block.merge(matrix, on='date', validate='one_to_one').sort_values('date')
    assert 'dias_corridos' in simplified and 'dias_corridos' not in original
    return frames, {'originais': [*original, 'dias_corridos'], 'simplificadas': simplified}


def main():
    choices_path = PREVIOUS / 'horizontes_escolhidos_2024.csv'
    reference = PREVIOUS / 'previsoes_pareadas.csv'
    files = {t: legacy.DATA_DIR / f'{t}.csv' for t in TICKERS}
    protected = [*files.values(), choices_path, reference, Path(legacy.__file__),
                 *OUT.parents[1].glob('**/*.tex'), *legacy.OUT_DIR.glob('*_predicoes.csv')]
    before = {p: sha256(p.read_bytes()).hexdigest() for p in protected}
    choices = pd.read_csv(choices_path).set_index('ticker').horizonte.to_dict()
    frames, sets = prepare(files)

    def run_individual(job):
        scenario, ticker = job
        h = choices[ticker]
        result, audit = predict(frames[ticker, h], sets[scenario], 1)
        result = result.assign(cenario=scenario, modelo='individual')
        return result, [dict(cenario=scenario, modelo='individual', ticker=ticker, horizonte=h, **a)
                        for a in audit]

    jobs = [(scenario, ticker) for scenario in sets for ticker in TICKERS]
    parts, logs = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, (result, audit) in enumerate(executor.map(run_individual, jobs), 1):
            parts.append(result)
            logs.extend(audit)
            if i % 10 == 0:
                print(f'{i}/{len(jobs)} modelos individuais concluídos.', flush=True)

    pooled = pd.concat(frames.values(), ignore_index=True).sort_values(['date','ticker','horizonte'])
    identities = pd.get_dummies(pooled[['ticker', 'horizonte']].astype(str), dtype=np.float32)
    pooled = pd.concat([pooled, identities], axis=1).reset_index(drop=True)
    for scenario, columns in sets.items():
        print(f'Treinando pool com features {scenario}.', flush=True)
        result, audit = predict(pooled, columns + identities.columns.tolist(), 3)
        result = result[result.horizonte.eq(result.ticker.map(choices))]
        parts.append(result.assign(cenario=scenario, modelo='pool'))
        logs.extend(dict(cenario=scenario, modelo='pool', ticker='todos', horizonte='todos', **a) for a in audit)

    forecasts = pd.concat(parts, ignore_index=True)
    counts = forecasts.groupby(KEYS).size()
    assert len(forecasts) == len(counts)*4 and counts.eq(4).all()
    scope = forecasts[forecasts.fase.eq('test')]
    details = []
    for name, group in [('todos', scope), ('apos_pausa', scope[scope.dias_corridos.gt(1)])]:
        table = compute_metrics(group, ['ticker','horizonte','cenario','modelo'])
        details.append(table.assign(recorte=name))
    metrics = pd.concat(details, ignore_index=True)
    metrics.to_csv(OUT / 'metricas_por_acao.csv', index=False)
    summary = metrics.groupby(['recorte','cenario','modelo']).agg(
        acoes=('ticker','nunique'), previsoes=('n','sum'), rmse=('rmse','mean'), mae=('mae','mean'),
        acerto=('hr','mean'), altas=('altas_pct','mean'), baixas=('baixas_pct','mean'),
        balanceada=('balanceada_pct','mean'), skill_zero=('skill_zero','mean')).reset_index()
    summary.to_csv(OUT / 'resumo_2025.csv', index=False)
    forecasts.to_csv(OUT / 'previsoes.csv', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'auditoria_treino.csv', index=False)
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
    (OUT / 'verificacao.json').write_text(json.dumps(dict(arquivos_conferidos=len(before),
        observacoes_pareadas=len(counts), features={name: len(cols) for name, cols in sets.items()},
        horizontes_congelados=choices), indent=2), encoding='utf-8')
    print(summary.round(5).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
