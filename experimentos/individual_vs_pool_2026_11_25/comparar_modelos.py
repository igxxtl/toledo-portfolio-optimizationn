"""Compara modelos individuais com um único XGBoost conjunto, em pares idênticos."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost

OUT = Path(__file__).resolve().parent
THESIS = OUT.parents[1]
sys.path.insert(0, str(OUT.parent / 'features_economicas_2026_11_25'))
os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')

from testar_features import ANCHOR, FOLDS, TICKERS, economic_features, fit_predict, metric, legacy

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
HORIZONS = (1, 5, 21)


def dataset(ticker, path):
    raw = pd.read_csv(path, parse_dates=['date']).set_index('date')
    assert raw.index.is_monotonic_increasing and not raw.index.has_duplicates
    assert raw.close.gt(0).all()
    observed = economic_features(raw).assign(dias_corridos=raw.index.to_series().diff().dt.days)
    features = observed.columns.tolist()
    frames = []
    for h in HORIZONS:
        date = raw.index.to_series()
        block = observed.assign(ticker=ticker, horizonte=h, date=date, maturity=date.shift(-h),
                                actual=np.log(raw.close.shift(-h) / raw.close))
        frames.append(block.dropna(subset=features + ['maturity', 'actual']).reset_index(drop=True))
    return frames, features


def predict(frame, columns, n_threads):
    parts, audit = [], []
    X, y = frame[columns].to_numpy(np.float32), frame.actual.to_numpy(float)
    for fold, start, end, phase in FOLDS:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        cutoff = pd.Timestamp('2025-01-01' if phase == 'validation' else '2025-12-31')
        train = np.flatnonzero((frame.date.lt(start) & frame.maturity.lt(start)).to_numpy())
        test = np.flatnonzero((frame.date.ge(start) & frame.date.lt(end) & frame.maturity.lt(cutoff)).to_numpy())
        predictions, check = fit_predict(X, y, frame, train, test, ANCHOR, 400, n_threads)
        assert pd.Timestamp(check['max_train_maturity']) < start
        assert pd.Timestamp(check['max_fit_maturity']) < pd.Timestamp(check['early_stop_start'])
        cols = ['ticker', 'horizonte', 'date', 'maturity', 'actual', 'dias_corridos']
        parts.append(frame.iloc[test][cols].assign(fase=phase, fold=fold, previsao=predictions))
        audit.append(dict(fold=fold, n_test=len(test), **check))
    return pd.concat(parts, ignore_index=True), audit


def scores(data):
    y, p = data.actual, data.previsao
    up, down = y.gt(0), y.lt(0)
    up_hit, down_hit = p[up].gt(0).mean(), p[down].lt(0).mean()
    return dict(**metric(y, p), altas_pct=100*up_hit, baixas_pct=100*down_hit,
                balanceada_pct=50*(up_hit+down_hit))


def compute_metrics(data, keys):
    return pd.DataFrame([dict(zip(keys, group), **scores(rows))
                         for group, rows in data.groupby(keys, sort=True)])


def main():
    files = {ticker: legacy.DATA_DIR / f'{ticker}.csv' for ticker in TICKERS}
    protected = [*files.values(), *legacy.OUT_DIR.glob('*_predicoes.csv'), *THESIS.glob('**/*.tex'),
                 OUT.parent / 'previsao_calendario_2026_11_25' / 'previsoes.csv']
    before = {path: sha256(path.read_bytes()).hexdigest() for path in protected}
    frames = {}
    for ticker, path in files.items():
        blocks, economic = dataset(ticker, path)
        frames.update({(ticker, h): block for h, block in zip(HORIZONS, blocks)})

    def run_individual(item):
        (ticker, h), frame = item
        result, logs = predict(frame, economic, 1)
        return result, [dict(modelo='individual', ticker=ticker, horizonte=h, **row) for row in logs]

    individual_results, audit = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, (result, logs) in enumerate(executor.map(run_individual, frames.items()), 1):
            individual_results.append(result)
            audit.extend(logs)
            if i % 12 == 0:
                print(f'{i}/{len(frames)} combinações individuais concluídas', flush=True)
    individual = pd.concat(individual_results, ignore_index=True)
    validation = compute_metrics(individual[individual.fase.eq('validation')], ['ticker', 'horizonte'])
    # A acurácia balanceada tem a mesma escala nos três horizontes. Desempates usam ganho contra zero.
    choices = validation.sort_values(['ticker', 'balanceada_pct', 'skill_zero', 'horizonte'],
                                      ascending=[True, False, False, True]).drop_duplicates('ticker')
    choices.to_csv(OUT / 'horizontes_escolhidos_2024.csv', index=False)
    validation.to_csv(OUT / 'validacao_individual_2024.csv', index=False)

    pooled = pd.concat(frames.values(), ignore_index=True).sort_values(['date', 'ticker', 'horizonte'])
    identity = pd.get_dummies(pooled[['ticker', 'horizonte']].astype(str), dtype=np.float32)
    pooled = pd.concat([pooled, identity], axis=1).reset_index(drop=True)
    print('Treinando um pool para todas as ações e horizontes em cada bloco temporal.', flush=True)
    combined, logs = predict(pooled, economic + identity.columns.tolist(), 3)
    audit.extend(dict(modelo='pool', ticker='todos', horizonte='todos', **row) for row in logs)
    key = ['ticker', 'horizonte', 'date', 'maturity', 'fase', 'fold']
    paired = individual.merge(combined[key + ['previsao', 'actual']], on=key, suffixes=('_individual','_pool'),
                              validate='one_to_one')
    assert len(paired) == len(individual) == len(combined)
    np.testing.assert_array_equal(paired.pop('actual_pool'), paired.actual_individual)
    paired = paired.rename(columns={'actual_individual':'actual'})
    paired['selecionado_2024'] = paired.horizonte.eq(paired.ticker.map(choices.set_index('ticker').horizonte))
    paired.to_csv(OUT / 'previsoes_pareadas.csv', index=False)
    pd.DataFrame(audit).to_csv(OUT / 'auditoria_treino.csv', index=False)

    data = paired[paired.fase.eq('test')].copy()
    data['previsao_zero'] = 0.
    reshaped = data.melt(id_vars=['ticker','horizonte','date','actual','dias_corridos','selecionado_2024'],
                        value_vars=['previsao_individual','previsao_pool','previsao_zero'],
                        var_name='modelo', value_name='previsao')
    reshaped.modelo = reshaped.modelo.str.removeprefix('previsao_')
    all_assets = compute_metrics(reshaped[reshaped.modelo.ne('zero')], ['ticker','horizonte','modelo'])
    all_assets.to_csv(OUT / 'metricas_por_acao_horizonte.csv', index=False)
    samples = {'todos_por_horizonte': reshaped, 'selecionado': reshaped[reshaped.selecionado_2024],
               'selecionado_apos_pausa': reshaped[reshaped.selecionado_2024 & reshaped.dias_corridos.gt(1)]}
    results = []
    for name, sample in samples.items():
        metrics = compute_metrics(sample, ['ticker','horizonte','modelo'])
        groups = ['modelo','horizonte'] if name == 'todos_por_horizonte' else ['modelo']
        summary = metrics.groupby(groups).agg(acoes=('ticker','nunique'), observacoes=('n','sum'),
            rmse=('rmse','mean'), mae=('mae','mean'), r2=('r2','mean'), acerto=('hr','mean'),
            altas=('altas_pct','mean'), baixas=('baixas_pct','mean'), balanceada=('balanceada_pct','mean'),
            skill_zero=('skill_zero','mean')).reset_index()
        results.append(summary.assign(recorte=name))
    result = pd.concat(results, ignore_index=True)
    result.to_csv(OUT / 'resumo_2025.csv', index=False)
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items())
    (OUT / 'verificacao.json').write_text(json.dumps(dict(arquivos_conferidos=len(before),
        pares=len(paired), xgboost=xgboost.__version__, horizontes=HORIZONS,
        escolhas_2024=choices.horizonte.value_counts().to_dict()), indent=2), encoding='utf-8')
    print(choices.groupby('horizonte').ticker.agg(list).to_string(), flush=True)
    print(result.round(5).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
