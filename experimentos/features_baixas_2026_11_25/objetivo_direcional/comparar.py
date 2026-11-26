"""Compara a regressão de retorno e a classificação logística de direção."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
sys.path.insert(0, str(PARENT / 'pesos_baixas'))
from treinar import FOLDS, KEYS, TICKERS, calibrate, indicators, legacy, prepare, summarize

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')


def logistic(frame, columns):
    x = frame[columns].to_numpy(np.float32)
    y = frame.actual.gt(0).to_numpy(np.int8)
    parts, audit = [], []
    for label, start, end, phase in FOLDS:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        cutoff = pd.Timestamp('2025-01-01' if phase == 'validation' else '2025-12-31')
        train = np.flatnonzero((frame.date.lt(start) & frame.maturity.lt(start) & frame.actual.ne(0)).to_numpy())
        test = np.flatnonzero((frame.date.ge(start) & frame.date.lt(end) & frame.maturity.lt(cutoff)).to_numpy())
        cut = int(.8 * len(train))
        internal = train[cut:]
        boundary = frame.date.iloc[internal[0]]
        initial = train[:cut]
        initial = initial[frame.maturity.iloc[initial].lt(boundary).to_numpy()]
        assert frame.maturity.iloc[train].max() < start and frame.maturity.iloc[initial].max() < boundary
        params = dict(objective='binary:logistic', eval_metric='logloss', tree_method='hist',
            learning_rate=.03, subsample=.8, colsample_bytree=.8, max_bin=64, max_depth=4,
            min_child_weight=10, reg_alpha=.1, reg_lambda=1., n_jobs=3, random_state=42, verbosity=0)
        early = XGBClassifier(**params, n_estimators=400, early_stopping_rounds=20)
        early.fit(x[initial], y[initial], eval_set=[(x[internal], y[internal])], verbose=False)
        rounds = int(early.best_iteration) + 1
        model = XGBClassifier(**params, n_estimators=rounds)
        model.fit(x[train], y[train], verbose=False)
        parts.append(frame.iloc[test][['ticker','date','maturity','horizonte','actual','dias_corridos']]
                     .assign(previsao=model.predict_proba(x[test])[:,1], fase=phase, fold=label))
        audit.append(dict(fold=label, n_train=len(train), n_fit=len(initial), rounds=rounds,
                          max_train_maturity=str(frame.maturity.iloc[train].max().date())))
        print(f'{label}: classificador treinado com {rounds} árvores.', flush=True)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit)


def main():
    reference_path = PARENT / 'previsoes.csv'
    weighted_path = PARENT / 'pesos_baixas' / 'previsoes_ponderadas_2025.csv'
    files = {ticker: legacy.DATA_DIR / f'{ticker}.csv' for ticker in TICKERS}
    protected = [*files.values(), reference_path, weighted_path, *legacy.OUT_DIR.glob('*_predicoes.csv'),
                 *OUT.parents[2].glob('**/*.tex')]
    hashes = {path: sha256(path.read_bytes()).hexdigest() for path in protected}
    frames, sets = prepare(files)
    pool = pd.concat([frames[ticker,21].join(indicators(path)[['beta_baixas_63']], on='date', validate='one_to_one')
                      for ticker,path in files.items()], ignore_index=True).sort_values(['date','ticker'])
    identities = pd.get_dummies(pool.ticker, prefix='ticker', dtype=np.float32)
    pool = pd.concat([pool, identities], axis=1).reset_index(drop=True)
    columns = sets['simplificadas'] + identities.columns.tolist() + ['beta_baixas_63']
    forecast, audit = logistic(pool, columns)
    saved = pd.read_csv(reference_path, parse_dates=['date','maturity']).query('dias_corridos > 1')
    reference = saved[saved.modelo.eq('referencia')]
    beta = saved[saved.modelo.eq('beta_baixas_63')]
    weighted = pd.read_csv(weighted_path, parse_dates=['date','maturity'])
    scope = forecast[forecast.dias_corridos.gt(1)]
    for other in (reference, beta, weighted):
        match = scope.merge(other[KEYS + ['actual']], on=KEYS, validate='one_to_one')
        assert len(match) == len(other)
        np.testing.assert_allclose(match.actual_x, match.actual_y, atol=1e-12, rtol=0)

    validation = scope[scope.fase.eq('validation')]
    validation_ref = reference[reference.fase.eq('validation')]
    target = summarize(validation_ref, 0.)['acerto_altas']
    cutoff, _ = calibrate(validation, target)
    sections = []
    for phase in ['validation','test']:
        models = [('regressao_referencia', reference, 0.), ('regressao_beta', beta, 0.),
                  ('classificador_0_5', scope, .5), ('classificador_calibrado', scope, cutoff)]
        if phase == 'test':
            models.append(('regressao_beta_ponderada', weighted, 0.))
        sections.extend(dict(fase=phase, variante=name, **summarize(data[data.fase.eq(phase)], threshold))
                        for name,data,threshold in models)

    forecast.to_csv(OUT / 'previsoes_classificador.csv', index=False)
    audit.to_csv(OUT / 'auditoria_treino.csv', index=False)
    summary = pd.DataFrame(sections)
    summary.to_csv(OUT / 'resumo.csv', index=False)
    assert all(sha256(path.read_bytes()).hexdigest() == digest for path,digest in hashes.items())
    (OUT / 'protocolo.json').write_text(json.dumps(dict(objetivo='binary:logistic', metrica_early_stopping='logloss',
        limiar_calibrado_2024=cutoff, referencia_acerto_altas_2024=target, arquivos_conferidos=len(hashes)),
        indent=2), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
