"""Compara os quantis condicionais do XGBoost com o pool pontual de 21 pregões."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / 'features_baixas_2026_11_25' / 'pesos_baixas'
sys.path.insert(0, str(PRIOR))
from treinar import FOLDS, KEYS, TICKERS, fit, indicators, legacy, prepare

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
QUANTILES = np.array([.1,.5,.9])
NAMES = ['p10','p50','p90']


def fit_quantiles(frame, x, y, fold):
    name, start, end, phase = fold
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cutoff = pd.Timestamp('2025-01-01' if phase=='validation' else '2025-12-31')
    train = np.flatnonzero((frame.date.lt(start) & frame.maturity.lt(start)).to_numpy())
    test = np.flatnonzero((frame.date.ge(start) & frame.date.lt(end) & frame.maturity.lt(cutoff)).to_numpy())
    cut = int(.8*len(train))
    internal = train[cut:]
    initial = train[:cut][frame.maturity.iloc[train[:cut]].lt(frame.date.iloc[internal[0]]).to_numpy()]
    assert frame.maturity.iloc[train].max() < start
    assert frame.maturity.iloc[initial].max() < frame.date.iloc[internal[0]]
    params = dict(objective='reg:quantileerror',quantile_alpha=QUANTILES,tree_method='hist',
        learning_rate=.03,subsample=.8,colsample_bytree=.8,max_bin=64,max_depth=4,min_child_weight=10,
        reg_alpha=.1,reg_lambda=1.,n_jobs=3,random_state=42,verbosity=0)
    center, scale = y[initial].mean(), y[initial].std()
    early = XGBRegressor(**params,n_estimators=400,early_stopping_rounds=20)
    early.fit(x[initial],(y[initial]-center)/scale,eval_set=[(x[internal],(y[internal]-center)/scale)],verbose=False)
    rounds = int(early.best_iteration)+1
    center, scale = y[train].mean(), y[train].std()
    model = XGBRegressor(**params,n_estimators=rounds)
    model.fit(x[train],(y[train]-center)/scale,verbose=False)
    predictions = model.predict(x[test]).astype(float)*scale+center
    assert predictions.shape==(len(test),3) and np.isfinite(predictions).all()
    unordered = np.diff(predictions,axis=1).min(axis=1)<0
    block = frame.iloc[test][['ticker','date','maturity','horizonte','actual','dias_corridos']].reset_index(drop=True)
    block[NAMES] = np.sort(predictions,axis=1)
    block = block.assign(fase=phase,fold=name,quantis_cruzados=unordered)
    return block, dict(fold=name,rounds=rounds,n_train=len(train),max_train_maturity=str(frame.maturity.iloc[train].max().date()))


def metrics(frame, low, middle, high):
    below, above = frame.actual.lt(frame[low]), frame.actual.gt(frame[high])
    width = frame[high]-frame[low]
    interval_score = width + 10*(frame[low]-frame.actual).clip(lower=0) + 10*(frame.actual-frame[high]).clip(lower=0)
    errors = (frame.actual-frame[middle]).abs()
    return dict(n=len(frame),cobertura_pct=100*(~(below|above)).mean(),abaixo_pct=100*below.mean(),
                acima_pct=100*above.mean(),largura=100*width.mean(),mae=100*errors.mean(),
                escore_intervalo=100*interval_score.mean())


def main():
    previous = PRIOR/'previsoes_ponderadas_2025.csv'
    files = {ticker:legacy.DATA_DIR/f'{ticker}.csv' for ticker in TICKERS}
    protected = [previous,*files.values(),*legacy.OUT_DIR.glob('*_predicoes.csv'),*OUT.parents[1].glob('**/*.tex')]
    hashes = {p:sha256(p.read_bytes()).hexdigest() for p in protected}
    frames, sets = prepare(files)
    pool = pd.concat([frames[t,21].join(indicators(path)[['beta_baixas_63']],on='date',validate='one_to_one')
                      for t,path in files.items()],ignore_index=True).sort_values(['date','ticker'])
    ids = pd.get_dummies(pool.ticker,prefix='ticker',dtype=np.float32)
    pool = pd.concat([pool,ids],axis=1).reset_index(drop=True)
    x = pool[sets['simplificadas']+list(ids)+['beta_baixas_63']].to_numpy(np.float32)
    y = pool.actual.to_numpy(float)
    parts, history, audit = [], [], []
    for fold in FOLDS:
        forecast, log = fit_quantiles(pool,x,y,fold)
        parts.append(forecast)
        audit.append(log)
        if fold[-1]=='validation':
            history.append(fit(pool,x,y,1.04,fold)[0])
        print(f'{fold[0]}: quantis treinados.',flush=True)
    validation = pd.concat(history,ignore_index=True).query('dias_corridos > 1')
    offsets = (validation.actual-validation.previsao).quantile([.1,.9]).to_numpy()
    forecasts = pd.concat(parts,ignore_index=True)
    old = pd.read_csv(previous,parse_dates=['date','maturity'])
    test = forecasts.query("fase == 'test' and dias_corridos > 1").merge(old[KEYS+['actual','previsao']],
        on=KEYS,suffixes=('','_pool'),validate='one_to_one')
    assert len(test)==len(old)==1020 and test.ticker.nunique()==20
    np.testing.assert_allclose(test.actual,test.pop('actual_pool'),rtol=0,atol=1e-12)
    test['pool_p10'] = test.previsao+offsets[0]
    test['pool_p90'] = test.previsao+offsets[1]
    variants = {'pool_com_intervalo_de_erros_2024':('pool_p10','previsao','pool_p90'),
                'regressao_quantílica':('p10','p50','p90')}
    summary = pd.DataFrame([dict(modelo=name,**metrics(test,*columns)) for name,columns in variants.items()])
    details = pd.DataFrame([dict(modelo=name,ticker=ticker,**metrics(g,*columns))
                            for name,columns in variants.items() for ticker,g in test.groupby('ticker')])
    assert all(sha256(p.read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    summary.to_csv(OUT/'resumo_2025.csv',index=False)
    details.to_csv(OUT/'metricas_por_acao_2025.csv',index=False)
    test.to_csv(OUT/'previsoes_pareadas_2025.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'auditoria.csv',index=False)
    (OUT/'protocolo.json').write_text(json.dumps(dict(quantis=QUANTILES.tolist(),cobertura_nominal=.8,
        offsets_referencia_residuos_2024=offsets.tolist(),observacoes_referencia_2024=len(validation),
        cruzamentos_ordenados_2025=int(test.quantis_cruzados.sum()),arquivos_conferidos=len(hashes)),
        indent=2),encoding='utf-8')
    print(summary.round(5).to_string(index=False))
    print(f'Previsões com quantis cruzados antes da ordenação: {test.quantis_cruzados.sum()}',flush=True)


if __name__ == '__main__':
    main()
