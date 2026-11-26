"""Testa calibração de magnitude e árvores mais flexíveis sem alterar a base."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

OUT = Path(__file__).resolve().parent
PREV = OUT.parent / 'features_baixas_2026_11_25' / 'pesos_baixas'
sys.path.insert(0, str(PREV))
from treinar import FOLDS, KEYS, TICKERS, fit, indicators, legacy, prepare, weights
from comparar_modelos import compute_metrics


def flexible(pool, x, y, fold):
    label, start, end, phase = fold
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cutoff = pd.Timestamp('2025-01-01' if phase == 'validation' else '2025-12-31')
    train = np.flatnonzero((pool.date.lt(start) & pool.maturity.lt(start)).to_numpy())
    test = np.flatnonzero((pool.date.ge(start) & pool.date.lt(end) & pool.maturity.lt(cutoff)).to_numpy())
    cut = int(.8*len(train))
    internal = train[cut:]
    fit_rows = train[:cut][pool.maturity.iloc[train[:cut]].lt(pool.date.iloc[internal[0]]).to_numpy()]
    settings = dict(objective='reg:squarederror',eval_metric='rmse',tree_method='hist',learning_rate=.03,
                    subsample=.8,colsample_bytree=.8,max_bin=64,max_depth=6,min_child_weight=3,
                    reg_alpha=.02,reg_lambda=.2,n_jobs=3,random_state=42,verbosity=0)
    center, scale = y[fit_rows].mean(), y[fit_rows].std()
    initial = XGBRegressor(**settings,n_estimators=400,early_stopping_rounds=20)
    initial.fit(x[fit_rows],(y[fit_rows]-center)/scale,sample_weight=weights(y[fit_rows],1.04),
                eval_set=[(x[internal],(y[internal]-center)/scale)],verbose=False)
    center, scale = y[train].mean(), y[train].std()
    model = XGBRegressor(**settings,n_estimators=1+initial.best_iteration)
    model.fit(x[train],(y[train]-center)/scale,sample_weight=weights(y[train],1.04),verbose=False)
    return pool.iloc[test][['ticker','date','maturity','horizonte','actual','dias_corridos']].assign(
        previsao=model.predict(x[test]).astype(float)*scale+center,fase=phase,fold=label)


def main():
    files = {t:legacy.DATA_DIR/f'{t}.csv' for t in TICKERS}
    frames, sets = prepare(files)
    p = pd.concat([frames[t,21].join(indicators(f)[['beta_baixas_63']],on='date',validate='one_to_one')
                   for t,f in files.items()],ignore_index=True).sort_values(['date','ticker'])
    ids = pd.get_dummies(p.ticker,prefix='ticker',dtype=np.float32)
    p = pd.concat([p,ids],axis=1).reset_index(drop=True)
    x = p[sets['simplificadas']+list(ids)+['beta_baixas_63']].to_numpy(np.float32)
    y = p.actual.to_numpy(float)
    base, alternative = [], []
    for fold in FOLDS:
        base.append(fit(p,x,y,1.04,fold)[0])
        alternative.append(flexible(p,x,y,fold))
    reference = pd.concat(base).query('dias_corridos > 1').reset_index(drop=True)
    changed = pd.concat(alternative).query('dias_corridos > 1').reset_index(drop=True)
    pd.testing.assert_frame_equal(reference[KEYS+['actual']],changed[KEYS+['actual']])
    saved = pd.read_csv(PREV/'previsoes_ponderadas_2025.csv',parse_dates=['date','maturity'])
    check = reference.query("fase == 'test'").merge(saved,on=KEYS,validate='one_to_one')
    assert len(check)==len(saved)==1020
    np.testing.assert_allclose(check.previsao_x,check.previsao_y,atol=1e-12,rtol=0)
    experiments = [reference.assign(modelo='referencia'),changed.assign(modelo='arvores_flexiveis')]
    calibration = []
    for name, data in [('referencia',reference),('arvores_flexiveis',changed)]:
        validation = data.query("fase == 'validation'")
        predicted, actual = validation.previsao.to_numpy(), validation.actual.to_numpy()
        for method in ('escala','escala_e_intercepto'):
            z = predicted if method=='escala' else predicted-predicted.mean()
            gain = max(0.,float(z@actual/(z@z)))
            intercept = 0. if method=='escala' else float(actual.mean()-gain*predicted.mean())
            experiments.append(data.assign(previsao=intercept+gain*data.previsao,modelo=f'{name}_{method}'))
            calibration.append(dict(modelo=name,metodo=method,ganho=gain,intercepto=intercept))
    combined = pd.concat(experiments,ignore_index=True)
    metrics = compute_metrics(combined,['fase','modelo','ticker'])
    amplitude = combined.groupby(['fase','modelo','ticker'])[['previsao','actual']].std().reset_index()
    amplitude['amplitude_pct'] = 100*amplitude.previsao/amplitude.actual
    metrics = metrics.merge(amplitude[['fase','modelo','ticker','amplitude_pct']],validate='one_to_one')
    report = metrics.groupby(['fase','modelo'])[['rmse','mae','r2','balanceada_pct','amplitude_pct']].mean()
    report[['rmse','mae']] *= 100
    report.to_csv(OUT/'resumo.csv')
    metrics.to_csv(OUT/'metricas_por_acao.csv',index=False)
    pd.DataFrame(calibration).to_csv(OUT/'calibracao_2024.csv',index=False)
    print(report.round(5).to_string())
    print(pd.DataFrame(calibration).round(5).to_string(index=False))


if __name__ == '__main__':
    main()
