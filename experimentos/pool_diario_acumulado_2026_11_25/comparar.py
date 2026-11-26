"""Compara previsão direta e soma de 21 previsões diárias com origem fixa."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

OUT = Path(__file__).resolve().parent
PREVIOUS = OUT.parent / 'features_baixas_2026_11_25' / 'pesos_baixas'
DIRECT = OUT.parent / 'alvo_log_vs_simples_2026_11_25' / 'previsoes_em_retorno_simples.csv'
sys.path.insert(0, str(PREVIOUS))
from treinar import FOLDS, KEYS, TICKERS, indicators, legacy, prepare, weights
from comparar_modelos import compute_metrics

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
HORIZON = 21
STEPS = np.arange(1, HORIZON + 1, dtype=np.float32)


def daily_labels(path):
    price = pd.read_csv(path, usecols=['date', 'close'], parse_dates=['date']).set_index('date').close
    step_return = np.log(price).diff()
    return pd.DataFrame({f'passo_{k:02d}': step_return.shift(-k) for k in range(1,HORIZON+1)})


def expanded(x, origins):
    # Uma linha por origem e passo. As colunas de mercado permanecem fixas na origem.
    values = np.empty((len(origins), HORIZON, x.shape[1] + 1), dtype=np.float32)
    values[:, :, :-1] = x[origins, None, :]
    values[:, :, -1] = STEPS
    return values.reshape(-1, values.shape[-1])


def rmse_acumulado(labels, predictions):
    total_error = (labels-predictions).reshape(-1,HORIZON).sum(axis=1)
    return float(np.sqrt(np.mean(total_error**2)))


def fit(frame, x, daily, multiplier, fold):
    name, start, end, phase = fold
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cutoff = pd.Timestamp('2025-01-01' if phase=='validation' else '2025-12-31')
    train = np.flatnonzero((frame.date.lt(start) & frame.maturity.lt(start)).to_numpy())
    test = np.flatnonzero((frame.date.ge(start) & frame.date.lt(end) & frame.maturity.lt(cutoff)).to_numpy())
    cut = int(len(train)*.8)
    internal = train[cut:]
    boundary = frame.date.iloc[internal[0]]
    initial = train[:cut]
    initial = initial[frame.maturity.iloc[initial].lt(boundary).to_numpy()]
    assert frame.maturity.iloc[train].max() < start and frame.maturity.iloc[initial].max() < boundary
    params = dict(objective='reg:squarederror', eval_metric=rmse_acumulado,tree_method='hist',
        learning_rate=.03, subsample=.8, colsample_bytree=.8, max_bin=64, max_depth=4,
        min_child_weight=10, reg_alpha=.1, reg_lambda=1., n_jobs=4, random_state=42, verbosity=0)
    y = frame.actual.to_numpy(float)
    center, scale = float(daily[initial].mean()), max(float(daily[initial].std()),1e-8)
    early = XGBRegressor(**params,n_estimators=400,early_stopping_rounds=20)
    early.fit(expanded(x,initial),((daily[initial]-center)/scale).ravel(),
              sample_weight=np.repeat(weights(y[initial],multiplier),HORIZON),
              eval_set=[(expanded(x,internal),((daily[internal]-center)/scale).ravel())],verbose=False)
    rounds = int(early.best_iteration)+1
    del early
    center, scale = float(daily[train].mean()), max(float(daily[train].std()),1e-8)
    model = XGBRegressor(**params,n_estimators=rounds)
    model.fit(expanded(x,train),((daily[train]-center)/scale).ravel(),
              sample_weight=np.repeat(weights(y[train],multiplier),HORIZON),verbose=False)
    pred = model.predict(expanded(x,test)).astype(float).reshape(-1,HORIZON)*scale+center
    result = frame.iloc[test][['ticker','date','maturity','horizonte','dias_corridos','actual']].reset_index(drop=True)
    result = result.join(pd.DataFrame(pred,columns=[f'previsao_{k:02d}' for k in range(1,HORIZON+1)]))
    result = result.assign(previsao_acumulada=pred.sum(axis=1),fase=phase,fold=name)
    print(f'{name}: {rounds} árvores, {len(train):,} origens de treino.',flush=True)
    return result,dict(fold=name,origens_treino=len(train),exemplos_diarios=len(train)*HORIZON,
                       rounds=rounds,max_train_maturity=str(frame.maturity.iloc[train].max().date()))


def main():
    protocol = PREVIOUS/'protocolo.json'
    multiplier = json.loads(protocol.read_text(encoding='utf-8'))['peso_escolhido']
    files = {ticker:legacy.DATA_DIR/f'{ticker}.csv' for ticker in TICKERS}
    sources = [protocol,DIRECT,*files.values(),*legacy.OUT_DIR.glob('*_predicoes.csv'),*OUT.parents[1].glob('**/*.tex')]
    hashes = {p:sha256(p.read_bytes()).hexdigest() for p in sources}
    frames, sets = prepare(files)
    pool = pd.concat([frames[ticker,HORIZON].join(indicators(path)[['beta_baixas_63']],on='date',validate='one_to_one')
                      .join(daily_labels(path),on='date',validate='one_to_one') for ticker,path in files.items()],
                     ignore_index=True).sort_values(['date','ticker'])
    ids = pd.get_dummies(pool.ticker,prefix='ticker',dtype=np.float32)
    pool = pd.concat([pool,ids],axis=1).reset_index(drop=True)
    cols = sets['simplificadas']+ids.columns.tolist()+['beta_baixas_63']
    x = pool[cols].to_numpy(np.float32)
    y = pool[[f'passo_{k:02d}' for k in range(1,HORIZON+1)]].to_numpy(float)
    np.testing.assert_allclose(y.sum(axis=1),pool.actual,rtol=0,atol=1e-12)
    packets, audit = [], []
    for fold in FOLDS:
        predicted, record = fit(pool,x,y,multiplier,fold)
        packets.append(predicted)
        audit.append(record)
    accumulated = pd.concat(packets,ignore_index=True)
    direct = pd.read_csv(DIRECT,parse_dates=['date','maturity']).query("alvo == 'log'")
    paired = accumulated.merge(direct[KEYS+['actual','previsao']],on=KEYS,suffixes=('','_direta'),validate='one_to_one')
    assert len(paired)==len(accumulated)==len(direct)
    np.testing.assert_allclose(paired.actual,np.log1p(paired.pop('actual_direta')),rtol=0,atol=1e-12)
    paired['previsao_direta'] = np.log1p(paired.pop('previsao'))
    paired.to_csv(OUT/'previsoes_pareadas.csv',index=False)

    parts, pooled = [], []
    for name,sample in [('todos',paired),('apos_pausa',paired[paired.dias_corridos.gt(1)])]:
        for model in ['direta','acumulada']:
            data = sample.assign(previsao=sample[f'previsao_{model}'])
            parts.append(compute_metrics(data,['fase','ticker']).assign(recorte=name,modelo=model))
            for phase,group in data.groupby('fase'):
                pooled.append(dict(recorte=name,fase=phase,modelo=model,
                    r2_agregado=1-(group.actual-group.previsao).pow(2).mean()/group.actual.var(ddof=0)))
    by_asset = pd.concat(parts,ignore_index=True)
    summary = by_asset.groupby(['recorte','fase','modelo']).agg(n=('n','sum'),
        rmse=('rmse','mean'),mae=('mae','mean'),r2=('r2','mean'),altas=('altas_pct','mean'),
        baixas=('baixas_pct','mean'),balanceada=('balanceada_pct','mean')).reset_index().merge(
            pd.DataFrame(pooled),on=['recorte','fase','modelo'],validate='one_to_one')
    summary[['rmse','mae']] *= 100
    summary.to_csv(OUT/'resumo.csv',index=False)
    by_asset.to_csv(OUT/'metricas_por_acao.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'auditoria.csv',index=False)
    assert all(sha256(path.read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    (OUT/'protocolo.json').write_text(json.dumps(dict(horizonte=HORIZON,atributos_base=len(cols),
        atributos_pool_diario=len(cols)+1,peso_baixas_21=multiplier,arquivos_conferidos=len(hashes),
        previsoes_pareadas=len(paired)),indent=2),encoding='utf-8')
    print(summary.round(5).to_string(index=False))


if __name__=='__main__':
    main()
