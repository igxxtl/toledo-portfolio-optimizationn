"""Busca retrospectiva de atributos e objetivos para o pool de 21 pregões."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

OUT = Path(__file__).resolve().parent
EXPERIMENTS = OUT.parent
PREVIOUS = EXPERIMENTS / 'features_baixas_2026_11_25'
sys.path.insert(0, str(PREVIOUS / 'pesos_baixas'))
from treinar import FOLDS, KEYS, TICKERS, indicators, legacy, prepare

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
SCALES = (.25, .5, .75, 1.)
OBJECTIVES = ('quadratico', 'pseudo_huber', 'absoluto', 'quadratico_vol', 'ridge')


def market_features(prices):
    daily = np.log(prices).diff()
    monthly = np.log(prices / prices.shift(21))
    return pd.DataFrame({
        'universo_fracao_altas_5': daily.gt(0).where(daily.notna()).mean(axis=1).rolling(5).mean(),
        'universo_mediana_5': np.log(prices / prices.shift(5)).median(axis=1),
        'universo_dispersao_21': monthly.std(axis=1),
        'universo_fracao_abaixo_da_media_21': prices.lt(prices.rolling(21).mean()).where(monthly.notna()).mean(axis=1),
    }).where(daily.count(axis=1).ge(15))


def extra_features(path, market, returns_by_asset):
    raw = pd.read_csv(path, parse_dates=['date']).set_index('date')
    ibov = np.log(raw.ibov_close).diff()
    r = np.log(raw.close).diff()
    volatility21 = r.rolling(21).std()
    risk_scale = (volatility21 * np.sqrt(21)).clip(lower=.005)
    features = indicators(path).assign(
        risco_normalizador=risk_scale,
        regime_ibov_vol_5_63=ibov.rolling(5).std()/ibov.rolling(63).std().replace(0, np.nan),
        retorno_ibov_5=ibov.rolling(5).sum(),
        retorno_acão_5=r.rolling(5).sum(),
        volatilidade_21=volatility21,
    ).join(market)
    cross_ranks = returns_by_asset.rank(axis=1, pct=True).where(returns_by_asset.count(axis=1).ge(15))
    features['rank_universo_retorno_21'] = cross_ranks[path.stem]
    return features.replace([np.inf,-np.inf], np.nan)


def fit_block(pool, columns, objective, fold):
    label, start, end, phase = fold
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cutoff = pd.Timestamp('2025-01-01' if phase == 'validation' else '2025-12-31')
    train = np.flatnonzero((pool.date.lt(start) & pool.maturity.lt(start)).to_numpy())
    test = np.flatnonzero((pool.date.ge(start) & pool.date.lt(end) & pool.maturity.lt(cutoff)).to_numpy())
    x, y = pool[columns].to_numpy(np.float32), pool.actual.to_numpy(float)
    normalizer = pool.risco_normalizador.to_numpy(float) if objective == 'quadratico_vol' else np.ones(len(y))
    assert np.isfinite(normalizer).all() and (normalizer > 0).all()
    if objective == 'ridge':
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=100.))
        model.fit(x[train], y[train])
        predictions = model.predict(x[test])
        rounds = None
    else:
        y = y / normalizer
        cut = int(len(train)*.8)
        internal = train[cut:]
        start_internal = pool.date.iloc[internal[0]]
        initial = train[:cut]
        initial = initial[pool.maturity.iloc[initial].lt(start_internal).to_numpy()]
        assert len(initial) >= 100 and pool.maturity.iloc[initial].max() < start_internal
        name = {'quadratico':'reg:squarederror','quadratico_vol':'reg:squarederror',
                'pseudo_huber':'reg:pseudohubererror','absoluto':'reg:absoluteerror'}[objective]
        params = dict(objective=name, eval_metric='rmse', tree_method='hist', learning_rate=.03,
            subsample=.8, colsample_bytree=.8, max_bin=64, max_depth=4, min_child_weight=10,
            reg_alpha=.1, reg_lambda=1., n_jobs=3, random_state=42, verbosity=0)
        center, scale = float(y[initial].mean()), max(float(y[initial].std()),1e-8)
        early = XGBRegressor(**params, n_estimators=400, early_stopping_rounds=20)
        early.fit(x[initial],(y[initial]-center)/scale,sample_weight=np.ones(len(initial)),
                  eval_set=[(x[internal],(y[internal]-center)/scale)], verbose=False)
        rounds = int(early.best_iteration)+1
        center, scale = float(y[train].mean()), max(float(y[train].std()),1e-8)
        model = XGBRegressor(**params, n_estimators=rounds)
        model.fit(x[train],(y[train]-center)/scale,sample_weight=np.ones(len(train)),verbose=False)
        predictions = (model.predict(x[test]).astype(float)*scale+center)*normalizer[test]
    assert np.isfinite(predictions).all() and pool.maturity.iloc[train].max() < start
    result = pool.iloc[test][['ticker','date','maturity','horizonte','dias_corridos','actual']].copy()
    result = result.assign(previsao=predictions, fase=phase, fold=label)
    return result, dict(fold=label, n_train=len(train), rounds=rounds,
                        max_train_maturity=str(pool.maturity.iloc[train].max().date()))


def stats(frame, scale, threshold):
    parts = []
    for ticker, group in frame.groupby('ticker'):
        y, pred = group.actual.to_numpy(), scale * group.previsao.to_numpy()
        mse = np.mean((y-pred)**2)
        up, down = y>0, y<0
        assert up.any() and down.any() and np.var(y)>0
        up_rate, down_rate = (pred[up]>threshold).mean(), (pred[down]<threshold).mean()
        parts.append(dict(ticker=ticker, n=len(y), rmse=100*np.sqrt(mse), mae=100*np.abs(y-pred).mean(),
                          r2=1-mse/np.var(y), skill_zero=1-mse/np.mean(y**2),
                          altas=100*up_rate, baixas=100*down_rate, balanceada=50*(up_rate+down_rate)))
    detailed = pd.DataFrame(parts)
    y, pred = frame.actual.to_numpy(), scale * frame.previsao.to_numpy()
    return dict(n=len(frame), **detailed.drop(columns=['ticker','n']).mean().to_dict(),
                r2_agregado=1-np.mean((y-pred)**2)/np.var(y), acoes_r2_positivo=int(detailed.r2.gt(0).sum())), detailed


def decision_threshold(frame, scale):
    y, p = frame.actual.to_numpy(), scale*frame.previsao.to_numpy()
    scores = np.unique(p)
    thresholds = np.unique(np.r_[0.,scores[0]-1e-9,scores[:-1]+np.diff(scores)/2,scores[-1]+1e-9])
    ups, downs = y>0, y<0
    up, down = frame.loc[ups], frame.loc[downs]
    uw = 1/up.groupby('ticker').ticker.transform('size').to_numpy()/up.ticker.nunique()
    dw = 1/down.groupby('ticker').ticker.transform('size').to_numpy()/down.ticker.nunique()
    recall_up = 100*(uw @ (p[ups,None]>thresholds))
    recall_down = 100*(dw @ (p[downs,None]<thresholds))
    table = pd.DataFrame(dict(threshold=thresholds, balance=(recall_up+recall_down)/2,
                              pior_classe=np.minimum(recall_up,recall_down), distancia_zero=np.abs(thresholds)))
    feasible = table[table.pior_classe.ge(40)]
    if not len(feasible):
        return float(table.sort_values(['pior_classe','balance','distancia_zero'],ascending=[False,False,True]).threshold.iloc[0])
    return float(feasible.sort_values(['balance','pior_classe','distancia_zero'],ascending=[False,False,True]).threshold.iloc[0])


def main():
    base_path = PREVIOUS/'previsoes.csv'
    weighted_path = PREVIOUS/'pesos_baixas/previsoes_ponderadas_2025.csv'
    files = {ticker:legacy.DATA_DIR/f'{ticker}.csv' for ticker in TICKERS}
    protected = [*files.values(),*legacy.OUT_DIR.glob('*_predicoes.csv'),*OUT.parents[1].glob('**/*.tex'),
                 Path(legacy.__file__),base_path,weighted_path]
    hashes = {path:sha256(path.read_bytes()).hexdigest() for path in protected}
    raw_prices = pd.DataFrame({ticker:pd.read_csv(path,parse_dates=['date']).set_index('date').close
                               for ticker,path in files.items()})
    market = market_features(raw_prices)
    monthly = np.log(raw_prices/raw_prices.shift(21))
    frames, sets = prepare(files)
    pool = pd.concat([frames[ticker,21].join(extra_features(path,market,monthly),on='date',validate='one_to_one')
                      for ticker,path in files.items()],ignore_index=True).sort_values(['date','ticker'])
    identities = pd.get_dummies(pool.ticker,prefix='ticker',dtype=np.float32)
    pool = pd.concat([pool,identities],axis=1).reset_index(drop=True)
    base = sets['simplificadas']+identities.columns.tolist()+['beta_baixas_63']
    stress = ['drawdown_21','regime_ibov_vol_5_63','retorno_ibov_5','fracao_risco_baixa_ibov_21']
    transversal = [*market.columns,'rank_universo_retorno_21','retorno_relativo_5']
    inputs = {'beta':base,'estresse':base+stress,'transversal':base+transversal,
              'combinado':base+stress+transversal+['salto_abertura','volume_em_dias_de_queda_21',
                                                   'retorno_acão_5','volatilidade_21']}
    validation_folds = [fold for fold in FOLDS if fold[-1]=='validation']
    test_folds = [fold for fold in FOLDS if fold[-1]=='test']
    saved = pd.read_csv(base_path,parse_dates=['date','maturity']).query("dias_corridos > 1 and modelo in ['referencia','beta_baixas_63']")
    base_val = saved.query("fase == 'validation' and modelo == 'beta_baixas_63'")
    by_validation, audits = [], []
    for group,columns in inputs.items():
        for objective in OBJECTIVES:
            pieces = []
            for fold in validation_folds:
                prediction,audit = fit_block(pool,columns,objective,fold)
                pieces.append(prediction)
                audits.append(dict(features=group,objetivo=objective,**audit))
            valid = pd.concat(pieces,ignore_index=True).query('dias_corridos > 1')
            match = valid.merge(base_val[KEYS+['previsao','actual']],on=KEYS,validate='one_to_one')
            assert len(match)==len(valid)==len(base_val)
            np.testing.assert_allclose(match.actual_x,match.actual_y,atol=1e-12,rtol=0)
            if group=='beta' and objective=='quadratico':
                np.testing.assert_allclose(match.previsao_x,match.previsao_y,atol=1e-12,rtol=0)
            scale = max(SCALES,key=lambda value:stats(valid,value,0.)[0]['r2'])
            threshold = decision_threshold(valid,scale)
            result,_ = stats(valid,scale,threshold)
            by_validation.append(dict(features=group,objetivo=objective,escala=scale,limiar=threshold,
                                      balanceada_zero=stats(valid,scale,0.)[0]['balanceada'],**result))
            print(f'{group}/{objective}: validação concluída.',flush=True)

    grid = pd.DataFrame(by_validation)
    grid['rank_conjunto'] = grid.r2.rank(pct=True)+grid.balanceada.rank(pct=True)
    grid.to_csv(OUT/'grade_2024.csv',index=False)
    viable = grid[grid.r2.gt(0)]
    picks = {'compromisso':(viable if len(viable) else grid).sort_values('rank_conjunto',ascending=False).iloc[0],
             'melhor_r2':grid.sort_values('r2',ascending=False).iloc[0],
             'melhor_direcional':grid.sort_values('balanceada',ascending=False).iloc[0]}
    chosen = pd.DataFrame([dict(criterio=name,**row.to_dict()) for name,row in picks.items()])
    chosen.to_csv(OUT/'escolhas_2024.csv',index=False)
    test_base = saved.query("fase == 'test' and modelo == 'referencia'")
    weighted = pd.read_csv(weighted_path,parse_dates=['date','maturity'])
    comparisons = [('referencia',test_base,1.,0.), ('beta_ponderado',weighted,1.,0.)]
    forecasts = []
    for _,choice in chosen.drop_duplicates(['features','objetivo']).iterrows():
        group, objective = choice.features, choice.objetivo
        name = f'{group}_{objective}'
        pieces = []
        for fold in test_folds:
            prediction,audit = fit_block(pool,inputs[group],objective,fold)
            audits.append(dict(features=group,objetivo=objective,**audit))
            pieces.append(prediction)
        test = pd.concat(pieces,ignore_index=True).query('dias_corridos > 1')
        forecasts.append(test.assign(modelo=name))
        comparisons.extend([(name+'_calibrado',test,choice.escala,choice.limiar),
                            (name+'_zero',test,choice.escala,0.)])
        print(f'{name}: teste concluído com a escolha congelada.',flush=True)
    results, details = [], []
    for name,test,scale,threshold in comparisons:
        check = test.merge(test_base[KEYS+['actual']],on=KEYS,validate='one_to_one')
        assert len(check)==len(test)==len(test_base)
        np.testing.assert_allclose(check.actual_x,check.actual_y,atol=1e-12,rtol=0)
        result,detailed = stats(test,scale,threshold)
        results.append(dict(modelo=name,escala=scale,limiar=threshold,**result))
        details.append(detailed.assign(modelo=name))
    report = pd.DataFrame(results)
    report.to_csv(OUT/'resumo_2025.csv',index=False)
    pd.concat(details,ignore_index=True).to_csv(OUT/'metricas_por_acao_2025.csv',index=False)
    pd.concat(forecasts,ignore_index=True).to_csv(OUT/'previsoes_2025.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'auditoria.csv',index=False)
    assert all(sha256(path.read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    (OUT/'protocolo.json').write_text(json.dumps(dict(modelos_busca=len(grid),arquivos_conferidos=len(hashes),
        janela_calibracao=2024,janela_teste=2025,horizonte=21,recorte='dias_corridos > 1',
        referencia_reproduzida=True),indent=2),encoding='utf-8')
    print(chosen.round(5).to_string(index=False),flush=True)
    print(report.round(5).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
