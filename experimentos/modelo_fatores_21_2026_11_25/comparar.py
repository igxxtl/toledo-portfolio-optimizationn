"""Previsão de 21 pregões decomposta em mercado e residual de cada ação."""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
PRIOR = OUT.parent / 'features_baixas_2026_11_25' / 'pesos_baixas'
sys.path.insert(0, str(PRIOR))
from treinar import FOLDS, KEYS, TICKERS, fit, indicators, legacy, prepare
from comparar_modelos import compute_metrics

os.environ['MPLCONFIGDIR'] = str(OUT / 'cache_matplotlib')
H = 21


def market_data(path):
    prices = pd.read_csv(path, usecols=['date','close'], parse_dates=['date']).set_index('date').close
    assert prices.index.is_monotonic_increasing and not prices.index.has_duplicates
    assert prices.gt(0).all()
    r = np.log(prices).diff()
    features = pd.DataFrame({
        'retorno_5': np.log(prices/prices.shift(5)),
        'retorno_21': np.log(prices/prices.shift(21)),
        'retorno_63': np.log(prices/prices.shift(63)),
        'vol_21': r.rolling(21).std(),
        'vol_63': r.rolling(63).std(),
        'drawdown_63': prices/prices.rolling(63).max()-1,
        'dias_corridos': prices.index.to_series().diff().dt.days,
    })
    frame = features.assign(maturidade=prices.index.to_series().shift(-H),
                            retorno=np.log(prices.shift(-H)/prices))
    return prices, frame.dropna(), features.columns.tolist()


def predict_market(market, features, block, start):
    train = market[market.maturidade.lt(pd.Timestamp(start))]
    test = market.loc[block.date]
    assert len(train)>100 and train.index.max()<pd.Timestamp(start)
    # O horizonte efetivo do índice precisa coincidir com o horizonte da ação.
    np.testing.assert_array_equal(test.maturidade.to_numpy(),block.maturity.to_numpy())
    model = make_pipeline(StandardScaler(),Ridge(alpha=40.))
    model.fit(train[features],train.retorno)
    return model.predict(test[features]),float(train.retorno.mean()),len(train)


def main():
    previous = PRIOR / 'previsoes_ponderadas_2025.csv'
    files = {ticker:legacy.DATA_DIR/f'{ticker}.csv' for ticker in TICKERS}
    index_path = legacy.DATA_DIR/'IBOV.csv'
    protected = [previous,index_path,*files.values(),*legacy.OUT_DIR.glob('*_predicoes.csv'),
                 *OUT.parents[1].glob('**/*.tex')]
    hashes = {path:sha256(path.read_bytes()).hexdigest() for path in protected}
    index_prices, market, market_columns = market_data(index_path)
    frames, sets = prepare(files)
    pool = pd.concat([frames[t,H].join(indicators(path)[['beta_baixas_63']],on='date',validate='one_to_one')
                      for t,path in files.items()],ignore_index=True).sort_values(['date','ticker'])
    identifiers = pd.get_dummies(pool.ticker,prefix='ticker',dtype=np.float32)
    pool = pd.concat([pool,identifiers],axis=1).reset_index(drop=True)
    columns = sets['simplificadas'] + list(identifiers) + ['beta_baixas_63']
    x = pool[columns].to_numpy(np.float32)
    y = pool.actual.to_numpy(float)
    beta = pool.eco_beta_63.to_numpy(float)
    realized_market = np.log(pool.maturity.map(index_prices)/pool.date.map(index_prices)).to_numpy(float)
    residual = y-beta*realized_market
    assert np.isfinite(residual).all()
    np.testing.assert_allclose(beta*realized_market+residual,y,atol=1e-12,rtol=0)
    auxiliary = pool[['ticker','date','maturity']].assign(retorno_acao=y, beta=beta, realizado_ibov=realized_market)
    predictions, audits = [], []
    for fold in FOLDS:
        # Peso uniforme no residual. O peso de 1,04 pertence ao alvo original do comparador.
        block, audit = fit(pool,x,residual,1.,fold)
        block = block.rename(columns={'previsao':'previsao_residual'}).drop(columns='actual')
        block = block.merge(auxiliary,on=['ticker','date','maturity'],validate='one_to_one')
        market_pred, historical, n_market = predict_market(market,market_columns,block,fold[1])
        block = block.rename(columns={'retorno_acao':'actual'}).assign(previsao_ibov=market_pred,media_ibov=historical)
        predictions.append(block.assign(fatores_ridge=block.beta*market_pred+block.previsao_residual,
            fatores_media=block.beta*historical+block.previsao_residual, somente_residual=block.previsao_residual))
        audits.append(dict(n_ibov_treino=n_market,**audit))
        print(f'{fold[0]}: mercado e residual estimados.',flush=True)
    forecasts = pd.concat(predictions,ignore_index=True)
    reference = pd.read_csv(previous,parse_dates=['date','maturity'])
    test = forecasts.query("fase == 'test' and dias_corridos > 1").merge(
        reference[KEYS+['actual','previsao']],on=KEYS,suffixes=('','_referencia'),validate='one_to_one')
    assert len(test)==len(reference)==1020
    np.testing.assert_allclose(test.actual,test.pop('actual_referencia'),rtol=0,atol=1e-12)
    test = test.rename(columns={'previsao':'pool_atual'})
    validation = forecasts.query("fase == 'validation' and dias_corridos > 1")
    groups = []
    definitions = ('fatores_ridge','fatores_media','somente_residual')
    for data, models in [(validation,definitions),(test,('pool_atual',*definitions))]:
        for name in models:
            group = compute_metrics(data.assign(previsao=data[name]),['fase','ticker'])
            groups.append(group.assign(modelo=name))
    by_asset = pd.concat(groups,ignore_index=True)
    summary = by_asset.groupby(['fase','modelo']).agg(n=('n','sum'),rmse=('rmse','mean'),mae=('mae','mean'),
        r2=('r2','mean'),altas=('altas_pct','mean'),baixas=('baixas_pct','mean'),
        balanceada=('balanceada_pct','mean'),r2_positivo=('r2',lambda s:s.gt(0).sum())).reset_index()
    summary[['rmse','mae']] *= 100
    assert all(sha256(path.read_bytes()).hexdigest()==digest for path,digest in hashes.items())
    summary.to_csv(OUT/'resumo.csv',index=False)
    by_asset.to_csv(OUT/'metricas_por_acao.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'auditoria.csv',index=False)
    test.to_csv(OUT/'previsoes_pareadas_2025.csv',index=False)
    (OUT/'protocolo.json').write_text(json.dumps(dict(horizonte=H,peso_residual=1.,ridge_alpha=40.,
        arquivos_conferidos=len(hashes),previsoes_pareadas=len(test)),indent=2),encoding='utf-8')
    print(summary.round(5).to_string(index=False))


if __name__ == '__main__':
    main()
