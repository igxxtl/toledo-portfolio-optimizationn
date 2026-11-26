from pathlib import Path
import sys, os
import pandas as pd
import numpy as np
out=Path('experimentos/fatores_artigo_2026_11_25').resolve()
out.mkdir(exist_ok=True)
prior=out.parent/'features_baixas_2026_11_25'/'pesos_baixas'
sys.path.insert(0,str(prior))
from treinar import FOLDS, KEYS, TICKERS, fit, indicators, legacy, prepare
from comparar_modelos import compute_metrics
os.environ['MPLCONFIGDIR']=str(out/'cache_matplotlib')
files={t:legacy.DATA_DIR/f'{t}.csv' for t in TICKERS}
frames,sets=prepare(files)
def wilder(s,n):
    valid=s.dropna()
    seed=pd.Series(index=s.index,dtype=float)
    if len(valid)>=n:
        seed.loc[valid.index[n-1]]=valid.iloc[:n].mean()
        seed.loc[valid.index[n:]]=valid.iloc[n:]
    return seed.ewm(alpha=1/n,adjust=False,ignore_na=True).mean()
def extra(path):
    raw=pd.read_csv(path,parse_dates=['date']).set_index('date')
    p=raw.close
    hi=raw.high.where(raw.high.gt(0))
    lo=raw.low.where(raw.low.gt(0))
    tr=pd.concat([hi-lo,(hi-p.shift(1)).abs(),(lo-p.shift(1)).abs()],axis=1).max(axis=1)
    up,down=hi.diff(),-lo.diff()
    plus=wilder(up.where(up.gt(down)&up.gt(0),0),14)
    minus=wilder(down.where(down.gt(up)&down.gt(0),0),14)
    dx=100*(plus-minus).abs()/(plus+minus).replace(0,np.nan)
    adx=wilder(dx,14)
    ema12=p.ewm(span=12,adjust=False,min_periods=12).mean()
    ema26=p.ewm(span=26,adjust=False,min_periods=26).mean()
    macd=ema12-ema26
    signal=macd.ewm(span=9,adjust=False,min_periods=9).mean()
    dp=p.diff()
    gain=wilder(dp.clip(lower=0),14)
    loss=wilder(-dp.clip(upper=0),14)
    rsi=100*gain/(gain+loss).replace(0,np.nan)
    return indicators(path)[['beta_baixas_63']].assign(
        art_atr=wilder(tr,14)/p,art_adxr=(adx+adx.shift(14))/200,
        art_macd=macd/p,art_macd_signal=signal/p,
        art_dist_ema=p/ema26-1,art_dist_sma=p/p.rolling(21).mean()-1,art_rsi=rsi/100
    ).replace([np.inf,-np.inf],np.nan)

p=pd.concat([frames[t,21].join(extra(path),on='date',validate='one_to_one') for t,path in files.items()],ignore_index=True).sort_values(['date','ticker'])
ids=pd.get_dummies(p.ticker,prefix='ticker',dtype=np.float32)
p=pd.concat([p,ids],axis=1).reset_index(drop=True)
base=sets['simplificadas']+list(ids)+['beta_baixas_63']
y=p.actual.to_numpy()
runs=[]
for model,features in [('referencia',[]),('atr_adxr_macd',['art_atr','art_adxr','art_macd','art_macd_signal']),('sete_fatores',['art_atr','art_adxr','art_macd','art_macd_signal','art_dist_ema','art_dist_sma','art_rsi'])]:
    for fold in FOLDS:
        pred,_=fit(p,p[base+features].to_numpy(np.float32),y,1.04,fold)
        runs.append(pred.assign(modelo=model))
    print(model,flush=True)
data=pd.concat(runs).query('dias_corridos > 1')
assert data.groupby(KEYS).size().eq(3).all()
ref=pd.read_csv(prior/'previsoes_ponderadas_2025.csv',parse_dates=['date','maturity'])
match=data.query("modelo == 'referencia' and fase == 'test'").merge(ref,on=KEYS,validate='one_to_one')
assert len(match)==len(ref)==1020
np.testing.assert_allclose(match.previsao_x,match.previsao_y,rtol=0,atol=1e-12)
by=compute_metrics(data,['fase','modelo','ticker'])
summary=by.groupby(['fase','modelo'])[['rmse','mae','r2','altas_pct','baixas_pct','balanceada_pct']].mean()
summary[['rmse','mae']]*=100
summary.to_csv(out/'resumo.csv')
by.to_csv(out/'metricas_por_acao.csv',index=False)
data.to_csv(out/'previsoes.csv',index=False)
print(summary.round(5).to_string())
