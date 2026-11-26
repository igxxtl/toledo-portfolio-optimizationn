"""Conjuntos economicos de features causais para a rodada experimental."""
import numpy as np
import pandas as pd

CORE=['lag_ret_1','lag_ret_5','lag_ret_21','mom_21','mom_63','mom_126','mom_252',
      'vol_21','vol_63','vol_rel_21','vol_rel_63','obv_z21','force_idx_z21','vwap_dist_21',
      'rsi_21','dist_ema_21','dist_ema_63','bb_pctb_21','beta_63','macro_ibov_ret',
      'macro_usdbrl_ret','cs_rank_21','cs_rank_63','cs_rank_126']


def logret(s,h=1):return np.log(s/s.shift(h))


def breadth_frames(raw_prices):
    """Estatisticas transversais de retornos ja observados, sem janelas futuras."""
    out=pd.DataFrame(index=raw_prices.index)
    for h in [1,5,21]:
        r=logret(raw_prices,h)
        count=r.notna().sum(axis=1)
        out[f'mkt_median_{h}']=r.median(axis=1).where(count>=10)
        out[f'mkt_breadth_{h}']=r.gt(0).sum(axis=1).div(count.replace(0,np.nan)).where(count>=10)
        out[f'mkt_dispersion_{h}']=r.std(axis=1,ddof=0).where(count>=10)
    return out


def extra_features(raw,breadth):
    p=raw.close.astype(float)
    o=raw.open.astype(float).where(raw.open>0)
    hi=raw.high.astype(float).where(raw.high>0);lo=raw.low.astype(float).where(raw.low>0)
    ret=logret(p);market=logret(raw.ibov_close.astype(float));spread=(hi-lo).replace(0,np.nan)
    vol21=ret.rolling(21).std();vol63=ret.rolling(63).std()
    out=pd.DataFrame(index=raw.index)
    for h in [1,3,5,10]:out[f'ex_ret_{h}']=logret(p,h)
    out['ex_gap']=np.log(o/p.shift(1));out['ex_intraday']=np.log(p/o)
    out['ex_range']=np.log(hi/lo);out['ex_candle_body']=(p-o)/spread
    out['ex_close_location']=(p-lo)/spread
    out['ex_atr_pct']=(pd.concat([hi-lo,(hi-p.shift()).abs(),(lo-p.shift()).abs()],axis=1).max(axis=1).rolling(21).mean()/p)
    out['ex_vol_fast_slow']=ret.rolling(5).std()/vol21.replace(0,np.nan)
    out['ex_vol_regime']=vol21/vol63.replace(0,np.nan)
    out['ex_downside_vol']=np.sqrt(ret.clip(upper=0).pow(2).rolling(21).mean())
    out['ex_positive_fraction']=ret.gt(0).astype(float).rolling(21).mean()
    out['ex_skew']=ret.rolling(21).skew()
    for h in [21,63]:out[f'ex_drawdown_{h}']=p/p.rolling(h).max()-1
    beta=ret.rolling(63).cov(market)/market.rolling(63).var().replace(0,np.nan)
    # Cada residual usa somente beta disponivel na propria data.
    residual=ret-beta*market
    out['ex_idio_vol']=residual.rolling(21).std()
    for h in [5,21]:out[f'ex_residual_sum_{h}']=residual.rolling(h).sum()
    if 'usdbrl_close' in raw:
        for h in [5,21]:out[f'ex_fx_{h}']=logret(raw.usdbrl_close.astype(float),h)
    for h in [5,21]:out[f'ex_relative_return_{h}']=logret(p,h)-logret(raw.ibov_close.astype(float),h)
    extra=list(out.columns)
    aligned=breadth.reindex(pd.DatetimeIndex(raw.date)).reset_index(drop=True)
    for c in aligned:out[c]=aligned[c].to_numpy()
    for h in [5,21]:out[f'mkt_own_deviation_{h}']=logret(p,h)-out[f'mkt_median_{h}']
    mvol21=market.rolling(21).std();mvol63=market.rolling(63).std()
    out['mkt_vol_regime']=mvol21/mvol63.replace(0,np.nan)
    out=out.replace([np.inf,-np.inf],np.nan)
    return out,extra,[c for c in out if c not in extra]


def select_columns(X,training_rows,columns,threshold=.98):
    """Filtro sem y, ajustado somente no fit anterior ao early stopping."""
    sample=X.iloc[training_rows][columns]
    kept=[];removed=[]
    for c in columns:
        if sample[c].notna().mean()<.9:
            removed.append(dict(feature=c,reason='missing_above_10pct',related=''));continue
        if sample[c].nunique(dropna=True)<=1:
            removed.append(dict(feature=c,reason='constant',related=''));continue
        kept.append(c)
    if not kept:raise ValueError('Nenhuma feature elegivel')
    correlation=sample[kept].corr(method='spearman').abs()
    final=[]
    for c in kept:
        redundant=next((k for k in final if correlation.loc[c,k]>threshold),None)
        if redundant is not None:removed.append(dict(feature=c,reason='spearman_above_098',related=redundant))
        else:final.append(c)
    return final,removed
