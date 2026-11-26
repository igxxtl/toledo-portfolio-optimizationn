"""Conecta os sete controles preditivos ao mesmo motor financeiro, sem tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .camadas_intermediarias import OUT
from .avaliar_camadas_preditivas import LAYERS
from intel.Blacklitterman import simular_dois_cenarios as engine


def main():
    dest=OUT/'qualidade_camadas'
    path=dest/'previsoes_e_resultados.csv'
    predictions=pd.read_csv(path)
    prices,returns,_=engine.load_inputs()
    dates=returns.loc['2025-01-01':'2025-12-31'].index
    signals={}
    for signal,g in predictions.groupby('signal_date'):
        signal=pd.Timestamp(signal)
        g=g.set_index('asset').loc[returns.columns]
        execution=returns.index[returns.index.get_loc(signal)+1]
        cov=returns.loc[:signal].tail(126).cov().to_numpy()+np.eye(len(g))*1e-7
        signals[execution]=dict(signal_date=signal,cov=cov,**{layer:g[layer].to_numpy() for layer in LAYERS})
    rows,curves,weights=[],[],[]
    for n in (5,10,20):
        for layer in LAYERS:
            m,c,w=engine.simulate(dates,returns,signals,n,layer,None,3.,10.)
            tags=dict(n=n,layer=layer)
            rows.append(dict(**tags,**m));curves.append(c.assign(**tags));weights.append(w.assign(**tags))
    result=pd.DataFrame(rows)
    old=pd.read_csv(OUT/'metricas.csv')
    for n in (5,10,20):
        for new,previous in [('BL','hybrid'),('Prior','prior'),('BL_XGB','xgb_gate')]:
            expected=old[(old.n==n)&old.scenario.eq('escolha_livre')&old.variant.eq(previous)].return_pct.iloc[0]
            actual=result[(result.n==n)&result.layer.eq(new)].return_pct.iloc[0]
            np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
    result.to_csv(dest/'financeiro_controlado.csv',index=False)
    pd.concat(curves).to_csv(dest/'financeiro_series.csv',index=False)
    pd.concat(weights).to_csv(dest/'financeiro_pesos.csv',index=False)
    (dest/'financeiro_manifest.json').write_text(json.dumps(dict(
        runs=len(rows),cardinalities=[5,10,20],cost_bps=10,window=126,risk_aversion=3,
        protocol='Mesma selecao por excesso/volatilidade e otimizacao com restricoes; muda somente o vetor de expectativas.',
        prices='Retornos simples diarios locais; sem uso de alvos realizados na escolha dos pesos.',
        predictions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        historical_hybrid_prior_xgb_gate_reproduced=True),indent=2),encoding='utf-8')
    print(result[['n','layer','return_pct','sharpe','max_drawdown_pct']].to_string(index=False))


if __name__=='__main__':main()
