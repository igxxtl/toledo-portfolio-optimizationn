"""Exporta o modelo conjunto fixo em todas as datas necessarias ao backtest.

Nao filtra previsoes pela maturidade do resultado: somente o treino e filtrado.
Isso recupera dezembro sem preencher previsoes e sem consultar seus alvos futuros.
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
from .experimento_melhoria import ROOT,TICKERS,FOLDS,masks
from .experimento_features import augmented
from .features_experimento import breadth_frames
from .experimento_metodologia import train

OUT=ROOT/'analise_resultados/simulacao_pool'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/xgb_matplotlib_cache'))
    from . import train_xgb as legacy
    method=ROOT/'analise_resultados/metodologia_xgb'
    protocol=json.loads((method/'protocolo.json').read_text(encoding='utf-8'))
    recipe=ROOT/'analise_resultados/evolucao_xgb/receita_pool_para_confirmacao.json'
    frozen=json.loads(recipe.read_text(encoding='utf-8'))
    if frozen['method']!='pool':raise ValueError('A receita congelada nao e o modelo conjunto fixo')
    if hashlib.sha256((method/'protocolo.json').read_bytes()).hexdigest()!=frozen['protocol_sha256']:
        raise ValueError('Protocolo do modelo mudou')
    for p,h in protocol['source_hashes'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('Entrada mudou: '+p)
    files=[legacy.DATA_DIR/f'{t}.csv' for t in TICKERS]
    prices=pd.DataFrame({t:pd.read_csv(p,parse_dates=['date']).set_index('date').close for t,p in zip(TICKERS,files)})
    breadth=breadth_frames(prices)
    ranks=legacy.build_cross_sectional_ranks(files)
    old=pd.read_csv(method/'previsoes.csv',parse_dates=['date'])
    forecasts=[];audits=[];checks=[]
    for h in [5,10,21]:
        group=protocol['groups_selected_only_in_2024'][str(h)]
        frames=[]
        for ticker in TICKERS:
            frame,pools=augmented(ticker,h,ranks,breadth,legacy)
            frames.append(frame.assign(ticker=ticker,horizon=h))
        columns=pools[group]
        pooled=pd.concat(frames,ignore_index=True).sort_values(['date','ticker']).reset_index(drop=True)
        for ticker in TICKERS:
            c='identity_'+ticker;pooled[c]=pooled.ticker.eq(ticker).astype(float);columns=columns+[c]
        for fold,start,end,phase in FOLDS:
            tr,_=masks(pooled,start,end,phase)
            te=np.flatnonzero(((pooled.date>=pd.Timestamp(start))&(pooled.date<pd.Timestamp(end))&pooled.current.notna()).to_numpy())
            predicted,audit=train(pooled,columns,group,tr,te,False)
            g=pooled.iloc[te][['ticker','horizon','date','maturity','actual']].copy()
            g['y_pred']=predicted;g['fold']=fold;g['phase']=phase
            g['max_train_maturity']=audit['max_train_maturity']
            # Reproduz TODAS as previsoes ja avaliadas na etapa anterior.
            cached=old[(old.horizon==h)&(old.fold==fold)][['ticker','date','pool']]
            compare=g.merge(cached,on=['ticker','date'],validate='one_to_one')
            if len(compare)!=len(cached):raise ValueError('Previsoes anteriores ausentes')
            error=float(np.max(np.abs(compare.y_pred-compare.pool)))
            np.testing.assert_allclose(compare.y_pred,compare.pool,atol=1e-12,rtol=0)
            checks.append(dict(horizon=h,fold=fold,existing=len(cached),exported=len(g),max_difference=error))
            audits.append(dict(horizon=h,fold=fold,first_prediction=str(g.date.min().date()),
                last_prediction=str(g.date.max().date()),n_predict=len(g),**audit))
            forecasts.append(g)
        print(f'Pool h={h}: todas as previsoes 2024/2025 exportadas e reproduzidas.',flush=True)
    data=pd.concat(forecasts,ignore_index=True)
    if data.duplicated(['ticker','horizon','date']).any():raise AssertionError('Previsoes duplicadas')
    data.to_csv(OUT/'previsoes_pool.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'auditoria_treino.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'reproducao_previsoes.csv',index=False)
    sources=[recipe,method/'protocolo.json',Path(__file__),Path(__file__).with_name('experimento_metodologia.py')]+files
    (OUT/'manifest_previsoes.json').write_text(json.dumps(dict(rows=len(data),recipe='pool fixo, sem normalizacao do alvo',
        note='Resultados futuros presentes na tabela sao somente para auditoria; o motor filtra pela maturidade a cada decisao.',
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2),encoding='utf-8')


if __name__=='__main__':main()
