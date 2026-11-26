"""Etapa seguinte: janelas, alvo normalizado e modelos conjuntos dos ativos."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
import pandas as pd
from .experimento_melhoria import ROOT,TICKERS,FOLDS,masks,fit_predict,metric
from .experimento_features import OUT as FEATURES_OUT,ANCHOR,augmented
from .features_experimento import breadth_frames,select_columns

OUT=ROOT/'analise_resultados/metodologia_xgb'
METHODS=['expansivo','movel_2a','movel_4a','vol_normalizada','pool','pool_vol']


def fit_rows(frame,tr):
    cut=int(len(tr)*.8);fit=tr[:cut];iv=tr[cut:]
    return fit[(frame.maturity.iloc[fit]<frame.date.iloc[iv[0]]).to_numpy()]


def train(frame,pool,group,tr,te,normal=False,identity=False):
    fit=fit_rows(frame,tr)
    if group=='completo':columns=pool
    else:columns,_=select_columns(frame,fit,pool)
    y=frame.actual.to_numpy(float)
    scale=np.maximum(frame.vol_21.to_numpy(float)*np.sqrt(frame.horizon.to_numpy(float)),1e-4)
    if normal:y=y/scale
    pred,audit=fit_predict(frame[columns].to_numpy(np.float32),y,frame,tr,te,ANCHOR,400,2)
    if normal:pred*=scale[te]
    return pred,dict(n_features=len(columns),**audit)


def summarize(out,features):
    data=pd.concat([pd.read_csv(p) for p in sorted((out/'cases').glob('*.csv'))],ignore_index=True)
    data.to_csv(out/'previsoes.csv',index=False)
    rows=[]
    for (ticker,h,phase),g in data.groupby(['ticker','horizon','phase']):
        for method in METHODS+['current','previous','feature_selected','zero','historical']:
            rows.append(dict(ticker=ticker,horizon=h,phase=phase,model=method,**metric(g.actual,g[method])))
    scores=pd.DataFrame(rows);scores.to_csv(out/'metricas.csv',index=False)
    choices=[];comparisons=[]
    for (ticker,h),g in scores[scores.phase.eq('validation')].groupby(['ticker','horizon']):
        best=g[g.model.isin(METHODS)].sort_values(['rmse','model']).iloc[0]
        choices.append(dict(ticker=ticker,horizon=h,chosen=best.model,rmse_validation=best.rmse))
        t=scores[(scores.ticker==ticker)&(scores.horizon==h)&scores.phase.eq('test')].set_index('model')
        selected=t.loc[best.model];old=t.loc['current']
        comparisons.append(dict(ticker=ticker,horizon=h,chosen=best.model,n=int(selected.n),
            rmse_current=old.rmse,rmse_new=selected.rmse,mae_current=old.mae,mae_new=selected.mae,
            rmse_previous=t.loc['previous'].rmse,rmse_features=t.loc['feature_selected'].rmse,
            rmse_expansivo=t.loc['expansivo'].rmse,rmse_zero=t.loc['zero'].rmse,rmse_historical=t.loc['historical'].rmse,
            hr_current=old.hr,hr_new=selected.hr,r2_current=old.r2,r2_new=selected.r2,
            skill_new=selected.skill_zero,rmse_reduction_pct=(1-selected.rmse/old.rmse)*100))
    choice=pd.DataFrame(choices);comp=pd.DataFrame(comparisons)
    choice.to_csv(out/'escolhas_2024.csv',index=False);comp.to_csv(out/'comparacao_2025.csv',index=False)
    summary=comp.groupby('horizon',as_index=False).agg(assets=('ticker','size'),n=('n','sum'),
        rmse_current=('rmse_current','mean'),rmse_previous=('rmse_previous','mean'),rmse_features=('rmse_features','mean'),
        rmse_new=('rmse_new','mean'),mae_current=('mae_current','mean'),mae_new=('mae_new','mean'),
        hr_current=('hr_current','mean'),hr_new=('hr_new','mean'),skill_new=('skill_new','mean'),
        wins_current=('rmse_reduction_pct',lambda s:int((s>0).sum())))
    summary['rmse_reduction_pct']=(1-summary.rmse_new/summary.rmse_current)*100
    summary.to_csv(out/'resumo_horizontes.csv',index=False)
    scores[scores.phase.eq('test')].groupby(['horizon','model'],as_index=False).agg(
        rmse=('rmse','mean'),mae=('mae','mean'),hr=('hr','mean'),skill_zero=('skill_zero','mean')).to_csv(out/'metodos_fixos_2025.csv',index=False)
    old=pd.read_csv(ROOT/'intel/Blacklitterman/outputs/dois_cenarios/horizontes_validacao.csv')
    old['ticker']=old.asset.str.replace('.SA','',regex=False)
    current=comp.merge(old[['ticker','horizon']],on=['ticker','horizon'],validate='one_to_one')
    current.to_csv(out/'comparacao_horizontes_atuais.csv',index=False)
    result=dict(cases=len(comp),chosen_counts=choice.chosen.value_counts().to_dict(),
        wins_current=int((comp.rmse_new<comp.rmse_current).sum()),
        wins_previous=int((comp.rmse_new<comp.rmse_previous).sum()),
        wins_features=int((comp.rmse_new<comp.rmse_features).sum()),
        wins_zero=int((comp.rmse_new<comp.rmse_zero).sum()),
        wins_historical=int((comp.rmse_new<comp.rmse_historical).sum()),
        current_horizons=dict(assets=len(current),rmse_current=float(current.rmse_current.mean()),
            rmse_new=float(current.rmse_new.mean()),hr_current=float(current.hr_current.mean()),hr_new=float(current.hr_new.mean())))
    (out/'resultado.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True);print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/xgb_matplotlib_cache'))
    from . import train_xgb as legacy
    from .experimento_features import data_audit
    prices,paths=data_audit(legacy,out);breadth=breadth_frames(prices)
    ranks=legacy.build_cross_sectional_ranks([legacy.DATA_DIR/f'{t}.csv' for t in TICKERS])
    fm=pd.read_csv(FEATURES_OUT/'metricas.csv')
    groups=['completo','enxuto','enxuto_novo','mercado_ampliado']
    # Um conjunto comum por horizonte para tornar pooling comparavel ao individual.
    # Normalizacao pelo benchmark de cada ativo impede dominancia dos mais volateis na selecao.
    ranked=fm[fm.phase.eq('validation')&fm.model.isin(groups)].groupby(['horizon','model'],as_index=False).skill_zero.mean()
    selected_groups={int(h):g.sort_values(['skill_zero','model'],ascending=[False,True]).iloc[0].model for h,g in ranked.groupby('horizon')}
    ranked.to_csv(out/'selecao_features_global_2024.csv',index=False)
    previous_choice=pd.read_csv(ROOT/'analise_resultados/melhoria_xgb/escolhas_validacao_2024.csv').set_index(['ticker','horizon']).chosen
    feature_choice=pd.read_csv(FEATURES_OUT/'escolhas_2024.csv').set_index(['ticker','horizon']).chosen
    hashes=json.loads((ROOT/'analise_resultados/melhoria_xgb/protocolo.json').read_text())['protected_baseline_hashes']
    sourcepaths=paths+[Path(__file__),FEATURES_OUT/'metricas.csv',FEATURES_OUT/'protocolo.json']
    protocol=dict(methods=METHODS,config=ANCHOR,groups_selected_only_in_2024=selected_groups,
        folds=[list(f) for f in FOLDS],rolling='2 ou 4 anos civis por data de origem; apenas alvos maturados antes do fold.',
        normalization='y/(volatilidade diaria 21 conhecida na origem * sqrt(h)); previsao reconstruida em retorno original.',
        pooling='Mesmo horizonte, vinte ativos, ordenacao por data e ticker; identidade one-hot; purga por maturidade tambem no early stopping.',
        selection='Metodo escolhido por RMSE 2024 por ativo/horizonte; sem reescolha por 2025.',
        holdout='Comparacao retrospectiva; nao existe janela inteiramente intocada no acervo atual.',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sourcepaths})
    (out/'protocolo.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    all_audits=[];started=time.monotonic()
    for h in [5,10,21]:
        group=selected_groups[h];datasets={}
        for ticker in TICKERS:
            frame,pools=augmented(ticker,h,ranks,breadth,legacy)
            frame=frame.assign(ticker=ticker,horizon=h);datasets[ticker]=(frame,pools[group])
        # Os dois modelos globais por horizonte/fold sao treinados uma unica vez.
        pooled=pd.concat([d[0] for d in datasets.values()],ignore_index=True).sort_values(['date','ticker']).reset_index(drop=True)
        pool=list(datasets[TICKERS[0]][1])
        identities=[]
        for ticker in TICKERS:
            c='identity_'+ticker;pooled[c]=pooled.ticker.eq(ticker).astype(float);identities.append(c)
        pool=pool+identities
        pool_predictions=[]
        for fold,start,end,phase in FOLDS:
            tr,te=masks(pooled,start,end,phase)
            g=pooled.iloc[te][['ticker','date']].copy()
            for method,normalized in [('pool',False),('pool_vol',True)]:
                p,a=train(pooled,pool,group,tr,te,normalized)
                g[method]=p
                all_audits.append(dict(ticker='ALL',horizon=h,fold=fold,method=method,
                    first_prediction=str(pooled.date.iloc[te].min().date()),**a))
            pool_predictions.append(g)
        global_predictions=pd.concat(pool_predictions).set_index(['ticker','date'])
        def run_case(ticker):
            frame,columns=datasets[ticker];frames=[];audits=[]
            cached=pd.read_csv(FEATURES_OUT/'cases'/f'{ticker}_h{h}.csv',parse_dates=['date']).set_index('date')
            for fold,start,end,phase in FOLDS:
                tr,te=masks(frame,start,end,phase);g=frame.iloc[te][['date','maturity','actual','current']].copy()
                base=cached.loc[g.date]
                np.testing.assert_allclose(g.actual,base.actual,atol=1e-14,rtol=0)
                g['expansivo']=base[group].to_numpy();g['previous']=base.previous.to_numpy()
                g['feature_selected']=base[feature_choice.loc[(ticker,h)]].to_numpy()
                g['historical']=base.historical.to_numpy();g['zero']=0.
                for method in ['movel_2a','movel_4a','vol_normalizada']:
                    actual_tr=tr
                    if method.startswith('movel'):
                        years=2 if method=='movel_2a' else 4
                        actual_tr=tr[(frame.date.iloc[tr]>=pd.Timestamp(start)-pd.DateOffset(years=years)).to_numpy()]
                    p,a=train(frame,columns,group,actual_tr,te,method=='vol_normalizada')
                    g[method]=p
                    audits.append(dict(ticker=ticker,horizon=h,fold=fold,method=method,
                        first_prediction=str(g.date.min().date()),**a))
                for method in ['pool','pool_vol']:
                    g[method]=global_predictions.loc[(ticker,g.date.to_list()),method].to_numpy()
                frames.append(g.assign(ticker=ticker,horizon=h,fold=fold,phase=phase))
            pd.concat(frames).to_csv(out/'cases'/f'{ticker}_h{h}.csv',index=False)
            print(f'Metodologia {ticker} h={h}: concluida; {time.monotonic()-started:.1f}s',flush=True)
            return audits
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            jobs=[executor.submit(run_case,t) for t in TICKERS]
            for job in as_completed(jobs):all_audits+=job.result()
    pd.DataFrame(all_audits).to_csv(out/'auditoria_temporal.csv',index=False)
    for name,digest in hashes.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise AssertionError('Baseline alterado')
    summarize(out,FEATURES_OUT)
    (out/'verificacao_baseline.json').write_text(json.dumps(dict(unchanged=True,audits=len(all_audits)),indent=2),encoding='utf-8')


if __name__=='__main__':main()
